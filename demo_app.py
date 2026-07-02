"""EMNLP-ready Streamlit demo for the Media Profiler.

Run with:
    .venv/bin/streamlit run demo_app.py

The app opens on cached system outputs for an instant reviewer
experience, while still allowing live URL analysis through the full pipeline.
"""

from __future__ import annotations

import html
import hashlib
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlparse


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader, avoiding an extra runtime dependency."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

import streamlit as st

from article_cache import ArticleCache
from evaluators import SystemRunner

CACHED_JSONL = "results/cached/gpt-5-mini-2025-08-07_system.jsonl"
SUMMARY_JSON = "results/summary.json"
FC_REATTRIBUTION_JSON = "results/cached/fact_check_reattribution.json"
LOADED_LANGUAGE_JSON = "results/cached/loaded_language_sources.json"
DEFAULT_MODEL = "gpt-5-mini-2025-08-07"
LIVE_RUN_CACHE_DIR = os.environ.get("MP_LIVE_RUN_CACHE_DIR", "results/live_runs")
LIVE_RUN_CACHE_VERSION = 1

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Media Profiler", layout="wide", initial_sidebar_state="expanded")


# ---------------------------------------------------------------------------
# Cached data
# ---------------------------------------------------------------------------

@st.cache_data
def load_cached_outlets() -> dict[str, dict]:
    """Load cached outlet records, indexed by outlet name."""
    outlets: dict[str, dict] = {}
    if not os.path.exists(CACHED_JSONL):
        return outlets
    with open(CACHED_JSONL, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw = rec.get("raw_output")
            if isinstance(raw, dict) and raw.get("name"):
                src = raw.get("source_url", "")
                if src and "i0.wp.com" in src:
                    continue
                outlets[raw["name"]] = raw
    return outlets


@st.cache_data
def load_benchmark_summary() -> dict[str, dict]:
    """Load only the public aggregate benchmark table."""
    if not os.path.exists(SUMMARY_JSON):
        return {}
    try:
        with open(SUMMARY_JSON, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    summary = data.get("agg")
    return summary if isinstance(summary, dict) else {}


@st.cache_data
def load_leaderboard_data() -> list[dict]:
    """Per-outlet prediction vs. ground-truth data for the leaderboard."""
    records: list[dict] = []
    if not os.path.exists(CACHED_JSONL):
        return records
    with open(CACHED_JSONL, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = rec.get("name") or (rec.get("raw_output") or {}).get("name", "")
            if not name:
                continue
            gb = int(rec.get("gold_bias_ordinal") or 0)
            pb = int(rec.get("pred_bias_ordinal") or 0)
            gf = int(rec.get("gold_factuality_ordinal") or 0)
            pf = int(rec.get("pred_factuality_ordinal") or 0)
            records.append({
                "name": name,
                "country": rec.get("gold_country") or "",
                "media_type": rec.get("gold_media_type") or "",
                "gold_bias": rec.get("gold_bias_ordinal_label") or "",
                "pred_bias": rec.get("pred_bias_ordinal_label") or "",
                "bias_delta": abs(gb - pb),
                "gold_factuality": rec.get("gold_factuality_ordinal_label") or "",
                "pred_factuality": rec.get("pred_factuality_ordinal_label") or "",
                "fact_delta": abs(gf - pf),
            })
    return records


def _load_sidecar(path: str) -> dict[str, dict]:
    """Load a post-processing sidecar JSON keyed by outlet name (empty if absent)."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


@st.cache_data
def load_fc_reattribution() -> dict[str, dict]:
    """Per-outlet failed-fact-check re-attribution (scripts/recheck_failed_fact_checks.py).

    Maps outlet name -> {domain, findings:[{url, summary, verdict, claim_source,
    published_by_outlet, claim_source_domain, ...}], failed_published_by_outlet, ...}.
    Absent => the demo falls back to the legacy un-gated behavior.
    """
    return _load_sidecar(FC_REATTRIBUTION_JSON)


@st.cache_data
def load_loaded_language_sources() -> dict[str, list]:
    """Per-outlet loaded-language examples sourced from cached articles
    (scripts/extract_loaded_language_sources.py): name -> [{quote, art_url, art_title}]."""
    data = _load_sidecar(LOADED_LANGUAGE_JSON)
    return {k: v for k, v in data.items() if isinstance(v, list)}


# ---------------------------------------------------------------------------
# Live run result cache
# ---------------------------------------------------------------------------

def normalize_live_cache_url(url: str) -> dict[str, str] | None:
    """Return domain-level cache URL parts, or None for invalid input."""
    raw = (url or "").strip()
    if not raw:
        return None
    candidate = raw if re.match(r"^https?://", raw, re.IGNORECASE) else f"https://{raw}"
    try:
        parsed = urlparse(candidate)
    except ValueError:
        return None
    host = (parsed.hostname or "").lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.[a-z0-9-]{2,}", host):
        return None
    return {"domain": host, "canonical_url": f"https://{host}"}


def _live_cache_key(url: str) -> dict[str, object] | None:
    normalized = normalize_live_cache_url(url)
    if normalized is None:
        return None
    return {
        "cache_version": LIVE_RUN_CACHE_VERSION,
        "domain": normalized["domain"],
        "canonical_url": normalized["canonical_url"],
        "model": DEFAULT_MODEL,
        "runner": "SystemRunner",
        "benchmark_mode": False,
        "use_synthesis": True,
    }


def _live_cache_path(key: dict[str, object]) -> str:
    encoded = json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return os.path.join(LIVE_RUN_CACHE_DIR, f"{digest}.json")


def load_live_cached_run(url: str) -> dict | None:
    """Load a previously saved live report for this URL/model/pipeline."""
    key = _live_cache_key(url)
    if key is None:
        return None
    path = _live_cache_path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("[LiveCache] Ignoring unreadable cache file %s: %s", path, e)
        return None
    if payload.get("cache_version") != LIVE_RUN_CACHE_VERSION or payload.get("key") != key:
        logger.warning("[LiveCache] Ignoring stale or mismatched cache file %s", path)
        return None
    result = payload.get("result")
    return result if isinstance(result, dict) else None


def save_live_cached_run(url: str, result: dict) -> bool:
    """Persist a successful live run result using an atomic JSON write."""
    if not isinstance(result, dict) or not result:
        return False
    key = _live_cache_key(url)
    if key is None:
        return False
    os.makedirs(LIVE_RUN_CACHE_DIR, exist_ok=True)
    path = _live_cache_path(key)
    payload = {
        "cache_version": LIVE_RUN_CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "key": key,
        "result": result,
    }
    tmp_path = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp_path, path)
        logger.info("[LiveCache] Saved live run: %s", path)
        return True
    except OSError as e:
        logger.warning("[LiveCache] Failed to save live run %s: %s", path, e)
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
        return False


# ---------------------------------------------------------------------------
# Live progress hook
# ---------------------------------------------------------------------------

PROGRESS_STEPS = [
    ("Got ", "Loaded articles", 1),
    ("Launching parallel", "Launching analyzer pool", 2),
    ("[Done] Traffic", "Traffic and longevity complete", 3),
    ("[Done] Media type", "Media type complete", 4),
    ("[Done] Fact check", "Fact-check search complete", 5),
    ("[Done] Transparency", "Transparency complete", 6),
    ("[Done] History", "History and ownership complete", 7),
    ("[Done] Opinion", "Opinion split complete", 8),
    ("[Done] Sourcing", "Sourcing analysis complete", 9),
    ("[Done] Pseudoscience", "Pseudoscience check complete", 10),
    ("[Done] One-sidedness", "One-sidedness analysis complete", 11),
    ("Running editorial bias", "Editorial bias running", 12),
    ("[Done] Editorial bias", "Editorial bias complete", 13),
    ("synthesi", "Synthesizing report", 14),
]
TOTAL_STEPS = 15


class ProgressLogHandler(logging.Handler):
    """Capture matched pipeline milestones from log messages."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self._lock = threading.Lock()
        self.label = "Starting analysis"
        self.step = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return
        for needle, label, step in PROGRESS_STEPS:
            if needle in msg:
                with self._lock:
                    if step >= self.step:
                        self.step = step
                        self.label = label
                break

    def snapshot(self) -> tuple[str, int]:
        with self._lock:
            return self.label, self.step


def derive_name_from_url(url: str) -> str:
    parsed = urlparse(url if url.startswith("http") else f"https://{url}")
    host = (parsed.netloc or parsed.path).replace("www.", "")
    base = host.split("/")[0].split(".")[0]
    return base.capitalize() if base else "Unknown Outlet"


def run_live(url: str) -> dict | None:
    """Run the full system pipeline against a URL, with a visible progress loop."""
    if not url.startswith("http"):
        url = "https://" + url

    runner = SystemRunner(
        model_name=DEFAULT_MODEL,
        article_cache=ArticleCache(),
        benchmark_mode=False,
        use_synthesis=True,
    )
    item = {"name": derive_name_from_url(url), "source_url": url}
    handler = ProgressLogHandler()
    root = logging.getLogger()
    prev_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    result_box: dict = {"value": None, "error": None}

    def _worker() -> None:
        try:
            result_box["value"] = runner.run(item)
        except Exception as e:  # noqa: BLE001
            result_box["error"] = f"{type(e).__name__}: {e}"

    thread = threading.Thread(target=_worker, daemon=True)
    start = time.time()
    thread.start()
    progress = st.progress(0, text="Starting analysis")
    status_slot = st.empty()
    try:
        while thread.is_alive():
            label, step = handler.snapshot()
            elapsed = int(time.time() - start)
            pct = min(int((step / TOTAL_STEPS) * 100), 95)
            progress.progress(pct, text=f"{label} - {elapsed}s elapsed")
            status_slot.markdown(
                f"""
<div class="run-status">
  <div><span class="eyebrow">Live pipeline</span><h4>{html.escape(label)}</h4></div>
  <strong>{elapsed}s</strong>
</div>
""",
                unsafe_allow_html=True,
            )
            time.sleep(0.4)
        thread.join(timeout=5)
        elapsed_total = round(time.time() - start, 1)
        progress.progress(100, text=f"Done in {elapsed_total}s")
        status_slot.markdown(
            f"""
<div class="run-status complete">
  <div><span class="eyebrow">Live pipeline</span><h4>Analysis complete</h4></div>
  <strong>{elapsed_total}s</strong>
</div>
""",
            unsafe_allow_html=True,
        )
    finally:
        root.removeHandler(handler)
        root.setLevel(prev_level)

    if result_box["error"]:
        st.error(
            "Analysis failed. The site may block scraping, return no parseable "
            f"articles, or the model/search backend may be unavailable. Details: {result_box['error']}"
        )
        return None

    result = result_box["value"]
    if result is not None:
        result["_duration_seconds"] = elapsed_total
    return result


# ---------------------------------------------------------------------------
# Styling and rendering helpers
# ---------------------------------------------------------------------------

THEME = {
    "bg": "#f6f7f4",
    "surface": "#ffffff",
    "surface2": "#f0f3f1",
    "text": "#202124",
    "muted": "#5d6461",
    "line": "#d7ded9",
    "accent": "#0f766e",
    "accent2": "#b4563d",
    "shadow": "0 18px 44px rgba(32, 33, 36, 0.10)",
    "display_font": "Georgia, 'Times New Roman', serif",
}

BIAS_COLOR = {
    "EXTREME LEFT": "#8b1e3f",
    "FAR LEFT": "#b4234a",
    "LEFT": "#d05d39",
    "LEFT-CENTER": "#3f7cac",
    "LEAST BIASED": "#2f7d4f",
    "RIGHT-CENTER": "#3f7cac",
    "RIGHT": "#d05d39",
    "FAR RIGHT": "#b4234a",
    "EXTREME RIGHT": "#8b1e3f",
    "QUESTIONABLE": "#8b1e3f",
    "PRO-SCIENCE": "#2f7d4f",
    "CONSPIRACY": "#8b1e3f",
    "SATIRE": "#7a4aa0",
}

FACT_COLOR = {
    "VERY HIGH": "#236b49",
    "HIGH": "#236b49",
    "MOSTLY FACTUAL": "#0f766e",
    "MIXED": "#9a6a16",
    "LOW": "#b44232",
    "VERY LOW": "#8b1e3f",
}

CRED_COLOR = {
    "HIGH CREDIBILITY": "#236b49",
    "MEDIUM CREDIBILITY": "#9a6a16",
    "LOW CREDIBILITY": "#b44232",
}

ANALYZERS = [
    ("TrafficLongevity", "Domain age and traffic rank"),
    ("MediaType", "Outlet format classification"),
    ("Opinion", "News, opinion, and satire split"),
    ("EditorialBias", "Political stance across policy domains"),
    ("FactCheckSearcher", "IFCN-style failed fact-check lookup"),
    ("Sourcing", "Citation and source quality"),
    ("Pseudoscience", "Health, science, and conspiracy signals"),
    ("Transparency", "Ownership, authorship, and funding disclosure"),
    ("OneSidedness", "Propaganda and one-sided framing"),
]


def inject_theme() -> None:
    t = THEME
    st.markdown(
        f"""
<style>
:root {{
  --mp-bg:{t['bg']}; --mp-surface:{t['surface']}; --mp-surface-2:{t['surface2']};
  --mp-text:{t['text']}; --mp-muted:{t['muted']}; --mp-line:{t['line']};
  --mp-accent:{t['accent']}; --mp-accent-2:{t['accent2']};
  --mp-shadow:{t['shadow']}; --mp-display-font:{t['display_font']};
  --mp-ui-font:'Inter','Source Sans Pro',Arial,sans-serif;
}}
.stApp {{ background:linear-gradient(180deg,rgba(255,255,255,.72),rgba(255,255,255,0) 280px),var(--mp-bg); color:var(--mp-text); }}
.block-container {{ max-width:1420px; padding-top:1.6rem; padding-bottom:3rem; }}
section[data-testid="stSidebar"] {{ background:var(--mp-surface); border-right:1px solid var(--mp-line); }}
h1,h2,h3 {{ color:var(--mp-text); letter-spacing:0; }}
.mp-hero {{ background:var(--mp-surface); border:1px solid var(--mp-line); border-radius:8px; box-shadow:var(--mp-shadow); padding:24px 26px 22px; margin-bottom:16px; }}
.mp-hero h1 {{ font-family:var(--mp-display-font); font-size:clamp(2rem,3.4vw,3.65rem); line-height:1.04; margin:0 0 10px; }}
.mp-hero p {{ color:var(--mp-muted); font-family:var(--mp-ui-font); font-size:1rem; line-height:1.55; max-width:980px; margin:0; }}
.eyebrow {{ color:var(--mp-accent); display:block; font-family:var(--mp-ui-font); font-size:.72rem; font-weight:800; letter-spacing:.08em; margin-bottom:8px; text-transform:uppercase; }}
.metric-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:12px 0 18px; }}
.metric-card,.profile-card,.evidence-card,.run-status,.timeline-item {{ background:var(--mp-surface); border:1px solid var(--mp-line); border-radius:8px; box-shadow:0 10px 28px rgba(31,41,55,.06); }}
.metric-card {{ min-height:114px; padding:16px; }}
.metric-card .label {{ color:var(--mp-muted); display:block; font-family:var(--mp-ui-font); font-size:.76rem; font-weight:700; text-transform:uppercase; }}
.metric-card .value {{ color:var(--mp-text); display:block; font-size:2rem; font-weight:800; line-height:1.15; margin-top:8px; }}
.metric-card .caption {{ color:var(--mp-muted); display:block; font-size:.84rem; line-height:1.35; margin-top:7px; }}
.profile-card {{ padding:18px; min-height:100%; }}
.profile-card p {{ color:var(--mp-muted); line-height:1.58; margin:0; }}
.badge-row {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }}
.rating-pill {{ border-radius:999px; color:#fff; display:inline-flex; align-items:center; font-family:var(--mp-ui-font); font-size:.77rem; font-weight:800; min-height:28px; padding:5px 11px; white-space:nowrap; }}
.score-block {{ margin-top:14px; }}
.score-label {{ color:var(--mp-muted); display:flex; font-family:var(--mp-ui-font); font-size:.82rem; font-weight:700; justify-content:space-between; margin-bottom:8px; }}
.score-track {{ background:var(--mp-surface-2); border:1px solid var(--mp-line); border-radius:999px; height:12px; overflow:hidden; }}
.score-fill {{ border-radius:999px; height:100%; min-width:2px; }}
.score-axis {{ color:var(--mp-muted); display:flex; font-family:var(--mp-ui-font); font-size:.76rem; justify-content:space-between; margin-top:6px; }}
.score-axis-3 {{ color:var(--mp-muted); display:flex; font-family:var(--mp-ui-font); font-size:.76rem; justify-content:space-between; margin-top:6px; }}
.score-track-bi {{ background:var(--mp-surface-2); border:1px solid var(--mp-line); border-radius:999px; height:12px; overflow:hidden; position:relative; }}
.score-fill-right {{ border-radius:0 999px 999px 0; height:100%; position:absolute; left:50%; }}
.score-fill-left {{ border-radius:999px 0 0 999px; height:100%; position:absolute; right:50%; }}
.score-center-mark {{ background:var(--mp-muted); height:100%; left:calc(50% - 1px); position:absolute; top:0; width:2px; z-index:2; }}
.evidence-card {{ margin-bottom:10px; padding:15px; }}
.evidence-card a {{ color:var(--mp-accent); font-weight:750; text-decoration:none; }}
.evidence-card p {{ color:var(--mp-muted); line-height:1.45; margin:8px 0 0; }}
.claim-meta {{ color:var(--mp-muted); display:flex; flex-wrap:wrap; gap:8px; font-family:var(--mp-ui-font); font-size:.78rem; margin-bottom:8px; }}
.claim-pill {{ background:var(--mp-surface-2); border:1px solid var(--mp-line); border-radius:999px; padding:3px 9px; }}
.article-list {{ margin-top:10px; }}
.small-muted {{ color:var(--mp-muted); font-size:.86rem; line-height:1.45; }}
.timeline-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; margin:8px 0 18px; }}
.timeline-item {{ padding:13px 14px; }}
.timeline-item strong {{ display:block; font-size:.95rem; margin-bottom:4px; }}
.timeline-item span {{ color:var(--mp-muted); display:block; font-size:.82rem; line-height:1.35; }}
.run-status {{ align-items:center; display:flex; justify-content:space-between; margin:8px 0 10px; padding:16px 18px; }}
.run-status h4 {{ margin:0; }} .run-status strong {{ color:var(--mp-accent); font-size:1.4rem; }} .complete {{ border-color:rgba(47,125,79,.38); }}
div[data-testid="stMetric"] {{ background:var(--mp-surface); border:1px solid var(--mp-line); border-radius:8px; padding:12px 14px; }}
@media(max-width:980px) {{ .metric-grid,.timeline-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
@media(max-width:640px) {{ .metric-grid,.timeline-grid {{ grid-template-columns:1fr; }} .mp-hero {{ padding:19px 18px; }} .mp-hero h1 {{ font-size:2rem; }} }}
</style>
""",
        unsafe_allow_html=True,
    )


def _coalesce(value, fallback: str = "-") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _esc(value) -> str:
    return html.escape(_coalesce(value), quote=True)


def _format_score(value) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.2f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _label_color(label: str, palette: dict[str, str], fallback: str = "#5d6461") -> str:
    return palette.get(_coalesce(label).upper(), fallback)


def _badge(label: str, palette: dict[str, str]) -> str:
    color = _label_color(label, palette)
    return f'<span class="rating-pill" style="background:{color};">{_esc(label).upper()}</span>'


def _metric_card(label: str, value: str, caption: str = "") -> str:
    return (
        '<div class="metric-card">'
        f'<span class="label">{_esc(label)}</span>'
        f'<span class="value">{_esc(value)}</span>'
        f'<span class="caption">{_esc(caption)}</span>'
        '</div>'
    )


def _host_from_url(url: str | None) -> str:
    if not url:
        return "-"
    parsed = urlparse(url if url.startswith("http") else f"https://{url}")
    host = parsed.netloc or parsed.path
    return host.replace("www.", "").split("/")[0] or "-"


def _default_outlet_name(cached: dict[str, dict]) -> str | None:
    if not cached:
        return None
    for name in cached:
        if name.lower() == "the guardian":
            return name
    for name in cached:
        if "guardian" in name.lower():
            return name
    return sorted(cached.keys())[0]


def _unique_values(cached: dict[str, dict], key: str) -> list[str]:
    values = {_coalesce(row.get(key), "Unknown") for row in cached.values()}
    return sorted(v for v in values if v and v != "-")


def _matches_filter(value: str, selected: str) -> bool:
    return selected == "All" or _coalesce(value, "Unknown") == selected


def filtered_outlet_names(cached: dict[str, dict], query: str, bias: str, factual: str, cred: str, media: str) -> list[str]:
    q = query.strip().lower()
    names: list[str] = []
    for name, row in cached.items():
        haystack = " ".join(
            [name, _coalesce(row.get("source_url")), _coalesce(row.get("bias_rating")),
             _coalesce(row.get("factual_reporting")), _coalesce(row.get("credibility_rating")),
             _coalesce(row.get("media_type"))]
        ).lower()
        if q and q not in haystack:
            continue
        if not _matches_filter(row.get("bias_rating"), bias):
            continue
        if not _matches_filter(row.get("factual_reporting"), factual):
            continue
        if not _matches_filter(row.get("credibility_rating"), cred):
            continue
        if not _matches_filter(row.get("media_type"), media):
            continue
        names.append(name)
    return sorted(names)


def _find_quick_pick(cached: dict[str, dict], kind: str) -> str | None:
    candidates = sorted(cached.items())
    if kind == "Balanced":
        for name, row in candidates:
            if _coalesce(row.get("bias_rating")).upper() == "LEAST BIASED" and _coalesce(row.get("credibility_rating")).upper() == "HIGH CREDIBILITY":
                return name
        for name, row in candidates:
            if _coalesce(row.get("bias_rating")).upper() == "LEAST BIASED":
                return name
    if kind == "Left":
        labels = {"LEFT", "LEFT-CENTER"}
    elif kind == "Far Left":
        labels = {"FAR LEFT", "EXTREME LEFT"}
    elif kind == "Right":
        labels = {"RIGHT", "RIGHT-CENTER"}
    elif kind == "Far Right":
        labels = {"FAR RIGHT", "EXTREME RIGHT"}
    elif kind == "Pro-science":
        labels = {"PRO-SCIENCE"}
    else:
        labels = set()
    if labels:
        for name, row in candidates:
            if _coalesce(row.get("bias_rating")).upper() in labels:
                return name
    if kind == "Low factuality":
        for name, row in candidates:
            if _coalesce(row.get("factual_reporting")).upper() in {"LOW", "VERY LOW"}:
                return name
    if kind == "High factuality":
        for name, row in candidates:
            if _coalesce(row.get("factual_reporting")).upper() in {"HIGH", "VERY HIGH"}:
                return name
    if kind == "Low credibility":
        for name, row in candidates:
            if _coalesce(row.get("credibility_rating")).upper() == "LOW CREDIBILITY":
                return name
    return None


def _apply_report(name: str, cached: dict[str, dict]) -> None:
    """Sync report state to ``name`` without touching the selectbox's widget key.

    Why: Streamlit forbids assigning to a widget's session_state key after the
    widget has been instantiated in the current run. The script-bottom branch
    that reacts to a selectbox change runs after the widget is built, so it
    must use this helper rather than _select_report.
    """
    row = cached.get(name)
    if row is None:
        return
    st.session_state["report"] = row
    st.session_state["last_picked"] = name
    st.session_state["report_source"] = "cached"


def _select_report(name: str, cached: dict[str, dict]) -> None:
    """Select an outlet from a callback (on_click) or pre-widget code path."""
    if cached.get(name) is None:
        return
    st.session_state["picked_outlet"] = name
    _apply_report(name, cached)


def _reset_to_default(cached: dict[str, dict]) -> None:
    default = _default_outlet_name(cached)
    if default:
        _select_report(default, cached)


def _section_text(rd: dict, key: str, fallback: str = "Not available in this run.") -> str:
    return _coalesce(rd.get(key), fallback)


def _count_system_wins(summary: dict[str, dict]) -> int:
    higher = {"factscore_precision", "factscore_recall", "meteor", "rougeL", "fc_det"}
    lower = {"error_rate", "bias_mae", "fact_mae"}
    wins = 0
    for metric in sorted(higher | lower):
        values = {m: r.get(metric) for m, r in summary.items() if isinstance(r, dict) and isinstance(r.get(metric), (int, float))}
        if "system" not in values:
            continue
        best = max(values.values()) if metric in higher else min(values.values())
        wins += int(values["system"] == best)
    return wins


def render_benchmark_strip(summary: dict[str, dict]) -> None:
    system = summary.get("system", {})
    if not system:
        st.info("Benchmark summary is not available.")
        return
    cards = [
        ("Bias MAE", f"{system.get('bias_mae', 0):.2f}", "Lower is better"),
        ("Factuality MAE", f"{system.get('fact_mae', 0):.2f}", "Lower is better"),
        ("FC Detection", f"{system.get('fc_det', 0):.0%}", "Applicable failed fact-check cases"),
        ("Metric wins", f"{_count_system_wins(summary)}/8", "System mode on the public subset"),
    ]
    st.markdown('<div class="metric-grid">' + "".join(_metric_card(*card) for card in cards) + "</div>", unsafe_allow_html=True)


def render_hero(rd: dict, source_label: str) -> None:
    name = _coalesce(rd.get("name"), "Media Profiler")
    src = rd.get("source_url") or rd.get("mbfc_url")
    host = _host_from_url(src)
    duration = rd.get("_duration_seconds")
    if duration and source_label == "Saved live system output":
        run_meta = f"{source_label}; original run completed in {_format_score(duration)}s"
    elif duration:
        run_meta = f"Live run completed in {_format_score(duration)}s"
    else:
        run_meta = source_label
    if rd.get("evidence_sufficient") is False:
        reason = _esc(rd.get("insufficient_evidence_reason") or "insufficient evidence")
        verdict_html = f'<span class="rating-pill" style="background:#8a8f98;">Verdict withheld — {reason}</span>'
    else:
        verdict_html = (
            _badge(rd.get("bias_rating"), BIAS_COLOR)
            + _badge(rd.get("factual_reporting"), FACT_COLOR)
            + _badge(rd.get("credibility_rating"), CRED_COLOR)
        )
    st.markdown(
        f"""
<div class="mp-hero">
  <span class="eyebrow">Media Profiler system demonstration</span>
  <h1>{_esc(name)}</h1>
  <p>A 9-analyzer evidence workbench for outlet-level bias, factuality, credibility,
  ownership, sourcing, and failed fact-check signals. Current profile:
  <strong>{_esc(host)}</strong>. {html.escape(run_meta)}</p>
  <div class="badge-row">
    {verdict_html}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_score_bar(title: str, value, min_value: float, max_value: float, left: str, right: str, color: str) -> None:
    numeric = _as_float(value)
    if numeric is None:
        st.markdown(f"**{title}:** -")
        return
    bounded = max(min(numeric, max_value), min_value)
    pct = ((bounded - min_value) / (max_value - min_value)) * 100
    st.markdown(
        f"""
<div class="score-block">
  <div class="score-label"><span>{_esc(title)}</span><span>{_esc(_format_score(value))}</span></div>
  <div class="score-track"><div class="score-fill" style="width:{pct:.1f}%;background:{color};"></div></div>
  <div class="score-axis"><span>{_esc(left)}</span><span>{_esc(right)}</span></div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_bias_bar(title: str, value, color: str) -> None:
    """Bidirectional bar: fills right from center for positive scores, left for negative."""
    numeric = _as_float(value)
    if numeric is None:
        st.markdown(f"**{title}:** -")
        return
    bounded = max(min(numeric, 10.0), -10.0)
    pct = abs(bounded) / 10.0 * 50.0
    if bounded >= 0:
        fill = f'<div class="score-fill-right" style="width:{pct:.1f}%;background:{color};"></div>'
    else:
        fill = f'<div class="score-fill-left" style="width:{pct:.1f}%;background:{color};"></div>'
    st.markdown(
        f'<div class="score-block">'
        f'<div class="score-label"><span>{_esc(title)}</span><span>{_esc(_format_score(value))}</span></div>'
        f'<div class="score-track-bi">{fill}<div class="score-center-mark"></div></div>'
        f'<div class="score-axis-3"><span>Left −10</span><span>0</span><span>Right +10</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_factuality_bar(title: str, value, color: str) -> None:
    """Left-to-right risk bar with a 3-point axis (0 / Mixed 5 / High risk 10)."""
    numeric = _as_float(value)
    if numeric is None:
        st.markdown(f"**{title}:** -")
        return
    bounded = max(min(numeric, 10.0), 0.0)
    pct = bounded / 10.0 * 100.0
    st.markdown(
        f'<div class="score-block">'
        f'<div class="score-label"><span>{_esc(title)}</span><span>{_esc(_format_score(value))}</span></div>'
        f'<div class="score-track"><div class="score-fill" style="width:{pct:.1f}%;background:{color};"></div></div>'
        f'<div class="score-axis-3"><span>Low risk 0</span><span>Mixed 5</span><span>High risk 10</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


EVIDENCE_STATUS_DIRECT = "Direct evidence"
EVIDENCE_STATUS_INDIRECT = "Indirect evidence"
EVIDENCE_STATUS_DOMAIN = "Domain-only match"
EVIDENCE_STATUS_BAD = "Bad match"
EVIDENCE_STATUS_NOT_ASSESSED = "Not assessed"
EVIDENCE_STATUSES = {
    EVIDENCE_STATUS_DIRECT,
    EVIDENCE_STATUS_INDIRECT,
    EVIDENCE_STATUS_DOMAIN,
    EVIDENCE_STATUS_BAD,
    EVIDENCE_STATUS_NOT_ASSESSED,
}

_POLICY_COVERAGE_LABELS = {
    "Economic Policy": "Economic development / policy coverage",
    "Immigration": "Immigration / mobility policy coverage",
    "Foreign Policy": "Official diplomacy / foreign policy coverage",
    "Environmental Policy": "Environmental / resource policy coverage",
    "Education": "Education / innovation policy coverage",
    "Healthcare": "Healthcare policy coverage",
    "Gun Rights": "Public safety / gun policy coverage",
    "Social Issues": "Social policy coverage",
}

PROPAGANDA_TECHNIQUE_DESCRIPTIONS = {
    "Appeal_to_Authority": "Citing authority figures to support claims",
    "Appeal_to_fear-prejudice": "Using fear or prejudice to influence",
    "Bandwagon,Reductio_ad_Hitlerum": "Everyone-does-it appeals or Nazi comparisons",
    "Black-and-White_Fallacy": "Presenting only two choices",
    "Causal_Oversimplification": "Oversimplifying cause-effect",
    "Doubt": "Questioning credibility without evidence",
    "Exaggeration,Minimisation": "Overstating or understating facts",
    "Flag-Waving": "Appealing to patriotism or nationalism",
    "Loaded_Language": "Using emotionally charged words",
    "Name_Calling,Labeling": "Using derogatory labels",
    "Repetition": "Repeating messages for emphasis",
    "Slogans": "Using catchy phrases",
    "Thought-terminating_Cliches": "Phrases discouraging critical thinking",
    "Whataboutism,Straw_Men,Red_Herring": "Deflection tactics",
}

FAILED_FACT_CHECK_VERDICTS = {"FALSE", "MOSTLY FALSE", "PANTS ON FIRE", "MISLEADING"}


def _normalize_evidence_status(value, indirect: bool = False) -> str:
    text = _coalesce(value, "").strip()
    if text in EVIDENCE_STATUSES:
        return text
    lowered = text.casefold()
    aliases = {
        "direct": EVIDENCE_STATUS_DIRECT,
        "direct evidence": EVIDENCE_STATUS_DIRECT,
        "indirect": EVIDENCE_STATUS_INDIRECT,
        "indirect evidence": EVIDENCE_STATUS_INDIRECT,
        "domain": EVIDENCE_STATUS_DOMAIN,
        "domain-only": EVIDENCE_STATUS_DOMAIN,
        "domain-only match": EVIDENCE_STATUS_DOMAIN,
        "bad": EVIDENCE_STATUS_BAD,
        "bad match": EVIDENCE_STATUS_BAD,
        "not assessed": EVIDENCE_STATUS_NOT_ASSESSED,
    }
    if lowered in aliases:
        return aliases[lowered]
    if indirect:
        return EVIDENCE_STATUS_INDIRECT
    return ""


def _default_status_reason(status: str) -> str:
    return {
        EVIDENCE_STATUS_DIRECT: "The exact span directly supports the narrow observable claim.",
        EVIDENCE_STATUS_INDIRECT: "The span is related to the claim but requires interpretation, so it is not used in the visible rating rationale.",
        EVIDENCE_STATUS_DOMAIN: "The span identifies the topic/domain but does not directly prove the claimed framing.",
        EVIDENCE_STATUS_BAD: "The span is missing, mismatched, or insufficient for the claim.",
        EVIDENCE_STATUS_NOT_ASSESSED: "The sample lacks the content type needed to assess this claim.",
    }.get(status, "Evidence status was not assigned by the analyzer.")


def _status_rating_supporting(status: str) -> bool:
    return status == EVIDENCE_STATUS_DIRECT


def _status_sort_key(status: str) -> int:
    order = {
        EVIDENCE_STATUS_DIRECT: 0,
        EVIDENCE_STATUS_DOMAIN: 1,
        EVIDENCE_STATUS_INDIRECT: 2,
        EVIDENCE_STATUS_NOT_ASSESSED: 3,
        EVIDENCE_STATUS_BAD: 4,
    }
    return order.get(status, 9)


def _policy_coverage_label(label_or_domain: str) -> str:
    text = _coalesce(label_or_domain, "Policy")
    if ":" in text:
        text = text.split(":", 1)[1].strip()
    if "(" in text:
        text = text.split("(", 1)[0].strip()
    return _POLICY_COVERAGE_LABELS.get(text, f"{text} coverage" if text else "Policy-domain coverage")


def _looks_like_policy_framing(label: str) -> bool:
    return _coalesce(label, "").startswith("Policy-domain framing:")


def _card_evidence_status(card: dict) -> str:
    statuses = [
        _normalize_evidence_status(item.get("evidence_status"), bool(item.get("indirect")))
        for item in card.get("evidence_items") or []
    ]
    statuses = [s for s in statuses if s]
    card_status = _normalize_evidence_status(card.get("evidence_status"))
    if card_status:
        statuses.append(card_status)
    if not statuses:
        return EVIDENCE_STATUS_BAD
    return sorted(statuses, key=_status_sort_key)[0]


def _card_rating_supporting(card: dict) -> bool:
    return any(
        _normalize_evidence_status(item.get("evidence_status"), bool(item.get("indirect"))) == EVIDENCE_STATUS_DIRECT
        for item in card.get("evidence_items") or []
    )

def _claim_evidence_ids(card: dict) -> list[str]:
    ids = []
    for item in card.get("evidence_items") or []:
        evidence_id = _coalesce(item.get("evidence_id"), "")
        if evidence_id and evidence_id != "-" and evidence_id not in ids:
            ids.append(evidence_id)
    return ids


def render_auditable_synthesis(rd: dict) -> None:
    details = _evidence_details_for_report(rd, rd.get("evidence_sources") or [])
    claim_cards = details.get("claim_cards") or []
    direct_cards = [card for card in claim_cards if card.get("rating_supporting")]
    context_cards = [card for card in claim_cards if not card.get("rating_supporting")]

    st.markdown("### Auditable Report")
    st.markdown("#### A. Category Background")
    desc = rd.get("bias_category_description")
    if desc:
        st.markdown(f"Category background, not outlet-specific evidence: {_esc(desc)}")
    else:
        st.markdown("Category background, not outlet-specific evidence: no category description was attached to this run.")

    st.markdown("#### B. Outlet-Specific Observed Evidence")
    st.markdown("##### Rating-supporting direct evidence")
    if not direct_cards:
        st.info("No direct evidence cards are available for the visible rating rationale.")
    for card in direct_cards[:7]:
        ids = ", ".join(_claim_evidence_ids(card)) or "no evidence ID"
        allowed = _clean_evidence_text(card.get("final_wording_allowed"), 520)
        st.markdown(f"- {allowed} ({ids})")

    st.markdown("##### Context / not rating-supporting")
    if not context_cards:
        st.markdown("No indirect, domain-only, bad-match, or not-assessed cards were produced.")
    for card in context_cards[:7]:
        ids = ", ".join(_claim_evidence_ids(card)) or "no evidence ID"
        allowed = _clean_evidence_text(card.get("final_wording_allowed"), 420)
        status = _coalesce(card.get("evidence_status"), EVIDENCE_STATUS_BAD)
        st.markdown(f"- [{status}] {allowed} ({ids})")

    st.markdown("#### C. Final Synthesis")
    synthesis_parts = []
    for card in direct_cards:
        ids = ", ".join(_claim_evidence_ids(card))
        allowed = _clean_evidence_text(card.get("final_wording_allowed"), 220)
        if ids and allowed:
            synthesis_parts.append(f"{allowed} ({ids})")
        if len(synthesis_parts) >= 4:
            break
    if synthesis_parts:
        st.markdown("Evidence-anchored direct signals contributing to the visible profile: " + "; ".join(synthesis_parts) + ".")
    else:
        st.markdown("No direct evidence-supported synthesis was generated from this cached run.")
    st.caption("Displayed ratings are the existing pipeline outputs; evidence-status gating does not recalculate scores.")

    with st.expander("Legacy synthesized prose (not the audited evidence view)", expanded=False):
        st.markdown("#### Overall Summary")
        st.markdown(_section_text(rd, "overall_summary"))
        n1, n2 = st.columns(2)
        with n1:
            st.markdown("#### History")
            st.markdown(_section_text(rd, "history"))
        with n2:
            st.markdown("#### Ownership and Funding")
            st.markdown(_section_text(rd, "ownership"))
        st.markdown("#### Analysis")
        st.markdown(_section_text(rd, "analysis"))


def render_overview(rd: dict, summary: dict[str, dict]) -> None:
    withheld = rd.get("evidence_sufficient") is False
    bias = _coalesce(rd.get("bias_rating"))
    factual = _coalesce(rd.get("factual_reporting"))
    cred = "Withheld" if withheld else _coalesce(rd.get("credibility_rating"))
    c1, c2, c3 = st.columns([1.05, 1.05, 1.1])
    with c1:
        st.markdown('<div class="profile-card">', unsafe_allow_html=True)
        st.markdown("### Bias and Factuality")
        if withheld:
            st.warning(
                f"Verdict withheld — {rd.get('insufficient_evidence_reason') or 'insufficient evidence'}. "
                "The collected evidence is shown in the other tabs."
            )
        else:
            st.markdown(f'<div class="badge-row">{_badge(bias, BIAS_COLOR)}{_badge(factual, FACT_COLOR)}</div>', unsafe_allow_html=True)
            render_bias_bar("Bias score", rd.get("bias_score"), _label_color(bias, BIAS_COLOR))
            render_factuality_bar("Factuality risk score", rd.get("factual_score"), _label_color(factual, FACT_COLOR))
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="profile-card">', unsafe_allow_html=True)
        st.markdown("### Profile")
        st.markdown(
            f"""
<p><strong>Credibility:</strong> {_esc(cred)}</p>
<p><strong>Country:</strong> {_esc(rd.get("country"))}</p>
<p><strong>Press freedom:</strong> {_esc(rd.get("country_freedom_rating"))}</p>
<p><strong>Media type:</strong> {_esc(rd.get("media_type"))}</p>
<p><strong>Traffic:</strong> {_esc(rd.get("traffic_popularity"))}</p>
""",
            unsafe_allow_html=True,
        )
        src = rd.get("source_url") or rd.get("mbfc_url")
        if src:
            st.link_button("Open outlet source", src, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="profile-card">', unsafe_allow_html=True)
        st.markdown("### Evidence Audit")
        st.markdown(
            "<p>See the <strong>Evidence tab</strong> for the full pipeline audit: "
            "analyzer reasoning, bias component scores, cited articles, vague sourcing "
            "examples, history, ownership, and fact-check findings.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    overall = _section_text(rd, "overall_summary")
    if overall:
        st.markdown("### Summary")
        st.markdown(overall)

    history_text = _section_text(rd, "history")
    ownership_text = _section_text(rd, "ownership")
    analysis_text = _section_text(rd, "analysis")
    if any([history_text, ownership_text, analysis_text]):
        with st.expander("History, Ownership & Analysis", expanded=False):
            if history_text:
                st.markdown("#### History")
                st.markdown(history_text)
            if ownership_text:
                st.markdown("#### Ownership and Funding")
                st.markdown(ownership_text)
            if analysis_text:
                st.markdown("#### Analysis")
                st.markdown(analysis_text)

def normalize_fact_checks(fc_list) -> list[str]:
    if not fc_list:
        return []
    if isinstance(fc_list, list) and len(fc_list) == 1:
        single = str(fc_list[0]).strip().lower()
        if single in {"none", "none in the last 5 years", "no failed fact checks"}:
            return []
    if isinstance(fc_list, dict):
        fc_list = fc_list.get("findings", [])
    checks: list[str] = []
    for item in fc_list:
        if isinstance(item, dict):
            verdict = item.get("verdict", "")
            claim = item.get("claim") or item.get("claim_summary") or item.get("title") or ""
            text = f"[{verdict}] {claim}" if verdict else claim
        else:
            text = str(item)
        if text.strip():
            checks.append(text.strip())
    return checks


def _clean_evidence_text(value, limit: int = 900) -> str:
    text = _coalesce(value, "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _confidence_text(value) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text:
            if text.lower().startswith("confidence:"):
                return text
            return f"Confidence: {text}"
    numeric = _as_float(value)
    if numeric is None:
        return "Confidence: Not assessed"
    if numeric >= 0.8:
        return "Confidence: High"
    if numeric >= 0.55:
        return "Confidence: Medium"
    return "Confidence: Low"


def _normalize_article(item, number: int | None = None) -> dict:
    if not isinstance(item, dict):
        return {"number": number, "title": str(item), "url": "", "snippet": ""}
    exact_span = _clean_evidence_text(item.get("exact_span") or item.get("excerpt") or item.get("evidence"), 520)
    relevance = _clean_evidence_text(item.get("relevance") or item.get("match_reason"), 360)
    evidence_status = _normalize_evidence_status(item.get("evidence_status"), bool(item.get("indirect", False)))
    status_reason = _clean_evidence_text(item.get("status_reason"), 360)
    rating_supporting = item.get("rating_supporting")
    if rating_supporting is None:
        rating_supporting = _status_rating_supporting(evidence_status)
    return {
        "evidence_id": _coalesce(item.get("evidence_id"), ""),
        "number": item.get("number", number),
        "title": _coalesce(item.get("title"), f"Article {number}" if number else "Article"),
        "url": _coalesce(item.get("url"), ""),
        "snippet": _clean_evidence_text(item.get("snippet") or item.get("context") or item.get("text"), 520),
        "exact_span": exact_span,
        "excerpt": exact_span,
        "relevance": relevance,
        "match_reason": relevance,
        "limitation": _clean_evidence_text(item.get("limitation"), 360),
        "not_prove": _clean_evidence_text(item.get("not_prove"), 360),
        "indirect": bool(item.get("indirect", False)),
        "evidence_status": evidence_status,
        "status_reason": status_reason,
        "rating_supporting": bool(rating_supporting),
    }


def _article_index_from_sources(evidence: list[dict]) -> list[dict]:
    articles = []
    for idx, src in enumerate(evidence or [], 1):
        if isinstance(src, dict):
            articles.append(_normalize_article(src, idx))
    return articles


def _extract_debug_field(section: str, label: str) -> str:
    if not section:
        return ""
    pattern = re.compile(rf"^{re.escape(label)}:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(section)
    return _clean_evidence_text(match.group(1), 1200) if match else ""


def _debug_article_index(rd: dict, evidence: list[dict]) -> list[dict]:
    articles = _article_index_from_sources(evidence)
    by_number = {a.get("number"): dict(a) for a in articles}
    sections = split_debug_evidence(rd.get("_debug_pipeline_evidence"))
    article_section = ""
    for key, value in sections.items():
        if key.startswith("Analyzed Articles"):
            article_section = value
            break
    for line in article_section.splitlines():
        line = line.strip()
        if not line.lower().startswith("article "):
            continue
        left, sep, snippet = line.partition(" — ")
        if not sep:
            left, sep, snippet = line.partition(" - ")
        match = re.match(r"Article\s+(\d+):\s*(.*)", left, re.IGNORECASE)
        if not match:
            continue
        number = int(match.group(1))
        title = match.group(2).strip().strip('"') or f"Article {number}"
        base = by_number.get(number, {"number": number, "url": ""})
        base.update({"number": number, "title": title, "snippet": _clean_evidence_text(snippet, 520)})
        by_number[number] = base
    return [by_number[n] for n in sorted(by_number) if by_number[n].get("title")]


def _article_text(article: dict) -> str:
    return f"{article.get('title', '')} {article.get('snippet', '')}".casefold()


def _matching_sentence(article: dict, terms: list[str]) -> str:
    title = _coalesce(article.get("title"), "")
    snippet = _coalesce(article.get("snippet"), "")
    lowered_terms = [t.casefold() for t in terms]
    if any(term in title.casefold() for term in lowered_terms):
        return title
    for chunk in re.split(r"(?<=[.!?])\s+|\n+", snippet):
        chunk = chunk.strip()
        if chunk and any(term in chunk.casefold() for term in lowered_terms):
            return chunk
    return title or snippet


def _score_article_for_terms(article: dict, terms: list[str], exclude_terms: list[str] | None = None) -> int:
    title = _coalesce(article.get("title"), "").casefold()
    snippet = _coalesce(article.get("snippet"), "").casefold()
    exclude_terms = exclude_terms or []
    if any(term.casefold() in f"{title} {snippet}" for term in exclude_terms):
        return 0
    score = 0
    for term in terms:
        needle = term.casefold()
        if needle in title:
            score += 5
        if needle in snippet:
            score += 2
    return score


def _evidence_items_for_terms(
    article_index: list[dict],
    terms: list[str],
    limit: int = 3,
    exclude_terms: list[str] | None = None,
    reason: str = "",
    limitation: str = "",
    not_prove: str = "",
    indirect: bool = False,
    evidence_status: str | None = None,
    status_reason: str | None = None,
) -> list[dict]:
    scored = []
    for pos, article in enumerate(article_index):
        score = _score_article_for_terms(article, terms, exclude_terms)
        if score <= 0:
            continue
        item = dict(article)
        item["exact_span"] = _clean_evidence_text(_matching_sentence(article, terms), 420)
        item["excerpt"] = item["exact_span"]
        item["relevance"] = reason
        item["match_reason"] = reason
        status = _normalize_evidence_status(evidence_status, indirect) or EVIDENCE_STATUS_DIRECT
        item["limitation"] = limitation
        item["not_prove"] = not_prove
        item["indirect"] = indirect
        item["evidence_status"] = status
        item["status_reason"] = status_reason or _default_status_reason(status)
        item["rating_supporting"] = _status_rating_supporting(status)
        scored.append((score, pos, item))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [item for _, _, item in scored[:limit]]


def _evidence_items_from_examples(examples: list[str], article_index: list[dict], limit: int = 6) -> list[dict]:
    items = []
    by_number = {a.get("number"): a for a in article_index}
    for example in examples[:limit]:
        text = _clean_evidence_text(example, 420)
        article = None
        match = re.search(r"\bArticle\s+(\d+)\b", text, re.IGNORECASE)
        if match:
            article = by_number.get(int(match.group(1)))
        if article is None:
            article_matches = _select_articles_for_text(text, article_index, limit=1)
            article = article_matches[0] if article_matches else {}
        item = dict(article)
        item.setdefault("title", "Evidence example")
        item["exact_span"] = text
        item["excerpt"] = text
        status = _normalize_evidence_status(item.get("evidence_status")) or EVIDENCE_STATUS_INDIRECT
        item["relevance"] = "Analyzer-provided example"
        item["match_reason"] = "Analyzer-provided example"
        item["evidence_status"] = status
        item["status_reason"] = item.get("status_reason") or _default_status_reason(status)
        item["rating_supporting"] = _status_rating_supporting(status)
        items.append(item)
    return items


def _select_articles_for_text(text: str, article_index: list[dict], limit: int = 4) -> list[dict]:
    if not article_index:
        return []
    terms = [w for w in re.findall(r"[\wÀ-ž]{5,}", text.casefold())[:16]]
    items = _evidence_items_for_terms(article_index, terms, limit=limit)
    if items:
        return items
    return article_index[:limit]


def _has_no_opinion_signal(rd: dict) -> bool:
    text = f"{rd.get('analysis', '')}\n{rd.get('_debug_pipeline_evidence', '')}".casefold()
    patterns = [
        r"no\s+(?:op/ed|op-ed|opinion|editorial)",
        r"no\s+opinion\s+pieces",
        r"editorial/opinion\s+bias\s+was\s+scored\s+neutrally",
        r"editorial_bias_score\s*(?:is|=|:)\s*0",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _no_opinion_evidence_span(rd: dict) -> str:
    text = f"{rd.get('analysis', '')}\n{rd.get('_debug_pipeline_evidence', '')}"
    patterns = [
        r"Because no OP/ED pieces were included[^.]*\.",
        r"no\s+(?:op/ed|op-ed|opinion|editorial)[^.]*\.",
        r"no\s+opinion\s+pieces[^.]*\.",
        r"editorial/opinion\s+bias\s+was\s+scored\s+neutrally[^.]*\.",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _clean_evidence_text(match.group(0), 420)
    return "No opinion/editorial articles were identified in the sampled set."


def _vague_examples_from_text(text: str) -> list[str]:
    match = re.search(r"Detected vague sourcing:\s*(.+?)\)?$", text, re.IGNORECASE)
    if not match:
        return []
    raw = match.group(1).strip()
    examples = []
    for part in re.split(r"\"\s*,\s*\"|'\s*,\s*'|;\s*", raw):
        cleaned = part.strip().strip("()[]'\" ")
        if cleaned and cleaned not in examples:
            examples.append(cleaned)
    return examples[:6]


def _loaded_language_examples(reasoning: str, evidence_sources: list[dict]) -> list[dict]:
    """Extract double-quoted phrases from reasoning as loaded language evidence.

    Returns list of {quote, art_url, art_title, art_num} with optional article link.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for m in re.finditer(r'"([^"]{8,120})"', reasoning):
        quote = m.group(1).strip()
        if quote in seen:
            continue
        seen.add(quote)
        # Look for an article number in the surrounding 250 chars
        ctx_start = max(0, m.start() - 250)
        ctx = reasoning[ctx_start: m.end() + 50]
        art_num = None
        art_url = ""
        art_title = ""
        for ref_m in re.finditer(r"\bArticle[s]?\s+(\d+)", ctx, re.IGNORECASE):
            n = int(ref_m.group(1))
            if 1 <= n <= len(evidence_sources):
                candidate = evidence_sources[n - 1]
                if _is_article(candidate):
                    art_num = n
                    art_url = candidate.get("url") or ""
                    art_title = candidate.get("title") or f"Article {n}"
                    break
        out.append({"quote": quote, "art_url": art_url, "art_title": art_title, "art_num": art_num})
    return out[:4]


# Charged wording used only as a last-resort, key-free heuristic when neither reasoning
# quotes nor an LLM-extracted sidecar are available. The sidecar is the primary source.
_LOADED_FALLBACK_TERMS = [
    "slam", "slammed", "blasted", "outrage", "outrageous", "shocking", "disgraceful",
    "brutal", "radical", "extremist", "regime", "propaganda", "smear", "corrupt",
    "scandal", "hysteria", "catastrophic", "disaster", "traitor", "hoax", "fearmonger",
    "shameful", "appalling", "reckless", "sinister", "menace", "tyranny", "lies",
]


def _loaded_language_from_articles(source_url: str, limit: int = 4) -> list[dict]:
    """Last-resort offline fallback: scan cached article text for charged wording.

    Returns the same shape as ``_loaded_language_examples`` so the renderer is unchanged.
    Prefer the LLM-extracted sidecar (``load_loaded_language_sources``); this heuristic
    only runs when neither reasoning quotes nor a sidecar entry exist.
    """
    if not source_url:
        return []
    try:
        parsed = urlparse(source_url if source_url.startswith("http") else f"https://{source_url}")
        domain = (parsed.netloc or parsed.path).replace("www.", "").split("/")[0]
        cache_path = os.path.join(os.path.dirname(__file__), "article_cache", domain, "articles.json")
        if not os.path.exists(cache_path):
            return []
        with open(cache_path, encoding="utf-8") as f:
            articles = json.load(f)
    except Exception:
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for art in articles if isinstance(articles, list) else []:
        text = (art.get("text") or "").strip()
        if not text:
            continue
        for chunk in re.split(r"(?<=[.!?])\s+|\n+", text):
            low = chunk.casefold()
            if 20 <= len(chunk) <= 220 and any(t in low for t in _LOADED_FALLBACK_TERMS):
                quote = chunk.strip()
                if quote.casefold() in seen:
                    continue
                seen.add(quote.casefold())
                out.append({
                    "quote": quote,
                    "art_url": art.get("url") or "",
                    "art_title": art.get("title") or "Article",
                    "art_num": None,
                })
                break  # one example per article
        if len(out) >= limit:
            break
    return out


def _resolve_loaded_examples(rd: dict, evidence_sources: list[dict], reasoning: str, limit: int = 6) -> list[dict]:
    """Layered loaded-language examples: reasoning quotes -> article sidecar -> heuristic.

    Every layer yields {quote, art_url, art_title, art_num}; deduped by quote text so the
    existing renderer (which links art_url when present) works unchanged.
    """
    examples = _loaded_language_examples(reasoning, evidence_sources)
    seen = {e["quote"].casefold() for e in examples}
    if len(examples) < 3:
        name = _coalesce(rd.get("name"), "")
        sidecar = load_loaded_language_sources().get(name) if name and name != "-" else None
        for s in sidecar or []:
            quote = (s.get("quote") or "").strip()
            if not quote or quote.casefold() in seen:
                continue
            seen.add(quote.casefold())
            examples.append({
                "quote": quote,
                "art_url": s.get("art_url") or "",
                "art_title": s.get("art_title") or "Article",
                "art_num": None,
            })
    if not examples:
        examples = _loaded_language_from_articles(rd.get("source_url") or "")
    return examples[:limit]


def _claim_card(
    claim: str,
    basis: str,
    analyzer: str,
    evidence_items: list[dict],
    metadata: dict | None = None,
    narrow_claim: str | None = None,
    confidence=None,
    final_allowed: str | None = None,
    final_not_allowed: str | None = None,
    evidence_status: str | None = None,
    status_reason: str | None = None,
    rating_supporting: bool | None = None,
) -> dict:
    status = _normalize_evidence_status(evidence_status)
    return {
        "claim": claim,
        "claim_label": claim,
        "narrow_claim": _clean_evidence_text(narrow_claim or basis, 900),
        "basis": _clean_evidence_text(basis, 900),
        "analyzer": analyzer,
        "confidence": confidence,
        "articles": evidence_items,
        "evidence_items": evidence_items,
        "final_wording_allowed": final_allowed or claim,
        "final_wording_not_allowed": final_not_allowed or "",
        "evidence_status": status,
        "status_reason": status_reason or (_default_status_reason(status) if status else ""),
        "rating_supporting": rating_supporting,
        "metadata": metadata or {},
    }


def _claim_cards_from_report_text(rd: dict, article_index: list[dict]) -> list[dict]:
    report_text = " ".join(_coalesce(rd.get(k), "") for k in ["analysis", "overall_summary", "bias_category_description"])
    cards = []
    claim_defs = [
        {
            "claim": "Government/institutional actor foregrounding",
            "terms": ["government", "president", "prime minister", "minister", "policy", "law", "election", "court", "high court", "central command", "pentagon", "lawmakers", "military", "fbi", "authorities", "officials", "vučić", "vucic", "predsednik", "vlada", "ministar", "opština", "opstina"],
            "trigger": ["government", "institutional", "official", "policy", "policies", "leaders", "lawmakers", "court", "military"],
            "basis": "The evidence points to official actors, legal bodies, military institutions, government decisions, or political leaders being foregrounded in the sampled article spans.",
            "narrow": "The sample foregrounds official, legal, military, or political actors in specific article spans.",
            "reason": "The span names or centers an official actor, institution, legal body, or government/political decision.",
            "limitation": "This is an actor/topic foregrounding signal from the sampled text.",
            "not_prove": "It does not prove that the outlet promotes, endorses, or advocates government policy.",
            "allowed": "Government/institutional actor foregrounding is present in the sampled spans.",
            "not_allowed": "Government policy promotion.",
            "exclude": ["pretnje", "ubistvo", "blokader", "najoštrije", "threat", "murder"],
            "status": "Direct evidence",
            "status_reason": "The exact span directly supports this observable framing/topic claim.",
        },
        {
            "claim": "Security/conflict framing",
            "terms": ["national security", "security", "defense", "military", "army", "police", "border", "war", "terrorism", "shooting", "crash", "aircraft", "ira", "detainee", "bases", "strait of hormuz", "sanctions", "vojska", "pvo", "radari", "brigade", "brigada", "agresori", "ormuz"],
            "trigger": ["security", "defense", "military", "war", "conflict", "terrorism"],
            "basis": "The evidence centers war, military action, policing, terrorism investigation, detainee abuse, sanctions, or conflict-related disruption.",
            "narrow": "Several sampled spans center conflict, military, policing, terrorism, or security-adjacent events.",
            "reason": "The local span centers security, conflict, military, policing, terrorism, or defense-related subject matter.",
            "limitation": "This supports a topic/framing signal only.",
            "not_prove": "It does not prove national-security advocacy, nationalism, or support for security policy.",
            "allowed": "Security/conflict framing appears in the sampled article spans.",
            "not_allowed": "National security emphasis or national-security advocacy.",
            "status": "Direct evidence",
            "status_reason": "The exact span directly supports this observable framing/topic claim.",
        },
        {
            "claim": "Critical framing of political actor or policy",
            "terms": ["downplays", "punctures", "perils", "accused", "claims", "true price unknown", "cost", "sabotaging", "critical", "shrugged off", "political gain", "threatened", "trump", "orbán", "orban"],
            "trigger": ["critical stance", "critical perspectives", "conservative leaders", "pro-fossil", "nationalist leaders"],
            "basis": "The evidence contains headline or snippet phrasing that presents a political actor, policy, or campaign claim through criticism, consequence, or challenge framing.",
            "narrow": "The sample includes critical framing of specific political actors or policy choices in local spans.",
            "reason": "The phrase presents a political actor, policy, or claim through criticism, cost, contradiction, or challenge framing.",
            "limitation": "This is news-framing evidence from sampled articles.",
            "not_prove": "It does not by itself prove partisan advocacy, party endorsement, or an opinion stance.",
            "allowed": "Critical framing of specific political actors or policies appears in the sampled spans.",
            "not_allowed": "Partisan attack or partisan opinion content.",
            "status": "Direct evidence",
            "status_reason": "The exact span directly supports this observable framing/topic claim.",
        },
        {
            "claim": "Rights-focused framing",
            "terms": ["rights", "freedoms", "privacy", "human rights", "violating freedoms", "detainee", "abuse", "rape", "legal accountability", "charges", "right to privacy"],
            "trigger": ["rights-focused", "human-rights", "human rights", "humanitarian"],
            "basis": "The evidence foregrounds rights, freedoms, privacy, detainee abuse, or legal-accountability language.",
            "narrow": "The sample includes article spans that frame events through rights, freedoms, privacy, or legal accountability.",
            "reason": "The span uses rights, freedoms, privacy, abuse, or legal-accountability language as a frame.",
            "limitation": "This supports a rights-focused framing signal in the sample.",
            "not_prove": "It does not prove a comprehensive rights-based editorial ideology by itself.",
            "allowed": "Rights-focused framing appears in the sampled spans.",
            "not_allowed": "Rights advocacy or explicit endorsement unless the text directly advocates a position.",
            "status": "Direct evidence",
            "status_reason": "The exact span directly supports this observable framing/topic claim.",
        },
        {
            "claim": "Energy/fossil-fuel risk framing",
            "terms": ["fossil", "oil", "gas", "drill", "renewable", "energy efficiency", "supply disruption", "oil markets", "fuel exports", "fossil fuel", "energy crisis"],
            "trigger": ["pro-regulation", "environmental", "fossil", "energy", "oil"],
            "basis": "The evidence frames oil, gas, fossil fuels, energy disruption, or renewable-energy tradeoffs through risk, cost, or consequence language.",
            "narrow": "The sample includes energy and fossil-fuel spans framed around disruption, cost, dependency, or risk.",
            "reason": "The span centers fossil fuel, energy-market, or renewable-energy consequences.",
            "limitation": "This is an energy-risk framing signal, not a direct policy endorsement.",
            "not_prove": "It does not prove pro-regulation advocacy unless the span explicitly endorses regulation.",
            "allowed": "Energy/fossil-fuel risk framing appears in the sampled spans.",
            "not_allowed": "Pro-regulation framing or environmental advocacy unless explicitly supported by the span.",
            "status": "Direct evidence",
            "status_reason": "The exact span directly supports this observable framing/topic claim.",
        },
        {
            "claim": "Adversarial/conflict wording for external actors",
            "terms": ["adversarial", "external actors", "western", "west", "foreign", "hostile", "interfering", "aggressor", "sanction", "threat", "brutal", "without mercy", "america", "u.s.", "iran", "russia", "embassy", "zapad", "amerika", "sad", "agresori", "brutalnu", "bez milosti", "brisel"],
            "trigger": ["adversarial", "external actors", "western", "foreign actors", "hostile", "opposition"],
            "basis": "The evidence uses conflict, hostility, sanction, threat, or external-actor language in specific spans.",
            "narrow": "The sample includes adversarial or conflict-oriented wording about external actors in local spans.",
            "reason": "The phrase frames an external actor through conflict, hostility, threat, or sanctions language.",
            "limitation": "This is a wording/framing signal in the sampled text.",
            "not_prove": "It does not prove propaganda or outlet-level hostility toward that actor.",
            "allowed": "Adversarial/conflict wording for external actors appears in the sampled spans.",
            "not_allowed": "Propaganda or hostile editorial stance unless directly supported.",
            "status": "Direct evidence",
            "status_reason": "The exact span directly supports this observable framing/topic claim.",
        },
    ]
    lowered_report = report_text.casefold()
    for spec in claim_defs:
        if not any(t.casefold() in lowered_report for t in spec["trigger"]):
            continue
        items = _evidence_items_for_terms(
            article_index,
            spec["terms"],
            limit=3,
            exclude_terms=spec.get("exclude"),
            reason=spec["reason"],
            limitation=spec["limitation"],
            not_prove=spec["not_prove"],
            indirect=spec.get("status") == EVIDENCE_STATUS_INDIRECT,
            evidence_status=spec.get("status", EVIDENCE_STATUS_DIRECT),
            status_reason=spec.get("status_reason"),
        )
        if items:
            cards.append(_claim_card(
                spec["claim"],
                spec["basis"],
                "Claim-local article match",
                items,
                narrow_claim=spec["narrow"],
                confidence="High" if spec.get("status") == EVIDENCE_STATUS_DIRECT else "Medium",
                final_allowed=spec["allowed"],
                final_not_allowed=spec["not_allowed"],
                evidence_status=spec.get("status", EVIDENCE_STATUS_DIRECT),
                status_reason=spec.get("status_reason"),
            ))

    if _has_no_opinion_signal(rd):
        span = _no_opinion_evidence_span(rd)
        cards.append(_claim_card(
            "Not assessed: no opinion sample",
            "Opinion/editorial bias should not be scored from this sample because the available evidence indicates that no opinion/editorial articles were included.",
            "EditorialBias",
            [{
                "title": "Editorial bias analysis",
                "url": "",
                "exact_span": span,
                "excerpt": span,
                "relevance": "The analyzer explicitly indicates that opinion/editorial content was not present in the sampled set.",
                "limitation": "This only covers the sampled articles available to the pipeline.",
                "not_prove": "It does not prove the outlet never publishes opinion pieces or has no partisan opinion content outside this sample.",
                "evidence_status": EVIDENCE_STATUS_NOT_ASSESSED,
                "status_reason": "The sample lacks opinion/editorial articles needed to assess partisan opinion content.",
                "rating_supporting": False,
            }],
            narrow_claim="No opinion/editorial articles were included in the sampled set, so partisan opinion content is not assessed from this sample.",
            confidence="Not assessed",
            final_allowed="Not assessed: no opinion/editorial sample was included.",
            final_not_allowed="Partisan opinion content or balanced editorial opinion based on this sample.",
            evidence_status=EVIDENCE_STATUS_NOT_ASSESSED,
            status_reason="The sample lacks opinion/editorial articles needed to assess partisan opinion content.",
            rating_supporting=False,
        ))
    else:
        opinion_terms = ["opinion", "column", "columns", "op-ed", "editorial", "kolumne", "kolumna"]
        opinion_items = _evidence_items_for_terms(
            article_index,
            opinion_terms,
            limit=2,
            reason="The article is explicitly labeled as opinion, column, editorial, or equivalent.",
            limitation="This supports article-type presence, not the ideological content of the opinion.",
            not_prove="It does not prove partisan opinion unless the specific opinion span advocates a partisan position.",
            evidence_status=EVIDENCE_STATUS_DIRECT,
            status_reason="The exact span directly supports the presence of opinion-labeled content.",
        )
        if opinion_items and any(term in lowered_report for term in ["opinion", "column", "partisan", "editorial"]):
            cards.append(_claim_card(
                "Opinion-labeled sample present",
                "The sample contains article-level evidence that at least one item is labeled as opinion, column, editorial, or equivalent.",
                "Claim-local article match",
                opinion_items,
                narrow_claim="At least one sampled item is labeled as opinion/editorial/column content.",
                confidence="Medium",
                final_allowed="Opinion-labeled content appears in the sampled set.",
                final_not_allowed="Partisan opinion content unless a partisan advocacy span is shown.",
                evidence_status=EVIDENCE_STATUS_DIRECT,
                status_reason="The exact span directly supports the presence of opinion-labeled content.",
            ))

    loaded_terms = ["loaded", "sensational", "adversarial", "brutal", "without mercy", "threat", "fake", "malicious", "aggressor", "outrage", "shocking", "enemy", "najoštrije", "pretnje", "ubistvo", "nesrećne", "bez milosti", "brutalnu", "agresori", "gorela", "piromanom"]
    if any(term in lowered_report for term in ["loaded", "sensational", "adversarial", "accus", "malicious"]):
        loaded_items = _evidence_items_for_terms(
            article_index,
            loaded_terms,
            limit=4,
            reason="The excerpt contains loaded, sensational, or accusatory wording.",
            limitation="This identifies phrase-level wording, not intent or outlet-wide manipulation.",
            not_prove="It does not prove systematic loaded language unless repeated examples support that broader claim.",
            evidence_status=EVIDENCE_STATUS_DIRECT,
            status_reason="The exact phrase directly supports a loaded/emotive wording example.",
        )
        if loaded_items:
            cards.append(_claim_card(
                "Loaded/emotive phrase examples",
                "The label is backed by specific phrases in headlines or snippets. The highlighted excerpts are phrase-level examples.",
                "Claim-local article match",
                loaded_items,
                metadata={"examples": [item.get("excerpt", "") for item in loaded_items if item.get("excerpt")]},
                narrow_claim="Specific sampled headlines or snippets contain loaded, emotive, sensational, or accusatory wording.",
                confidence="Medium",
                final_allowed="Loaded/emotive phrase examples appear in the sampled spans.",
                final_not_allowed="Manipulative language or propaganda unless the evidence supports that stronger label.",
                evidence_status=EVIDENCE_STATUS_DIRECT,
                status_reason="The exact phrase directly supports a loaded/emotive wording example.",
            ))
    return cards

def _fallback_claim_cards(rd: dict, article_index: list[dict]) -> list[dict]:
    sections = split_debug_evidence(rd.get("_debug_pipeline_evidence"))
    cards = []
    sourcing = sections.get("Sourcing Analysis", "")
    if sourcing:
        basis = _extract_debug_field(sourcing, "Reasoning") or sourcing
        vague_examples = _vague_examples_from_text(basis)
        evidence_items = [
            {
                "title": "Vague sourcing example",
                "url": "",
                "exact_span": example,
                "excerpt": example,
                "relevance": "Sourcing analyzer flagged this attribution as vague or unnamed.",
                "limitation": "This supports a sourcing-transparency caveat for the sample, not a verdict that all sourcing is weak.",
                "not_prove": "It does not prove the outlet generally lacks credible sourcing.",
                "evidence_status": EVIDENCE_STATUS_DIRECT,
                "status_reason": "The exact phrase directly supports a sourcing-transparency caveat.",
                "rating_supporting": True,
            }
            for example in vague_examples
        ]
        cards.append(_claim_card(
            "Sourcing transparency caveat",
            _clean_evidence_text(basis, 1000),
            "Sourcing",
            evidence_items,
            metadata={"vague_sourcing_examples": vague_examples},
            narrow_claim="The sourcing analyzer found vague or anonymous attribution examples that reduce source precision in the sample.",
            confidence="Medium" if vague_examples else "Low",
            final_allowed="The sample includes sourcing-transparency caveats around vague or anonymous attribution.",
            final_not_allowed="Weak sourcing as an outlet-wide conclusion without broader evidence.",
            evidence_status=EVIDENCE_STATUS_DIRECT if vague_examples else EVIDENCE_STATUS_INDIRECT,
            status_reason="The exact phrase directly supports a sourcing-transparency caveat." if vague_examples else "The sourcing reasoning is related but no exact vague-attribution span was attached.",
            rating_supporting=bool(vague_examples),
        ))
    return cards


_CONSERVATIVE_CLAIM_LABELS = {
    "Government policy promotion": "Government/institutional actor foregrounding",
    "National security emphasis": "Security/conflict framing",
    "Partisan opinion content": "Opinion-labeled sample present",
    "Loaded language examples": "Loaded/emotive phrase examples",
    "Sourcing quality": "Sourcing transparency caveat",
    "Story selection and framing pattern": "Story/framing signal reported by analyzer",
    "One-sided or adversarial framing": "Balance/adversarial framing signal reported by analyzer",
}


_CLAIM_DEFAULTS = {
    "Government/institutional actor foregrounding": {
        "narrow": "The sample foregrounds official, legal, military, or political actors in specific article spans.",
        "allowed": "Government/institutional actor foregrounding is present in the sampled spans.",
        "not_allowed": "Government policy promotion.",
        "limitation": "This is an actor/topic foregrounding signal from the sampled text.",
        "not_prove": "It does not prove that the outlet promotes, endorses, or advocates government policy.",
    },
    "Security/conflict framing": {
        "narrow": "The sample centers conflict, military, policing, terrorism, or security-adjacent events in specific spans.",
        "allowed": "Security/conflict framing appears in the sampled article spans.",
        "not_allowed": "National security emphasis or national-security advocacy.",
        "limitation": "This supports a topic/framing signal only.",
        "not_prove": "It does not prove national-security advocacy, nationalism, or support for security policy.",
    },
    "Opinion-labeled sample present": {
        "narrow": "At least one sampled item is labeled as opinion/editorial/column content.",
        "allowed": "Opinion-labeled content appears in the sampled set.",
        "not_allowed": "Partisan opinion content unless a partisan advocacy span is shown.",
        "limitation": "This supports article-type presence, not the ideological content of the opinion.",
        "not_prove": "It does not prove partisan opinion unless the specific opinion span advocates a partisan position.",
    },
    "Not assessed: no opinion sample": {
        "narrow": "No opinion/editorial articles were included in the sampled set, so partisan opinion content is not assessed from this sample.",
        "allowed": "Not assessed: no opinion/editorial sample was included.",
        "not_allowed": "Partisan opinion content or balanced editorial opinion based on this sample.",
        "limitation": "This only covers the sampled articles available to the pipeline.",
        "not_prove": "It does not prove the outlet never publishes opinion pieces or has no partisan opinion content outside this sample.",
    },
    "Loaded/emotive phrase examples": {
        "narrow": "Specific sampled headlines or snippets contain loaded, emotive, sensational, or accusatory wording.",
        "allowed": "Loaded/emotive phrase examples appear in the sampled spans.",
        "not_allowed": "Manipulative language or propaganda unless the evidence supports that stronger label.",
        "limitation": "This identifies phrase-level wording, not intent or outlet-wide manipulation.",
        "not_prove": "It does not prove systematic loaded language unless repeated examples support that broader claim.",
    },
    "Sourcing transparency caveat": {
        "narrow": "The sample includes vague or anonymous attribution examples or sourcing caveats that reduce source precision.",
        "allowed": "The sample includes sourcing-transparency caveats around vague or anonymous attribution.",
        "not_allowed": "Weak sourcing as an outlet-wide conclusion without broader evidence.",
        "limitation": "This supports a sourcing-transparency caveat for the sample.",
        "not_prove": "It does not prove the outlet generally lacks credible sourcing.",
    },
}


def _conservative_claim_label(label: str) -> str:
    return _CONSERVATIVE_CLAIM_LABELS.get(label, label)


def _claim_defaults(label: str) -> dict[str, str]:
    if label in _CLAIM_DEFAULTS:
        return _CLAIM_DEFAULTS[label]
    if ":" in label:
        return {
            "narrow": "The analyzer reported a policy-domain framing signal tied to the attached indicators.",
            "allowed": f"Policy-domain framing signal: {label}.",
            "not_allowed": "Outlet-wide ideology claim unless the attached spans directly support it.",
            "limitation": "This is limited to the analyzer indicators and sampled articles.",
            "not_prove": "It does not prove a comprehensive outlet-wide policy platform by itself.",
        }
    return {
        "narrow": "The claim is limited to the attached evidence spans and analyzer basis.",
        "allowed": label,
        "not_allowed": "A stronger outlet-level conclusion without direct evidence anchors.",
        "limitation": "This claim is limited to the sampled evidence shown here.",
        "not_prove": "It does not prove a broader conclusion beyond the attached spans.",
    }


def _article_used_label(article: dict) -> str:
    number = article.get("number")
    title = _coalesce(article.get("title"), "Evidence item")
    return f"Article {number}: {title}" if number else title


def _finalize_claim_cards(cards: list[dict]) -> list[dict]:
    finalized = []
    evidence_num = 1
    for card in cards:
        raw_label = _coalesce(card.get("claim_label") or card.get("claim"), "Evidence claim")
        initial_label = _conservative_claim_label(raw_label)
        defaults = _claim_defaults(initial_label)
        metadata = card.get("metadata") if isinstance(card.get("metadata"), dict) else {}
        evidence_items = [
            _normalize_article(item) for item in (card.get("evidence_items") or metadata.get("evidence_items") or [])
            if isinstance(item, dict)
        ]
        articles = [_normalize_article(a) for a in card.get("articles") or [] if isinstance(a, dict)]
        if not evidence_items and articles:
            evidence_items = articles

        card_default_status = _normalize_evidence_status(card.get("evidence_status"))
        if not card_default_status:
            card_default_status = _normalize_evidence_status(defaults.get("status")) or EVIDENCE_STATUS_INDIRECT
        normalized_items = []
        for item in evidence_items:
            if not item.get("exact_span"):
                item["exact_span"] = _clean_evidence_text(item.get("title") or item.get("snippet"), 520)
                item["excerpt"] = item["exact_span"]
            if not item.get("evidence_id") or item.get("evidence_id") == "-":
                item["evidence_id"] = f"E{evidence_num}"
                evidence_num += 1
            status = _normalize_evidence_status(item.get("evidence_status"), bool(item.get("indirect"))) or card_default_status
            if not item.get("exact_span"):
                status = EVIDENCE_STATUS_BAD
            item["evidence_status"] = status
            item["status_reason"] = _clean_evidence_text(item.get("status_reason"), 360) or _default_status_reason(status)
            item["rating_supporting"] = _status_rating_supporting(status)
            if not item.get("relevance"):
                item["relevance"] = _clean_evidence_text(item.get("match_reason"), 360) or "This is the local phrase/span attached to the claim."
            if not item.get("limitation"):
                item["limitation"] = defaults["limitation"]
            if not item.get("not_prove"):
                item["not_prove"] = defaults["not_prove"]
            normalized_items.append(item)
        if not articles and normalized_items:
            articles = normalized_items

        card_status = _card_evidence_status({**card, "evidence_items": normalized_items})
        rating_supporting = _card_rating_supporting({"evidence_items": normalized_items})
        label = initial_label
        final_allowed = _clean_evidence_text(card.get("final_wording_allowed"), 600) or defaults["allowed"]
        final_not_allowed = _clean_evidence_text(card.get("final_wording_not_allowed"), 600) or defaults["not_allowed"]
        narrow_claim = _clean_evidence_text(card.get("narrow_claim"), 900) or defaults["narrow"]
        if _looks_like_policy_framing(initial_label) and not rating_supporting:
            coverage = _policy_coverage_label(initial_label)
            label = coverage
            narrow_claim = f"The sampled span identifies {coverage.lower()}, but does not directly prove ideological framing."
            final_allowed = f"{coverage} appears in sampled articles."
            final_not_allowed = f"{initial_label} unless a direct ideological stance span is shown."
            card_status = EVIDENCE_STATUS_DOMAIN if card_status != EVIDENCE_STATUS_BAD else card_status
            for item in normalized_items:
                if item.get("evidence_status") != EVIDENCE_STATUS_BAD:
                    item["evidence_status"] = EVIDENCE_STATUS_DOMAIN
                    item["status_reason"] = "The span identifies the topic/domain but does not directly prove ideological framing."
                    item["rating_supporting"] = False

        articles_used = []
        for item in normalized_items or articles:
            label_text = _article_used_label(item)
            if label_text and label_text not in articles_used:
                articles_used.append(label_text)
        status_reason = _clean_evidence_text(card.get("status_reason"), 500) or _default_status_reason(card_status)
        finalized.append({
            "claim": label,
            "claim_label": label,
            "narrow_claim": narrow_claim,
            "basis": _clean_evidence_text(card.get("basis"), 1000),
            "analyzer": _coalesce(card.get("analyzer"), "Pipeline"),
            "confidence": card.get("confidence"),
            "articles": articles,
            "evidence_items": normalized_items,
            "articles_used": articles_used,
            "final_wording_allowed": final_allowed,
            "final_wording_not_allowed": final_not_allowed,
            "evidence_status": card_status,
            "status_reason": status_reason,
            "rating_supporting": rating_supporting,
            "metadata": metadata,
        })
    return finalized

def _evidence_details_for_report(rd: dict, evidence: list[dict]) -> dict:
    raw_details = rd.get("evidence_details") if isinstance(rd.get("evidence_details"), dict) else {}
    article_index = [_normalize_article(a, i + 1) for i, a in enumerate(raw_details.get("article_index") or [])]
    if not article_index:
        article_index = _debug_article_index(rd, evidence)

    cards = []
    generic_unlocal_claims = {
        "Editorial stance and policy framing",
        "Story selection and framing pattern",
        "One-sided or adversarial framing",
    }
    for card in raw_details.get("claim_cards") or []:
        if not isinstance(card, dict):
            continue
        basis = _clean_evidence_text(card.get("basis"), 1000)
        if not basis:
            continue
        metadata = card.get("metadata") if isinstance(card.get("metadata"), dict) else {}
        evidence_items = [
            _normalize_article(item) for item in (card.get("evidence_items") or metadata.get("evidence_items") or [])
            if isinstance(item, dict)
        ]
        examples = metadata.get("examples") or metadata.get("vague_sourcing_examples") or []
        if not evidence_items and examples:
            evidence_items = _evidence_items_from_examples(examples, article_index)
        articles = [_normalize_article(a) for a in card.get("articles") or [] if isinstance(a, dict)]
        claim = _coalesce(card.get("claim"), "Evidence claim")
        if claim in generic_unlocal_claims and not evidence_items:
            continue
        if not articles and evidence_items:
            articles = evidence_items
        if not evidence_items and not articles and claim != "Sourcing quality":
            continue
        cards.append({
            "claim": claim,
            "claim_label": card.get("claim_label") or claim,
            "narrow_claim": card.get("narrow_claim"),
            "basis": basis,
            "analyzer": _coalesce(card.get("analyzer"), "Pipeline"),
            "confidence": card.get("confidence"),
            "articles": articles,
            "evidence_items": evidence_items,
            "final_wording_allowed": card.get("final_wording_allowed"),
            "final_wording_not_allowed": card.get("final_wording_not_allowed"),
            "evidence_status": card.get("evidence_status"),
            "status_reason": card.get("status_reason"),
            "rating_supporting": card.get("rating_supporting"),
            "metadata": metadata,
        })

    seen_claims = {c["claim"] for c in cards}
    for source_card in (
        _claim_cards_from_report_text(rd, article_index)
        + _fallback_claim_cards(rd, article_index)
    ):
        if source_card["claim"] not in seen_claims:
            cards.append(source_card)
            seen_claims.add(source_card["claim"])
    if not cards and article_index:
        cards.append({
            "claim": "Attached article evidence",
            "narrow_claim": "The cached report includes article-level sources, but no analytical claim cards were attached.",
            "basis": "The report includes these article-level sources as the available evidence for the cached run.",
            "analyzer": "EvidenceSources",
            "confidence": "Not assessed",
            "articles": article_index[:6],
            "evidence_items": article_index[:6],
            "final_wording_allowed": "Article-level evidence is attached for inspection.",
            "final_wording_not_allowed": "Any analytical conclusion without a claim-local evidence span.",
            "evidence_status": EVIDENCE_STATUS_DOMAIN,
            "status_reason": "Attached article references are source context, not claim-local proof.",
            "rating_supporting": False,
            "metadata": {},
        })

    loaded = raw_details.get("loaded_language_examples") or []
    if not loaded:
        for card in cards:
            examples = card.get("metadata", {}).get("examples") if isinstance(card.get("metadata"), dict) else None
            if examples:
                loaded = examples
                break

    cards = _finalize_claim_cards(cards)

    return {
        "article_index": article_index,
        "claim_cards": cards,
        "loaded_language_examples": loaded,
        "sourcing": raw_details.get("sourcing") if isinstance(raw_details.get("sourcing"), dict) else {},
        "component_reasoning": raw_details.get("component_reasoning") if isinstance(raw_details.get("component_reasoning"), dict) else {},
    }


def _render_article_evidence(article: dict) -> None:
    url = _coalesce(article.get("url"), "")
    title = _coalesce(article.get("title"), "Article")
    number = article.get("number")
    evidence_id = _coalesce(article.get("evidence_id"), "-")
    article_label = f"Article {number}: {title}" if number else title
    snippet = _clean_evidence_text(article.get("snippet"), 520)
    exact_span = _clean_evidence_text(article.get("exact_span") or article.get("excerpt") or article.get("evidence") or "", 520)
    relevance = _clean_evidence_text(article.get("relevance") or article.get("match_reason"), 360)
    limitation = _clean_evidence_text(article.get("limitation"), 360)
    not_prove = _clean_evidence_text(article.get("not_prove"), 360)
    status = _normalize_evidence_status(article.get("evidence_status"), bool(article.get("indirect"))) or EVIDENCE_STATUS_DOMAIN
    status_reason = _clean_evidence_text(article.get("status_reason"), 360) or _default_status_reason(status)
    rating_supporting = bool(article.get("rating_supporting")) and status == EVIDENCE_STATUS_DIRECT
    if url and url != "-":
        article_html = f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{_esc(article_label)}</a>'
    else:
        article_html = f"<strong>{_esc(article_label)}</strong>"
    exact_html = f'<p><strong>Exact span:</strong> {_esc(exact_span)}</p>' if exact_span else ""
    relevance_html = f'<p class="small-muted"><strong>Relevance:</strong> {_esc(relevance)}</p>' if relevance else ""
    limitation_html = f'<p class="small-muted"><strong>Limitation:</strong> {_esc(limitation)}</p>' if limitation else ""
    not_prove_html = f'<p class="small-muted"><strong>What this does NOT prove:</strong> {_esc(not_prove)}</p>' if not_prove else ""
    status_html = f'<span class="claim-pill">Evidence status: {_esc(status)}</span>'
    support_html = f'<span class="claim-pill">Rating-supporting: {"Yes" if rating_supporting else "No"}</span>'
    status_reason_html = f'<p class="small-muted"><strong>Status reason:</strong> {_esc(status_reason)}</p>' if status_reason else ""
    context_html = ""
    if snippet and snippet != exact_span:
        context_html = f'<p class="small-muted"><strong>Context:</strong> {_esc(snippet)}</p>'
    st.markdown(
        f"""
<div class="evidence-card">
  <div class="claim-meta"><span class="claim-pill">Evidence ID: {_esc(evidence_id)}</span>{status_html}{support_html}</div>
  <p><strong>Article:</strong> {article_html}</p>
  {exact_html}
  {status_reason_html}
  {relevance_html}
  {limitation_html}
  {not_prove_html}
  {context_html}
</div>
""",
        unsafe_allow_html=True,
    )


def _render_claim_card(card: dict, idx: int) -> None:
    claim = _coalesce(card.get("claim_label") or card.get("claim"), "Evidence claim")
    analyzer = _coalesce(card.get("analyzer"), "Pipeline")
    confidence = _confidence_text(card.get("confidence"))
    narrow = _clean_evidence_text(card.get("narrow_claim"), 1200)
    basis = _clean_evidence_text(card.get("basis"), 1200)
    allowed = _clean_evidence_text(card.get("final_wording_allowed"), 700)
    not_allowed = _clean_evidence_text(card.get("final_wording_not_allowed"), 700)
    articles_used = card.get("articles_used") or []
    status = _normalize_evidence_status(card.get("evidence_status")) or EVIDENCE_STATUS_BAD
    status_reason = _clean_evidence_text(card.get("status_reason"), 500) or _default_status_reason(status)
    rating_supporting = bool(card.get("rating_supporting")) and status == EVIDENCE_STATUS_DIRECT
    expanded = idx == 0
    with st.expander(f"{claim} - {analyzer}", expanded=expanded):
        st.markdown(
            f"""
<div class="claim-meta">
  <span class="claim-pill">Analyzer source: {_esc(analyzer)}</span>
  <span class="claim-pill">{_esc(confidence)}</span>
  <span class="claim-pill">Evidence status: {_esc(status)}</span>
  <span class="claim-pill">Rating-supporting: {"Yes" if rating_supporting else "No"}</span>
</div>
""",
            unsafe_allow_html=True,
        )
        if status_reason:
            st.markdown("**Evidence status reason**")
            st.markdown(status_reason)
        if narrow:
            st.markdown("**Narrow claim**")
            st.markdown(narrow)
        if articles_used:
            st.markdown("**Articles used**")
            for article_label in articles_used[:8]:
                st.markdown(f"- {_clean_evidence_text(article_label, 220)}")
        if basis:
            st.markdown("**Why this supports the claim**")
            st.markdown(basis)
        evidence_items = card.get("evidence_items") or card.get("metadata", {}).get("evidence_items") or []
        articles = evidence_items or card.get("articles") or []
        if articles:
            st.markdown("**Evidence**")
            st.markdown('<div class="article-list">', unsafe_allow_html=True)
            for article in articles[:6]:
                _render_article_evidence(article)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.caption("No claim-local article span was attached for this claim.")
        if allowed:
            st.markdown("**Final wording allowed in report**")
            st.markdown(allowed)
        if not_allowed:
            st.markdown("**Final wording NOT allowed**")
            st.markdown(not_allowed)

def _propaganda_findings_for_report(rd: dict) -> list[dict]:
    details = rd.get("evidence_details") if isinstance(rd.get("evidence_details"), dict) else {}
    raw_findings = details.get("propaganda_findings") or []
    findings = []
    for item in raw_findings:
        if not isinstance(item, dict):
            continue
        technique = _coalesce(item.get("technique"), "")
        snippet = _clean_evidence_text(item.get("text_snippet") or item.get("quote") or item.get("exact_span"), 520)
        if not technique or not snippet:
            continue
        findings.append({
            "technique": technique,
            "description": PROPAGANDA_TECHNIQUE_DESCRIPTIONS.get(technique, "Propaganda technique evidence"),
            "text_snippet": snippet,
            "context": _clean_evidence_text(item.get("context"), 700),
            "article_number": item.get("article_number"),
            "article_title": _coalesce(item.get("article_title"), "Article"),
            "article_url": _coalesce(item.get("article_url") or item.get("url"), ""),
            "confidence": item.get("confidence"),
            "explanation": _clean_evidence_text(item.get("explanation") or item.get("reason"), 500),
            "limitation": _clean_evidence_text(item.get("limitation"), 360) or "This is article-level rhetoric evidence; it does not by itself prove outlet-wide intent.",
        })
    return findings


def _propaganda_confidence_label(value) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return _confidence_text(value)
    return f"Confidence: {numeric:.2f}"


def _render_propaganda_techniques(rd: dict) -> None:
    findings = _propaganda_findings_for_report(rd)
    st.markdown("#### Propaganda Techniques")
    st.caption("Source: LLM one-sidedness/propaganda analyzer; evidence-only and not used to recalculate scores.")
    if not findings:
        st.info("No concrete propaganda technique findings were returned for the sampled articles.")
        return
    grouped: dict[str, list[dict]] = {}
    for finding in findings:
        grouped.setdefault(finding["technique"], []).append(finding)
    for technique, items in grouped.items():
        description = PROPAGANDA_TECHNIQUE_DESCRIPTIONS.get(technique, "Propaganda technique evidence")
        with st.expander(f"{technique} ({len(items)})", expanded=False):
            st.caption(description)
            for finding in items:
                article_title = finding.get("article_title") or "Article"
                number = finding.get("article_number")
                article_label = f"Article {number}: {article_title}" if number else article_title
                url = finding.get("article_url") or ""
                article_html = (
                    f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{_esc(article_label)}</a>'
                    if url else f"<strong>{_esc(article_label)}</strong>"
                )
                context_html = f'<p class="small-muted"><strong>Context:</strong> {_esc(finding["context"])}</p>' if finding.get("context") else ""
                explanation_html = f'<p><strong>Why this matches:</strong> {_esc(finding["explanation"])}</p>' if finding.get("explanation") else ""
                st.markdown(
                    f'''
<div class="evidence-card">
  <div class="claim-meta">
    <span class="claim-pill">Technique: {_esc(technique)}</span>
    <span class="claim-pill">{_esc(_propaganda_confidence_label(finding.get("confidence")))}</span>
  </div>
  <p><strong>Article:</strong> {article_html}</p>
  <p><strong>Exact span:</strong> {_esc(finding["text_snippet"])}</p>
  {context_html}
  {explanation_html}
  <p class="small-muted"><strong>Limitation:</strong> {_esc(finding["limitation"])}</p>
</div>
''',
                    unsafe_allow_html=True,
                )


def render_evidence(rd: dict) -> None:
    evidence = rd.get("evidence_sources") or []
    checks = normalize_fact_checks(rd.get("failed_fact_checks"))

    # Pre-parse debug sections once — used in both columns
    sections = split_debug_evidence(rd.get("_debug_pipeline_evidence"))
    bias_parsed = parse_editorial_bias_section(sections.get("Editorial Bias Analysis", ""))
    sourcing_parsed = parse_sourcing_section(sections.get("Sourcing Analysis", ""))
    fc_parsed = _structured_fact_checks(rd, parse_fact_check_section(sections.get("Fact Check Search", "")))

    left, right = st.columns([1.5, 1])
    with left:
        st.markdown("### Evidence Audit")
        st.caption(
            "Each section below shows what a specific pipeline analyzer actually found, "
            "with verbatim quotes and clickable article links. All data comes from the "
            "structured pipeline log, not from heuristic reconstruction."
        )
        render_evidence_from_debug(rd)
        st.divider()
        _render_propaganda_techniques(rd)

        article_pages = [a for a in evidence if _is_article(a)]
        non_article_count = len(evidence) - len(article_pages)
        expander_label = f"All sampled articles ({len(article_pages)}"
        if non_article_count:
            expander_label += f" · {non_article_count} non-editorial page(s) excluded"
        expander_label += ")"
        all_article_texts = _load_article_texts(rd.get("source_url", ""))
        with st.expander(expander_label, expanded=False):
            if not article_pages:
                st.info("No article sources in this record.")
            for i, art in enumerate(article_pages, 1):
                _render_cited_article(art, i, all_article_texts)

    with right:
        st.markdown("### Signals at a Glance")

        # Fact checks summary — count only after filtering aggregators, then gate on
        # whether the outlet itself published the checked claim (re-attribution sidecar).
        allowed_fc = fc_parsed.get("allowed_findings", [])
        blocked_fc = len(fc_parsed.get("blocked_findings", []))
        attributed = fc_parsed.get("attributed", False)
        non_published_failed = fc_parsed.get("non_published_failed", []) if attributed else []
        total_fc = len(allowed_fc)
        failed_fc = _fc_failed_count(fc_parsed, allowed_fc)
        if failed_fc > 0:
            st.error(f"**{failed_fc} failed** fact check{'s' if failed_fc != 1 else ''} published by outlet (of {total_fc} found)")
        elif total_fc > 0:
            st.success(f"**0 failed** published by outlet out of {total_fc} checks found")
        else:
            st.info("No IFCN fact-check results found")
            st.caption("Absence ≠ never fact-checked · only 8 IFCN-approved sites searched")
        if blocked_fc:
            st.caption(f"ⓘ {blocked_fc} aggregator result(s) excluded (MBFC/AllSides are not IFCN fact-checkers).")
        # Main findings exclude failed checks not published by the outlet when re-attributed.
        non_published_failed_ids = {id(f) for f in non_published_failed}
        main_fc = [f for f in allowed_fc if id(f) not in non_published_failed_ids] if attributed else allowed_fc
        if main_fc:
            for f in main_fc:
                _render_fact_check_finding(f)
        if non_published_failed:
            with st.expander("False or Misleading Claims Involving This Outlet (Not Published by It)", expanded=False):
                for f in non_published_failed:
                    _render_fact_check_finding(f, mode="non_published")

        st.divider()

        # Loaded language
        ll_label = "Loaded language: Detected" if bias_parsed["uses_loaded_language"] else "Loaded language: Not detected"
        ll_color = "#b44232" if bias_parsed["uses_loaded_language"] else "#2f7d4f"
        st.markdown(
            f'<span class="rating-pill" style="background:{ll_color};">{_esc(ll_label)}</span>',
            unsafe_allow_html=True,
        )
        if bias_parsed["uses_loaded_language"]:
            ideology = bias_parsed.get("ideology_summary", "").strip()
            if ideology:
                st.caption(ideology)
            ll_examples = _resolve_loaded_examples(rd, evidence, bias_parsed.get("reasoning", ""))
            if ll_examples:
                with st.expander("Example phrases from analyzed articles", expanded=False):
                    for ex in ll_examples:
                        quote_block = f'> *"{_esc(ex["quote"])}"*'
                        if ex["art_url"]:
                            quote_block += (
                                f'\n>\n> — <a href="{html.escape(ex["art_url"], quote=True)}"'
                                f' target="_blank" rel="noopener noreferrer"'
                                f' style="color:#4a90d9;">{_esc(ex["art_title"])}</a>'
                            )
                            st.markdown(quote_block, unsafe_allow_html=True)
                        else:
                            st.markdown(quote_block)
            else:
                reasoning_text = bias_parsed.get("reasoning", "").strip()
                if reasoning_text:
                    with st.expander("Analyzer reasoning", expanded=False):
                        st.markdown(reasoning_text)

        # Sourcing score
        if sourcing_parsed["score"] is not None:
            st.metric("Sourcing risk score", f"{sourcing_parsed['score']:.1f}", help="0 = excellent · 10 = very poor")
            st.caption("0–2 excellent · 3–4 good · 5–6 mixed · 7–8 poor · 9–10 very poor")

        # Article count
        st.metric("Articles sampled", len(evidence))

        # Bias component scores
        comp = bias_parsed.get("component_scores") or {}
        if comp:
            st.divider()
            st.markdown("**Bias Components**")
            m1, m2 = st.columns(2)
            m1.metric("Economic 35%", f"{comp['economic']:+.1f}", help="−10 communist · +10 laissez-faire")
            m2.metric("Social 35%", f"{comp['social']:+.1f}", help="−10 progressive · +10 conservative")
            m3, m4 = st.columns(2)
            m3.metric("News 15%", f"{comp['news_reporting']:+.1f}", help="Balance of straight news reporting")
            m4.metric("Editorial 15%", f"{comp['editorial']:+.1f}", help="0 if no op-ed sampled")



def split_debug_evidence(text: str | None) -> dict[str, str]:
    if not text:
        return {}
    sections: dict[str, list[str]] = {"Pipeline Evidence": []}
    current = "Pipeline Evidence"
    pattern = re.compile(r"^==\s*(.+?)\s*==\s*$")
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            current = match.group(1).title()
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


# ---------------------------------------------------------------------------
# Debug-section parsers
# ---------------------------------------------------------------------------

def parse_editorial_bias_section(text: str) -> dict:
    result: dict = {"uses_loaded_language": False, "ideology_summary": "", "reasoning": "", "article_refs": []}
    if not text:
        return result
    m = re.search(r"Uses Loaded Language:\s*(True|False)", text, re.IGNORECASE)
    if m:
        result["uses_loaded_language"] = m.group(1).lower() == "true"
    m = re.search(r"Ideology Summary:\s*(.+?)(?=\nReasoning:|\Z)", text, re.DOTALL | re.IGNORECASE)
    if m:
        result["ideology_summary"] = m.group(1).strip()
    m = re.search(r"Reasoning:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    if m:
        result["reasoning"] = m.group(1).strip()
    # Handle "Article N", "Articles N, M, P", "Articles N–M" (range)
    nums: list[int] = []
    for ref_m in re.finditer(r"\bArticles?\s+([\d][\d,\s–—–\-]*)", result["reasoning"], re.IGNORECASE):
        chunk = ref_m.group(1).strip().rstrip(",")
        for range_m in re.finditer(r"(\d+)\s*[–—–\-]\s*(\d+)", chunk):
            a, b = int(range_m.group(1)), int(range_m.group(2))
            if 0 < b - a < 20:
                nums.extend(range(a, b + 1))
        for n_str in re.findall(r"\d+", chunk):
            n = int(n_str)
            if 1 <= n <= 50:
                nums.append(n)
    result["article_refs"] = sorted(set(nums))
    # Extract 4 bias component scores from reasoning.
    # Strategy 1: inline summary "economic (-4), social (-5), news (-3) and editorial (0)"
    comp_pat = re.compile(
        r"economic\s*\((-?\d+(?:\.\d+)?)\).*?social\s*\((-?\d+(?:\.\d+)?)\)"
        r".*?news\s*\((-?\d+(?:\.\d+)?)\).*?editorial\s*\((-?\d+(?:\.\d+)?)\)",
        re.IGNORECASE | re.DOTALL,
    )
    cm = comp_pat.search(result["reasoning"])
    if cm:
        result["component_scores"] = {
            "economic": float(cm.group(1)),
            "social": float(cm.group(2)),
            "news_reporting": float(cm.group(3)),
            "editorial": float(cm.group(4)),
        }
    else:
        # Strategy 2: extract each component individually (handles bullet-point and varied
        # inline formats such as "Economic (-5) and Social (-6)...; Editorial bias (-7)")
        def _one_score(text: str, keywords: list) -> float | None:
            for kw in keywords:
                m = re.search(
                    rf"\b{re.escape(kw)}(?:\s+(?:score|policy|bias|balance|reporting))?"
                    r"\s*\(([-+]?\d+(?:\.\d+)?)(?!\s*%)\)",
                    text, re.IGNORECASE,
                )
                if m:
                    return float(m.group(1))
            return None
        r = result["reasoning"]
        econ = _one_score(r, ["economic policy", "economic"])
        soc  = _one_score(r, ["social policy", "social"])
        news = _one_score(r, ["straight news balance", "news reporting", "news balance",
                               "straight news", "news"])
        edit = _one_score(r, ["editorial bias", "editorial opinion", "editorial"])
        if all(v is not None for v in [econ, soc, news, edit]):
            result["component_scores"] = {
                "economic": econ, "social": soc,
                "news_reporting": news, "editorial": edit,
            }
        else:
            result["component_scores"] = {}
    return result


def parse_history_section(text: str) -> dict:
    result: dict = {"founded": "", "original_name": "", "key_events": [], "summary": ""}
    if not text:
        return result
    m = re.search(r"Founded:\s*(\d{4})", text, re.IGNORECASE)
    if m:
        result["founded"] = m.group(1)
    m = re.search(r"Original Name:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    if m:
        result["original_name"] = m.group(1).strip()
    # Key events may be a single "Key Events: YEAR: event.; YEAR: event" line
    ke_match = re.search(r"Key Events?:\s*(.+?)(?=\nSummary:|\Z)", text, re.DOTALL | re.IGNORECASE)
    if ke_match:
        events_raw = ke_match.group(1).strip()
        parts = re.split(r";\s*(?=\d{4}[\s:])", events_raw)
        result["key_events"] = [p.strip().rstrip(".") for p in parts if p.strip()]
    else:
        result["key_events"] = [
            line.strip() for line in text.splitlines()
            if re.match(r"^\s*\d{4}[\s:–]", line)
        ]
    m = re.search(r"Summary:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    if m:
        result["summary"] = m.group(1).strip()
    return result


def parse_ownership_section(text: str) -> dict:
    result: dict = {"owner": "", "parent_company": "", "funding_model": "", "headquarters": ""}
    if not text:
        return result
    for field, pattern in [
        ("owner", r"Owner:\s*(.+)$"),
        ("parent_company", r"Parent Company:\s*(.+)$"),
        ("funding_model", r"Funding Model:\s*(.+)$"),
        ("headquarters", r"Headquarters:\s*(.+)$"),
    ]:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            result[field] = m.group(1).strip()
    return result


def parse_fact_check_section(text: str) -> dict:
    result: dict = {"total_checks": 0, "failed_checks": 0, "findings": []}
    if not text:
        return result
    m = re.search(r"Total Checks Found:\s*(\d+)", text, re.IGNORECASE)
    if m:
        result["total_checks"] = int(m.group(1))
    m = re.search(r"Failed Checks:\s*(\d+)", text, re.IGNORECASE)
    if m:
        result["failed_checks"] = int(m.group(1))
    finding_pat = re.compile(
        r"^\s*-\s+(.+?)\s*(?:—|--)\s*Verdict:\s*([^\(]+?)(?:\s*\(Source:\s*(https?://[^\)]+)\))?$",
        re.IGNORECASE,
    )
    for line in text.splitlines():
        fm = finding_pat.match(line)
        if fm:
            result["findings"].append({
                "summary": fm.group(1).strip(),
                "verdict": fm.group(2).strip(),
                "url": (fm.group(3) or "").strip(),
            })
    return result


def parse_sourcing_section(text: str) -> dict:
    result: dict = {"score": None, "reasoning": "", "vague_examples": []}
    if not text:
        return result
    m = re.search(r"Score:\s*([\d.]+)", text, re.IGNORECASE)
    if m:
        try:
            result["score"] = float(m.group(1))
        except ValueError:
            pass
    m = re.search(r"Reasoning:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    if m:
        result["reasoning"] = m.group(1).strip()
    result["vague_examples"] = _vague_examples_from_text(text)
    return result


_NON_ARTICLE_RE = re.compile(
    r"(?i)\b(privacy\s*policy|terms\s*of\s*(use|service)|about\s*us|contact\s*us?"
    r"|author\s*(profile|page|bio)|404|page\s*not\s*found|cookie\s*policy"
    r"|disclaimer|advertise|sitemap|subscribe|log\s*in|sign\s*up)\b"
)

# Bias-rating aggregators that must never appear as fact-check evidence (anti-contamination).
# MBFC is the ground-truth label source; AllSides/NewsGuard are also aggregators, not IFCN fact-checkers.
_FC_EXCLUDED_DOMAINS = {
    "mediabiasfactcheck.com",
    "allsides.com",
    "adfontesmedia.com",
    "adfontes.media",
    "newsguardtech.com",
    "ground.news",
    "realorsatire.com",
    "fakenewscodex.com",
    "thecredibilitycoalition.org",
}


def _is_article(art: dict) -> bool:
    """Return False for non-editorial pages (Privacy Policy, About Us, etc.)."""
    title = art.get("title") or ""
    return not _NON_ARTICLE_RE.search(title)


def _fc_finding_allowed(finding: dict) -> bool:
    """Return False if a fact-check finding URL belongs to an excluded aggregator domain."""
    url = (finding.get("url") or "").lower()
    return not any(domain in url for domain in _FC_EXCLUDED_DOMAINS)


def _fc_counts_as_failed(finding: dict) -> bool:
    """A check counts as the outlet's failure only if the outlet published the claim."""
    return bool(finding.get("published_by_outlet")) and bool(finding.get("is_failed"))


def _fc_failed_count(fc: dict, findings: list[dict]) -> int:
    """Failed-check count for display: outlet-gated when re-attributed, else legacy."""
    if fc.get("attributed"):
        return sum(1 for f in findings if _fc_counts_as_failed(f))
    return sum(1 for f in findings if f.get("is_failed"))


def _is_failed_fact_check_verdict(verdict: str) -> bool:
    return _coalesce(verdict, "").strip().upper() in FAILED_FACT_CHECK_VERDICTS


def _fact_checker_from_url(url: str) -> str:
    parsed = urlparse(url if url.startswith("http") else f"https://{url}") if url else None
    host = (parsed.netloc if parsed else "").lower().removeprefix("www.")
    if "apnews.com" in host:
        return "AP Fact Check"
    if "reuters.com" in host:
        return "Reuters Fact Check"
    if "washingtonpost.com" in host:
        return "Washington Post Fact Checker"
    names = {
        "politifact.com": "PolitiFact",
        "factcheck.org": "FactCheck.org",
        "snopes.com": "Snopes",
        "fullfact.org": "Full Fact",
        "leadstories.com": "Lead Stories",
        "factcheck.kz": "Factcheck.kz",
    }
    for domain, name in names.items():
        if domain in host:
            return name
    if host:
        return host.split(".")[0].replace("-", " ").title()
    return "Unknown fact-checker"


def _norm_join_key(text) -> str:
    """Normalized key for joining demo findings to sidecar findings by claim text."""
    return re.sub(r"\s+", " ", _coalesce(text, "")).strip().casefold()[:160]


def _claim_source_value(value) -> str:
    if hasattr(value, "value"):
        value = value.value
    source = _coalesce(value, "unknown").strip().lower()
    return source if source in {"published_by_outlet", "about_outlet", "third_party", "unknown"} else "unknown"


def _item_has_claim_attribution(item: dict) -> bool:
    return any(
        key in item
        for key in ("claim_source", "published_by_outlet", "claim_source_domain", "attribution_confidence")
    )


def _structured_fact_checks(rd: dict, fallback: dict | None = None) -> dict:
    """Normalize fact-check findings and merge claim-source re-attribution.

    Findings come from the structured ``fact_check_result`` when present (live runs) or
    from the ``fallback`` (debug-text parse, which is what 100% of cached records use).
    Each finding is enriched with ``is_failed``/``source_site`` and, when an outlet has a
    re-attribution sidecar entry, with ``claim_source``/``published_by_outlet``/
    ``claim_source_domain``. ``attributed`` reports whether sidecar data was applied.
    """
    fallback = fallback or {"total_checks": 0, "failed_checks": 0, "findings": []}
    fc = rd.get("fact_check_result") if isinstance(rd.get("fact_check_result"), dict) else None
    if fc:
        raw_items = fc.get("findings") or []
        total_hdr = fc.get("total_checks_count")
        failed_hdr = fc.get("failed_checks_count")
        coverage = fc.get("coverage_sufficient", True)
    else:
        raw_items = fallback.get("findings") or []
        total_hdr = fallback.get("total_checks")
        failed_hdr = fallback.get("failed_checks")
        coverage = fallback.get("coverage_sufficient", True)

    # Sidecar re-attribution, looked up by outlet name and joined per finding by URL
    # (then by normalized claim text for URL-less findings).
    name = _coalesce(rd.get("name"), "")
    reattr = load_fc_reattribution().get(name) if name and name != "-" else None
    raw_has_attribution = any(isinstance(item, dict) and _item_has_claim_attribution(item) for item in raw_items)
    attributed = isinstance(reattr, dict) or raw_has_attribution or bool(fc and "about_outlet_count" in fc)
    by_url: dict[str, dict] = {}
    by_text: dict[str, dict] = {}
    if isinstance(reattr, dict):
        for sf in reattr.get("findings") or []:
            if sf.get("url"):
                by_url[sf["url"]] = sf
            if sf.get("summary"):
                by_text[_norm_join_key(sf["summary"])] = sf

    findings = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        url = _coalesce(item.get("url"), "")
        verdict = _coalesce(item.get("verdict"), "Unknown")
        source = _coalesce(item.get("source_site") or item.get("source"), "") or _fact_checker_from_url(url)
        raw_claim = item.get("claim_summary") or item.get("claim") or item.get("summary") or item.get("title")
        claim = _clean_evidence_text(raw_claim, 420)
        if not claim and not verdict:
            continue
        is_failed = item.get("is_failed")
        if is_failed is None:
            is_failed = _is_failed_fact_check_verdict(verdict)
        finding = {
            "summary": claim,
            "claim_summary": claim,
            "verdict": verdict,
            "source_site": source,
            "url": url,
            "is_failed": bool(is_failed),
            "claim_source": _claim_source_value(item.get("claim_source")),
            "published_by_outlet": bool(item.get("published_by_outlet", False)),
            "claim_source_domain": item.get("claim_source_domain"),
            "attribution_confidence": item.get("attribution_confidence", 0.0),
        }
        sf = None
        if attributed:
            sf = (by_url.get(url) if url else None) or by_text.get(_norm_join_key(raw_claim))
        if sf:
            finding["claim_source"] = _claim_source_value(sf.get("claim_source"))
            finding["published_by_outlet"] = bool(sf.get("published_by_outlet", False))
            finding["claim_source_domain"] = sf.get("claim_source_domain")
            finding["attribution_confidence"] = sf.get("attribution_confidence", 0.0)
        findings.append(finding)

    allowed_findings = [f for f in findings if _fc_finding_allowed(f)]
    blocked_findings = [f for f in findings if not _fc_finding_allowed(f)]
    if attributed:
        published_failed = [f for f in allowed_findings if _fc_counts_as_failed(f)]
        non_published_failed = [
            f for f in allowed_findings
            if f.get("is_failed") and not bool(f.get("published_by_outlet"))
        ]
    else:
        published_failed = [f for f in allowed_findings if f.get("is_failed")]
        non_published_failed = []
    about_outlet = [f for f in findings if f["claim_source"] == "about_outlet"]
    return {
        "total_checks": int(total_hdr) if isinstance(total_hdr, int) else len(findings),
        "failed_checks": (
            len(published_failed)
            if attributed
            else int(failed_hdr) if isinstance(failed_hdr, int) else sum(1 for f in findings if f["is_failed"])
        ),
        "failed_published_by_outlet": len(published_failed),
        "published_failed": published_failed,
        "non_published_failed": non_published_failed,
        "allowed_findings": allowed_findings,
        "blocked_findings": blocked_findings,
        "about_outlet": about_outlet,
        "attributed": attributed,
        "findings": findings,
        "coverage_sufficient": coverage,
    }


def _render_fact_check_finding(finding: dict, mode: str = "default") -> None:
    verdict = _coalesce(finding.get("verdict"), "Unknown")
    is_non_published = mode in {"about", "non_published"} or (
        finding.get("is_failed")
        and finding.get("published_by_outlet") is False
        and finding.get("claim_source") in {"about_outlet", "third_party", "unknown"}
    )
    # A failed claim not published by the outlet is not styled as the outlet's failure.
    failed = (not is_non_published) and (bool(finding.get("is_failed")) or _is_failed_fact_check_verdict(verdict))
    v_color = "#b44232" if failed else "#5d6461"
    claim = _clean_evidence_text(finding.get("claim_summary") or finding.get("summary"), 400)
    url = _coalesce(finding.get("url"), "")
    source = _coalesce(finding.get("source_site"), "") or _fact_checker_from_url(url)
    source_html = f'<span class="claim-pill">Fact-checker: {_esc(source)}</span>' if source else ""
    link_html = (
        f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer"'
        f' style="display:block;margin-top:4px;font-size:0.82em;color:#4a90d9;">View source</a>'
        if url else ""
    )
    st.markdown(
        f'<div class="evidence-card"><div class="claim-meta">'
        f'<span class="rating-pill" style="background:{v_color};">{_esc(verdict)}</span>{source_html}</div>'
        f'<p><strong>Checked claim:</strong> {_esc(claim or "Claim summary unavailable")}</p>{link_html}</div>',
        unsafe_allow_html=True,
    )


def _load_article_texts(source_url: str) -> dict[str, str]:
    """Return {article_url: full_text} from article_cache for this outlet's domain."""
    try:
        from article_cache import ArticleCache
        from urllib.parse import urlparse as _urlparse
        parsed = _urlparse(source_url if source_url.startswith("http") else f"https://{source_url}")
        domain = (parsed.netloc or parsed.path).lstrip("www.")
        cache_path = os.path.join(os.path.dirname(__file__), "article_cache", domain, "articles.json")
        if not os.path.exists(cache_path):
            return {}
        with open(cache_path, encoding="utf-8") as f:
            articles = json.load(f)
        return {a["url"]: a.get("text", "") for a in articles if a.get("url") and a.get("text")}
    except Exception:
        return {}


_COMPONENT_KEYWORDS: dict[str, list[str]] = {
    "economic": ["economic", "fiscal", "tax", "welfare", "gdp", "trade", "market",
                 "financial", "capitali", "socialist", "fossil", "energy policy", "spending"],
    "social": ["social", "cultural", "diversity", "race", "gender", "lgbtq", "abortion",
               "immigration", "religion", "human.right", "equity", "anti.racism"],
    "news_reporting": ["straight news", "news reporting", "news balance", "news item",
                       "factual report", "headline", "framing", "selection", "sourcing",
                       "balance", "counterpoint", "verification"],
    "editorial": ["editorial", "opinion piece", r"op[\-\s]ed", "commentary",
                  "editorial bias", "editorial score"],
}


def _extract_component_sentences(reasoning: str, component: str) -> tuple[list[str], list[int]]:
    """Return (relevant_sentences, article_refs_in_those_sentences) for a bias component."""
    keywords = _COMPONENT_KEYWORDS.get(component, [])
    sentences = re.split(r'(?<=[.!?])\s+', reasoning.strip())
    relevant = [s for s in sentences if any(re.search(kw, s, re.IGNORECASE) for kw in keywords)]

    combined = " ".join(relevant)
    nums: list[int] = []
    for ref_m in re.finditer(r"\bArticles?\s+([\d][\d,\s–—\-]*)", combined, re.IGNORECASE):
        chunk = ref_m.group(1).strip().rstrip(",")
        for range_m in re.finditer(r"(\d+)\s*[–—\-]\s*(\d+)", chunk):
            a, b = int(range_m.group(1)), int(range_m.group(2))
            if 0 < b - a < 20:
                nums.extend(range(a, b + 1))
        for n_str in re.findall(r"\d+", chunk):
            n = int(n_str)
            if 1 <= n <= 50:
                nums.append(n)
    return relevant, sorted(set(nums))


def resolve_article_refs(refs: list[int], evidence_sources: list[dict]) -> list[dict]:
    return [evidence_sources[n - 1] for n in refs if 1 <= n <= len(evidence_sources)]


def _render_cited_article(art: dict, number: int | None = None, article_texts: dict | None = None) -> None:
    url = _coalesce(art.get("url"), "")
    title = _coalesce(art.get("title"), f"Article {number}" if number else "Article")
    label = f"Article {number}: {title}" if number else title
    link_html = (
        f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{_esc(label)}</a>'
        if url else f"<strong>{_esc(label)}</strong>"
    )
    st.markdown(f'<div class="evidence-card">{link_html}</div>', unsafe_allow_html=True)

    full_text = (article_texts or {}).get(url, "")
    if full_text:
        quote = _clean_evidence_text(full_text, 600)
        st.markdown(f"> {quote}")
        if len(full_text) > 600:
            with st.expander("Show full article text", expanded=False):
                st.markdown(full_text)
    else:
        snippet = _clean_evidence_text(art.get("snippet") or art.get("text"), 400)
        if snippet:
            st.markdown(f"> {snippet}")


# ---------------------------------------------------------------------------
# Debug-based evidence audit renderer
# ---------------------------------------------------------------------------

def render_evidence_from_debug(rd: dict) -> None:
    sections = split_debug_evidence(rd.get("_debug_pipeline_evidence"))
    evidence_sources = rd.get("evidence_sources") or []
    article_texts = _load_article_texts(rd.get("source_url", ""))

    if not sections:
        st.info("No structured debug log in this record. Showing report analysis only.")
        st.markdown(_section_text(rd, "analysis"))
        return

    # Article coverage warning — shown when the pipeline had very few articles to work with
    _article_count = len(evidence_sources)
    if _article_count == 0:
        st.warning(
            "**Limited evidence base:** No articles were retrieved for this outlet. "
            "Analysis below is based on About-page text only. "
            "This may reflect a paywall, bot-protection, or sparse web presence — "
            "conclusions should be interpreted with caution.",
        )
    elif _article_count == 1:
        st.warning(
            "**Limited evidence base:** Only 1 article was retrieved. "
            "Bias conclusions based on a single article may not represent this outlet's typical coverage.",
        )

    # Parse all sections upfront so later blocks can reference them
    bias = parse_editorial_bias_section(sections.get("Editorial Bias Analysis", ""))

    # A. Editorial Bias
    bias_text = sections.get("Editorial Bias Analysis", "")
    if bias_text:
        st.markdown("#### Editorial Bias Analysis")
        st.caption("Source: pipeline log — Editorial Bias Analyzer")
        ll_label = "Loaded language: Detected" if bias["uses_loaded_language"] else "Loaded language: Not detected"
        ll_color = "#b44232" if bias["uses_loaded_language"] else "#2f7d4f"
        st.markdown(
            f'<span class="rating-pill" style="background:{ll_color};">{_esc(ll_label)}</span>',
            unsafe_allow_html=True,
        )
        if bias["ideology_summary"]:
            st.markdown(f"> {bias['ideology_summary']}")
        if bias["uses_loaded_language"]:
            ll_examples = _resolve_loaded_examples(rd, evidence_sources, bias.get("reasoning", ""))
            if ll_examples:
                with st.expander("Loaded-language examples (sourced from articles)", expanded=False):
                    for ex in ll_examples:
                        quote_block = f'> *"{_esc(ex["quote"])}"*'
                        if ex.get("art_url"):
                            quote_block += (
                                f'\n>\n> — <a href="{html.escape(ex["art_url"], quote=True)}"'
                                f' target="_blank" rel="noopener noreferrer"'
                                f' style="color:#4a90d9;">{_esc(ex["art_title"])}</a>'
                            )
                            st.markdown(quote_block, unsafe_allow_html=True)
                        else:
                            st.markdown(quote_block)
        if bias["reasoning"]:
            with st.expander("Full analyzer reasoning", expanded=False):
                st.markdown(bias["reasoning"])
        cited_raw = resolve_article_refs(bias["article_refs"], evidence_sources)
        cited_pairs = [(n, a) for n, a in zip(bias["article_refs"], cited_raw) if _is_article(a)]
        if cited_pairs:
            refs_label = ", ".join(str(n) for n, _ in cited_pairs[:8])
            if len(cited_pairs) > 8:
                refs_label += f" … +{len(cited_pairs) - 8} more"
            st.markdown(f"**Pipeline cited {len(cited_pairs)} article(s)** in its reasoning (Articles {refs_label}):")
            for ref_num, art in cited_pairs[:6]:
                _render_cited_article(art, ref_num, article_texts)
            if len(cited_pairs) > 6:
                with st.expander(f"Show remaining {len(cited_pairs) - 6} cited articles", expanded=False):
                    for ref_num, art in cited_pairs[6:]:
                        _render_cited_article(art, ref_num, article_texts)
        elif evidence_sources:
            article_pages = [a for a in evidence_sources if _is_article(a)]
            if article_pages:
                st.markdown(f"**All {len(article_pages)} sampled article(s)** (analyzer referenced them collectively):")
                for i, art in enumerate(article_pages[:6], 1):
                    _render_cited_article(art, i, article_texts)
                if len(article_pages) > 6:
                    with st.expander(f"Show remaining {len(article_pages) - 6} articles", expanded=False):
                        for i, art in enumerate(article_pages[6:], 7):
                            _render_cited_article(art, i, article_texts)

    st.divider()

    # B. Sourcing
    sourcing_text = sections.get("Sourcing Analysis", "")
    st.markdown("#### Sourcing Analysis")
    st.caption("Source: pipeline log — Sourcing Analyzer")
    if sourcing_text:
        sourcing = parse_sourcing_section(sourcing_text)
        if sourcing["score"] is not None:
            st.metric("Sourcing risk score (0 = best, 10 = worst)", f"{sourcing['score']:.1f}")
            st.caption("Scale: 0–2 excellent · 3–4 good · 5–6 mixed · 7–8 poor · 9–10 very poor")
        if sourcing["vague_examples"]:
            st.markdown("**Verbatim vague-attribution examples found by the analyzer:**")
            for ex in sourcing["vague_examples"]:
                st.markdown(f'> {_clean_evidence_text(ex, 260)}')
        if sourcing["reasoning"]:
            with st.expander("Full sourcing reasoning", expanded=False):
                st.markdown(sourcing["reasoning"])
    else:
        st.caption("No sourcing analysis found in pipeline evidence for this outlet.")

    st.divider()

    # C. Fact Checks
    fc_text = sections.get("Fact Check Search", "")
    fc = _structured_fact_checks(rd, parse_fact_check_section(fc_text))
    if fc_text or rd.get("fact_check_result"):
        st.markdown("#### Fact Check Search")
        st.caption("Source: pipeline log — FactCheckSearcher (IFCN sites)")
        attributed = fc.get("attributed", False)
        c1, c2 = st.columns(2)
        c1.metric("Checks found", fc["total_checks"])
        c2.metric(
            "Failed checks published by outlet" if attributed else "Failed checks",
            fc["failed_published_by_outlet"] if attributed else fc["failed_checks"],
        )
        if fc["total_checks"] == 0:
            factual = rd.get("factual_reporting", "").upper()
            if factual in ("HIGH", "VERY HIGH", "MOSTLY FACTUAL"):
                st.info(
                    "No fact-check failures found in our IFCN search. "
                    "For this outlet's factuality tier this is consistent with reliable reporting.",
                    icon="✅",
                )
            else:
                st.info(
                    "No fact-check failures found in our IFCN search. "
                    "Our search covers 8 IFCN-certified fact-checkers; "
                    "smaller or niche outlets may have limited coverage regardless of accuracy.",
                    icon="ℹ️",
                )
        allowed_findings = fc.get("allowed_findings", [])
        blocked_count = len(fc.get("blocked_findings", []))
        if blocked_count:
            st.caption(f"ⓘ {blocked_count} result(s) from bias-rating aggregators (e.g. MBFC, AllSides) excluded — not IFCN fact-checkers.")
        non_published_failed = fc.get("non_published_failed", []) if attributed else []
        non_published_failed_ids = {id(f) for f in non_published_failed}
        main_findings = [f for f in allowed_findings if id(f) not in non_published_failed_ids] if attributed else allowed_findings
        if main_findings:
            st.markdown("**Findings:**")
            for finding in main_findings:
                _render_fact_check_finding(finding)
        if non_published_failed:
            st.markdown("**False or Misleading Claims Involving This Outlet (Not Published by It):**")
            for finding in non_published_failed:
                _render_fact_check_finding(finding, mode="non_published")

    st.divider()

    # D. Bias Component Scores
    comp = bias.get("component_scores") if isinstance(bias, dict) else {}
    reasoning_text = bias.get("reasoning", "") if isinstance(bias, dict) else ""
    st.markdown("#### Bias Component Scores")
    st.caption("Source: pipeline log — Editorial Bias Analyzer (MBFC 4-category methodology, weights: Economic 35% · Social 35% · News 15% · Editorial 15%)")
    if reasoning_text:
        _COMP_META = [
            ("economic",       "Economic policy",   "35%", "−10 = communist · 0 = mixed · +10 = laissez-faire"),
            ("social",         "Social policy",     "35%", "−10 = progressive · +10 = traditional conservative"),
            ("news_reporting", "News reporting",    "15%", "How balanced are straight news articles?"),
            ("editorial",      "Editorial/opinion", "15%", "How biased are opinion pieces? (0 if none sampled)"),
        ]
        for key, label, weight, help_text in _COMP_META:
            score = comp.get(key)
            score_str = f" — {score:+.1f}" if score is not None else ""
            with st.expander(f"**{label} ({weight}){score_str}**", expanded=False):
                st.caption(help_text)
                sents, comp_refs = _extract_component_sentences(reasoning_text, key)
                if sents:
                    st.markdown("**Why this score — analyzer reasoning:**")
                    for sent in sents:
                        st.markdown(f"> {sent.strip()}")
                else:
                    st.markdown("**Why this score — full analyzer reasoning:**")
                    preview = reasoning_text[:800] + ("…" if len(reasoning_text) > 800 else "")
                    st.markdown(f"> {preview}")

                # Article evidence: use refs from matched sentences → all bias refs → all sources
                ref_nums = comp_refs if comp_refs else (bias.get("article_refs") or [])
                if ref_nums:
                    arts = resolve_article_refs(ref_nums, evidence_sources)
                    art_pairs = [(n, a) for n, a in zip(ref_nums, arts) if _is_article(a)]
                else:
                    # Guardian-style: reasoning cites all articles collectively — show all sources
                    art_pairs = [(i, a) for i, a in enumerate(evidence_sources, 1) if _is_article(a)]
                if art_pairs:
                    st.markdown(f"**Articles ({len(art_pairs)}):**")
                    for ref_num, art in art_pairs[:4]:
                        _render_cited_article(art, ref_num, article_texts)
                    if len(art_pairs) > 4:
                        with st.expander(f"Show {len(art_pairs) - 4} more", expanded=False):
                            for ref_num, art in art_pairs[4:]:
                                _render_cited_article(art, ref_num, article_texts)
    else:
        st.caption("No analyzer reasoning available for this outlet.")

    st.divider()

    # E. History
    history_text = sections.get("History", "")
    history = parse_history_section(history_text)
    has_history = history["founded"] or history["original_name"] or history["key_events"]
    st.markdown("#### Outlet History")
    st.caption("Source: pipeline log — History & Longevity Analyzer")
    if has_history:
        meta_parts = []
        if history["founded"]:
            meta_parts.append(f"Founded: **{history['founded']}**")
        if history["original_name"]:
            meta_parts.append(f"Original name: *{_coalesce(history['original_name'])}*")
        if meta_parts:
            st.markdown(" · ".join(meta_parts))
        if history["key_events"]:
            st.markdown("**Key events:**")
            for evt in history["key_events"][:6]:
                st.markdown(f"- {_clean_evidence_text(evt, 200)}")
        if history["summary"]:
            with st.expander("Full history summary", expanded=False):
                st.markdown(history["summary"])
    else:
        st.caption("No history data found in pipeline evidence for this outlet.")

    # F. Ownership
    ownership_text = sections.get("Ownership", "")
    ownership = parse_ownership_section(ownership_text)
    has_ownership = any(ownership.get(k) for k in ("owner", "parent_company", "funding_model", "headquarters"))
    st.markdown("#### Ownership & Funding")
    st.caption("Source: pipeline log — History & Longevity Analyzer")
    if has_ownership:
        fields = [
            ("Owner", ownership.get("owner")),
            ("Parent company", ownership.get("parent_company")),
            ("Funding model", ownership.get("funding_model")),
            ("Headquarters", ownership.get("headquarters")),
        ]
        html_parts = "".join(
            f"<p><strong>{label}:</strong> {_esc(val)}</p>"
            for label, val in fields if val
        )
        st.markdown(f'<div class="profile-card">{html_parts}</div>', unsafe_allow_html=True)
    else:
        st.caption("No ownership data found in pipeline evidence for this outlet.")


def render_analyzer_trace(rd: dict) -> None:
    st.markdown("### Analyzer Trace")
    st.markdown(
        '<div class="timeline-grid">'
        + "".join(f'<div class="timeline-item"><strong>{_esc(name)}</strong><span>{_esc(desc)}</span></div>' for name, desc in ANALYZERS)
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("### Pipeline Milestones")
    milestones = [
        ("Input", "Outlet homepage URL"),
        ("Scrape", "Recent articles and site metadata"),
        ("Search", "History, ownership, and external evidence"),
        ("Analyze", "Nine specialized analyzers in parallel"),
        ("Aggregate", "Structured outlet profile"),
        ("Synthesize", "MBFC-style narrative report"),
    ]
    st.markdown(
        '<div class="timeline-grid">'
        + "".join(f'<div class="timeline-item"><strong>{_esc(name)}</strong><span>{_esc(desc)}</span></div>' for name, desc in milestones)
        + "</div>",
        unsafe_allow_html=True,
    )
    debug_sections = split_debug_evidence(rd.get("_debug_pipeline_evidence"))
    if not debug_sections:
        st.info("The compact cached report does not include a debug evidence log.")
        return
    st.markdown("### Structured Evidence Log")
    preferred = ["Pipeline Scores", "Editorial Bias Analysis", "Fact Check Search", "Sourcing Analysis", "History", "Ownership", "Analyzed Articles (15 Total)"]
    shown = set()
    for key in preferred:
        if key in debug_sections:
            with st.expander(key, expanded=key == "Pipeline Scores"):
                st.code(debug_sections[key][:6000], language="text")
            shown.add(key)
    for key, value in debug_sections.items():
        if key not in shown:
            with st.expander(key, expanded=False):
                st.code(value[:6000], language="text")


def render_system_overview(summary: dict) -> None:
    # A. Abstract
    st.markdown(
        '<div class="profile-card">'
        "<p><strong>Media Profiler</strong> is an automated pipeline for outlet-level "
        "media bias and factuality profiling. Given a news outlet's homepage URL, it "
        "runs nine specialized analyzers in parallel — covering editorial stance, "
        "sourcing quality, failed fact checks, transparency, pseudoscience signals, "
        "one-sidedness, traffic, media type, and opinion split — then synthesises a "
        "structured MBFC-style profile with a bias score (−10 to +10) and a factuality "
        "risk score (0–10).</p>"
        "<p>The system is benchmarked against "
        "<a href='https://mediabiasfactcheck.com' target='_blank'>Media Bias / Fact Check</a> "
        "ground-truth labels spanning the full bias and "
        "factuality spectrum. System mode outperforms LLM-only, article-only, and "
        "search-only baselines on all eight evaluation metrics.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # C. Pipeline architecture
    st.markdown("### Pipeline Architecture")
    st.graphviz_chart(
        """
digraph G {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
    edge [fontname="Helvetica", fontsize=9];
    url [label="Outlet URL", fillcolor="#e8f3f1"];
    scrape [label="Scrape articles\nand metadata", fillcolor="#f6efe8"];
    search [label="Batch web research\nhistory + ownership", fillcolor="#f6efe8"];
    pool [label="9 analyzer pool", fillcolor="#e9eef7"];
    aggregate [label="Aggregate structured\nreport data", fillcolor="#ecf5ea"];
    synth [label="LLM synthesis", fillcolor="#eef3f8"];
    report [label="Evidence workbench\n+ MBFC-style profile", fillcolor="#f7eeee"];
    url -> scrape -> search -> pool -> aggregate -> synth -> report;
}
"""
    )

    # D. Analyzer responsibilities
    st.markdown("### Nine Analyzers")
    st.table([{"Analyzer": name, "Responsibility": desc} for name, desc in ANALYZERS])

    # E. Scoring methodology
    st.markdown("### Scoring Methodology")
    col_bias, col_fact = st.columns(2)
    with col_bias:
        st.markdown(
            '<div class="profile-card"><strong>Bias Score</strong> &nbsp;'
            '<span style="color:var(--mp-muted);font-size:0.85em">−10 Far Left → +10 Far Right</span>',
            unsafe_allow_html=True,
        )
        st.table([
            {"Component": "Economic policy stance", "Weight": "35%", "Scale": "−10 communist · +10 laissez-faire"},
            {"Component": "Social policy stance",   "Weight": "35%", "Scale": "−10 progressive · +10 conservative"},
            {"Component": "News reporting balance", "Weight": "15%", "Scale": "−10 left-skewed · +10 right-skewed"},
            {"Component": "Editorial/opinion bias", "Weight": "15%", "Scale": "−10 left · +10 right (0 if no op-ed)"},
        ])
        st.markdown("</div>", unsafe_allow_html=True)
    with col_fact:
        st.markdown(
            '<div class="profile-card"><strong>Factuality Risk Score</strong> &nbsp;'
            '<span style="color:var(--mp-muted);font-size:0.85em">0 lowest risk → 10 highest risk</span>',
            unsafe_allow_html=True,
        )
        st.table([
            {"Component": "Failed fact checks",       "Weight": "40%", "Scale": "0–10"},
            {"Component": "Sourcing quality",         "Weight": "25%", "Scale": "0–10"},
            {"Component": "Transparency disclosure",  "Weight": "25%", "Scale": "0–10"},
            {"Component": "Bias / propaganda signals","Weight": "10%", "Scale": "0–10"},
        ])
        st.markdown("</div>", unsafe_allow_html=True)

    # F. MBFC taxonomy
    with st.expander("MBFC rating taxonomy", expanded=False):
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("**Bias labels**")
            for label, color in BIAS_COLOR.items():
                st.markdown(
                    f'<span class="rating-pill" style="background:{color};margin:2px 0;display:inline-block;">'
                    f"{_esc(label)}</span>",
                    unsafe_allow_html=True,
                )
        with tc2:
            st.markdown("**Factuality labels**")
            for label, color in FACT_COLOR.items():
                st.markdown(
                    f'<span class="rating-pill" style="background:{color};margin:2px 0;display:inline-block;">'
                    f"{_esc(label)}</span>",
                    unsafe_allow_html=True,
                )

    # G. Benchmark comparison table
    if summary:
        st.markdown("### System vs. Baselines")
        mode_labels = {"llm": "LLM-only", "articles": "Articles", "search": "Search", "system": "System (ours) ★"}
        rows = []
        for mode in ["llm", "articles", "search", "system"]:
            row = summary.get(mode)
            if row:
                rows.append({
                    "Mode": mode_labels.get(mode, mode),
                    "Bias MAE ↓": f"{row.get('bias_mae', 0):.2f}",
                    "Fact MAE ↓": f"{row.get('fact_mae', 0):.2f}",
                    "FC Detection ↑": f"{row.get('fc_det', 0):.1%}",
                    "FS Precision ↑": f"{row.get('factscore_precision', 0):.1%}",
                    "FS Recall ↑": f"{row.get('factscore_recall', 0):.1%}",
                    "Error Rate ↓": f"{row.get('error_rate', 0):.1%}",
                    "ROUGE-L ↑": f"{row.get('rougeL', 0):.1%}",
                })
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("↓ lower is better · ↑ higher is better · System mode uses all nine analyzers with full evidence pipeline")


def render_results(summary: dict, leaderboard: list[dict]) -> None:
    import pandas as pd

    st.markdown("### Benchmark Performance")
    render_benchmark_strip(summary)
    st.caption("System mode · MBFC ground-truth labels")
    st.divider()

    n = len(leaderboard)

    # Accuracy summary strip
    if n:
        b_exact = sum(1 for r in leaderboard if r["bias_delta"] == 0)
        b_near  = sum(1 for r in leaderboard if r["bias_delta"] <= 1)
        f_exact = sum(1 for r in leaderboard if r["fact_delta"] == 0)
        f_near  = sum(1 for r in leaderboard if r["fact_delta"] <= 1)
        cards = [
            ("Bias exact match",        f"{b_exact/n:.1%}", f"{b_exact}/{n} outlets"),
            ("Bias within ±1 level",    f"{b_near/n:.1%}",  f"{b_near}/{n} outlets"),
            ("Fact exact match",        f"{f_exact/n:.1%}", f"{f_exact}/{n} outlets"),
            ("Fact within ±1 level",    f"{f_near/n:.1%}",  f"{f_near}/{n} outlets"),
        ]
        st.markdown('<div class="metric-grid">' + "".join(_metric_card(*c) for c in cards) + "</div>", unsafe_allow_html=True)
        st.caption("Ordinal-label accuracy vs. MBFC ground truth · system mode")

    # Filter controls
    fc1, fc2 = st.columns(2)
    with fc1:
        correctness = st.selectbox(
            "Filter by correctness",
            ["All", "Bias exact ✓", "Bias wrong ✗", "Fact exact ✓", "Fact wrong ✗", "Both exact ✓", "Either wrong ✗"],
            key="lb_correctness",
        )
    with fc2:
        countries = ["All"] + sorted({r["country"] for r in leaderboard if r["country"]})
        country_f = st.selectbox("Country", countries, key="lb_country")

    # Apply filters
    filtered = leaderboard
    if correctness == "Bias exact ✓":
        filtered = [r for r in filtered if r["bias_delta"] == 0]
    elif correctness == "Bias wrong ✗":
        filtered = [r for r in filtered if r["bias_delta"] > 0]
    elif correctness == "Fact exact ✓":
        filtered = [r for r in filtered if r["fact_delta"] == 0]
    elif correctness == "Fact wrong ✗":
        filtered = [r for r in filtered if r["fact_delta"] > 0]
    elif correctness == "Both exact ✓":
        filtered = [r for r in filtered if r["bias_delta"] == 0 and r["fact_delta"] == 0]
    elif correctness == "Either wrong ✗":
        filtered = [r for r in filtered if r["bias_delta"] > 0 or r["fact_delta"] > 0]
    if country_f != "All":
        filtered = [r for r in filtered if r["country"] == country_f]

    st.caption(f"{len(filtered)} of {n} outlets · ✓ exact · ~ off by 1 · ✗ wrong")

    if not filtered:
        st.info("No outlets match the current filters.")
    else:
        def _icon(delta: int) -> str:
            return "✓" if delta == 0 else ("~" if delta == 1 else "✗")

        df = pd.DataFrame([{
            "Outlet": r["name"],
            "Country": r["country"],
            "Type": r["media_type"],
            "Gold Bias": r["gold_bias"],
            "Pred Bias": r["pred_bias"],
            "B": _icon(r["bias_delta"]),
            "Gold Fact": r["gold_factuality"],
            "Pred Fact": r["pred_factuality"],
            "F": _icon(r["fact_delta"]),
        } for r in filtered])

        event = st.dataframe(
            df,
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key="leaderboard_df",
            column_config={
                "B": st.column_config.TextColumn("B", help="Bias: ✓ exact · ~ off by 1 · ✗ wrong", width="small"),
                "F": st.column_config.TextColumn("F", help="Factuality: ✓ exact · ~ off by 1 · ✗ wrong", width="small"),
            },
        )

        sel = (event.selection or {}).get("rows", [])
        if sel:
            sel_name = filtered[sel[0]]["name"]
            st.info(f"Selected: **{sel_name}**")
            if st.button("Load full profile →", key="lb_load_btn") and sel_name in cached_outlets:
                _select_report(sel_name, cached_outlets)
                st.session_state["picked_outlet"] = sel_name
                st.session_state["leaderboard_df"] = {"selection": {"rows": [], "columns": []}}
                st.rerun()

    # System vs baselines table (collapsed)
    if summary:
        with st.expander("System vs. baselines (aggregate)", expanded=False):
            mode_labels = {"llm": "LLM-only", "articles": "Articles", "search": "Search", "system": "System ★"}
            agg_rows = []
            for mode in ["llm", "articles", "search", "system"]:
                row = summary.get(mode)
                if row:
                    agg_rows.append({
                        "Mode":       mode_labels.get(mode, mode),
                        "Bias MAE ↓": f"{row.get('bias_mae', 0):.2f}",
                        "Fact MAE ↓": f"{row.get('fact_mae', 0):.2f}",
                        "FC Det ↑":   f"{row.get('fc_det', 0):.1%}",
                        "FS Prec ↑":  f"{row.get('factscore_precision', 0):.1%}",
                        "FS Rec ↑":   f"{row.get('factscore_recall', 0):.1%}",
                        "Err Rate ↓": f"{row.get('error_rate', 0):.1%}",
                        "ROUGE-L ↑":  f"{row.get('rougeL', 0):.1%}",
                    })
            st.dataframe(agg_rows, hide_index=True, width="stretch")
            st.caption("↓ lower is better · ↑ higher is better")


# ---------------------------------------------------------------------------
# Outlet comparison
# ---------------------------------------------------------------------------

def _render_comparison_panel(rd: dict, label: str) -> None:
    bias = _coalesce(rd.get("bias_rating"))
    factual = _coalesce(rd.get("factual_reporting"))
    cred = _coalesce(rd.get("credibility_rating"))

    st.markdown(f"#### {_esc(label)}")
    st.markdown(
        f'<div class="badge-row">{_badge(bias, BIAS_COLOR)}{_badge(factual, FACT_COLOR)}{_badge(cred, CRED_COLOR)}</div>',
        unsafe_allow_html=True,
    )
    render_bias_bar("Bias score", rd.get("bias_score"), _label_color(bias, BIAS_COLOR))
    render_factuality_bar("Factuality risk", rd.get("factual_score"), _label_color(factual, FACT_COLOR))

    st.markdown(
        f"<p><strong>Country:</strong> {_esc(rd.get('country'))}</p>"
        f"<p><strong>Press freedom:</strong> {_esc(rd.get('country_freedom_rating'))}</p>"
        f"<p><strong>Media type:</strong> {_esc(rd.get('media_type'))}</p>"
        f"<p><strong>Traffic:</strong> {_esc(rd.get('traffic_popularity'))}</p>",
        unsafe_allow_html=True,
    )

    sections = split_debug_evidence(rd.get("_debug_pipeline_evidence"))
    bias_p = parse_editorial_bias_section(sections.get("Editorial Bias Analysis", ""))
    sourcing = parse_sourcing_section(sections.get("Sourcing Analysis", ""))
    fc = parse_fact_check_section(sections.get("Fact Check Search", ""))
    history = parse_history_section(sections.get("History", ""))
    ownership = parse_ownership_section(sections.get("Ownership", ""))

    st.markdown("**Evidence highlights**")
    ll_label = "Loaded language: Detected" if bias_p["uses_loaded_language"] else "Loaded language: Not detected"
    ll_color = "#b44232" if bias_p["uses_loaded_language"] else "#2f7d4f"
    st.markdown(
        f'<span class="rating-pill" style="background:{ll_color};">{_esc(ll_label)}</span>',
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    if sourcing["score"] is not None:
        m1.metric("Sourcing risk", f"{sourcing['score']:.1f}", help="0 best · 10 worst")
    m2.metric("FC found", fc["total_checks"])
    m3.metric("FC failed", fc["failed_checks"])

    comp = bias_p.get("component_scores") or {}
    if comp:
        st.markdown("**Bias components**")
        ca, cb = st.columns(2)
        ca.metric("Economic (35%)", f"{comp['economic']:+.1f}")
        cb.metric("Social (35%)", f"{comp['social']:+.1f}")
        cc, cd = st.columns(2)
        cc.metric("News (15%)", f"{comp['news_reporting']:+.1f}")
        cd.metric("Editorial (15%)", f"{comp['editorial']:+.1f}")

    meta = []
    if history.get("founded"):
        meta.append(f"Founded **{history['founded']}**")
    if ownership.get("owner"):
        meta.append(f"Owner: {_esc(ownership['owner'])}")
    if ownership.get("funding_model"):
        meta.append(f"Funding: {_esc(ownership['funding_model'])}")
    if meta:
        st.markdown(" · ".join(meta))

    if bias_p["ideology_summary"]:
        with st.expander("Ideology summary", expanded=False):
            st.markdown(bias_p["ideology_summary"])


def render_comparison(outlets: dict[str, dict]) -> None:
    outlet_names = sorted(outlets.keys())
    if len(outlet_names) < 2:
        st.info("Need at least two cached outlets to compare.")
        return

    ca, cb = st.columns(2)
    with ca:
        name_a = st.selectbox("Outlet A", outlet_names, key="cmp_a")
    with cb:
        b_default = next((n for n in outlet_names if n != name_a), outlet_names[0])
        b_idx = outlet_names.index(b_default)
        name_b = st.selectbox("Outlet B", outlet_names, index=b_idx, key="cmp_b")

    if name_a == name_b:
        st.warning("Select two different outlets to compare.")
        return

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        _render_comparison_panel(outlets[name_a], name_a)
    with col_b:
        _render_comparison_panel(outlets[name_b], name_b)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

cached_outlets = load_cached_outlets()
benchmark_summary = load_benchmark_summary()
leaderboard_data = load_leaderboard_data()

with st.sidebar:
    st.markdown("## Media Profiler")

inject_theme()

default_name = _default_outlet_name(cached_outlets)
if default_name and "picked_outlet" not in st.session_state:
    st.session_state["picked_outlet"] = default_name
if default_name and "report" not in st.session_state:
    _select_report(default_name, cached_outlets)

live_cache_key: dict[str, object] | None = None
live_cached_result: dict | None = None
live_primary_btn = False
live_rerun_btn = False

with st.sidebar:
    st.markdown("### Live URL analysis")
    url_input = st.text_input("Outlet URL", placeholder="https://www.bbc.com", key="url_input")
    live_url_text = url_input.strip()
    if live_url_text:
        live_cache_key = _live_cache_key(live_url_text)
        live_cached_result = load_live_cached_run(live_url_text) if live_cache_key else None

    if live_cached_result:
        domain = _coalesce((live_cache_key or {}).get("domain"), "this outlet")
        st.caption(f"Saved live run available for {domain}.")
        live_primary_btn = st.button("Load saved run", type="primary", width="stretch")
        live_rerun_btn = st.button("Rerun pipeline", width="stretch")
    else:
        if live_url_text and live_cache_key is None:
            st.caption("Enter a valid outlet domain or URL.")
        live_primary_btn = st.button("Run live pipeline", type="primary", width="stretch")

    st.divider()
    st.markdown("### Browse Outlets")
    query = st.text_input("Search", placeholder="Guardian, science, low credibility…")

    with st.expander("Filters", expanded=False):
        bias_filter   = st.selectbox("Bias",        ["All"] + _unique_values(cached_outlets, "bias_rating"))
        factual_filter= st.selectbox("Factuality",  ["All"] + _unique_values(cached_outlets, "factual_reporting"))
        cred_filter   = st.selectbox("Credibility", ["All"] + _unique_values(cached_outlets, "credibility_rating"))
        media_filter  = st.selectbox("Media type",  ["All"] + _unique_values(cached_outlets, "media_type"))

    active_filters = [
        v for v in [
            bias_filter   if bias_filter   != "All" else None,
            factual_filter if factual_filter != "All" else None,
            cred_filter   if cred_filter   != "All" else None,
            media_filter  if media_filter  != "All" else None,
        ] if v
    ]
    if active_filters:
        st.caption("Active: " + " · ".join(active_filters))

    filtered_names = filtered_outlet_names(cached_outlets, query, bias_filter, factual_filter, cred_filter, media_filter)
    if filtered_names and st.session_state.get("picked_outlet") not in filtered_names:
        st.session_state["picked_outlet"] = filtered_names[0]
    if filtered_names:
        picked = st.selectbox("Outlet", filtered_names, key="picked_outlet")
        cur = cached_outlets.get(picked, {})
        st.markdown(
            f'<div style="margin:-6px 0 4px;">'
            f'{_badge(_coalesce(cur.get("bias_rating")), BIAS_COLOR)}'
            f'&nbsp;{_badge(_coalesce(cur.get("factual_reporting")), FACT_COLOR)}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        picked = None
        st.warning("No cached reports match the current filters.")

    st.caption(f"{len(filtered_names)} of {len(cached_outlets)} outlets")
    st.button("Reset to default", width="stretch", on_click=_reset_to_default, args=(cached_outlets,))

    st.markdown("### Quick picks")
    _QUICK_PICKS = [
        ("Balanced",      "LEAST BIASED"),
        ("Left",          "LEFT / LEFT-CENTER"),
        ("Far Left",      "FAR LEFT / EXTREME LEFT"),
        ("Right",         "RIGHT / RIGHT-CENTER"),
        ("Far Right",     "FAR RIGHT / EXTREME RIGHT"),
        ("High factuality","HIGH factuality"),
        ("Low factuality", "LOW / VERY LOW factuality"),
        ("Pro-science",   "PRO-SCIENCE"),
    ]
    for label, _hint in _QUICK_PICKS:
        target = _find_quick_pick(cached_outlets, label)
        btn_label = f"{label}" if not target else f"{label}: {target}"
        st.button(
            btn_label,
            key=f"quick_{label}",
            disabled=target is None,
            width="stretch",
            on_click=_select_report,
            args=(target or "", cached_outlets),
        )

live_url_text = url_input.strip()
should_load_live_cache = bool(live_primary_btn and live_url_text and live_cached_result)
should_warn_invalid_live_url = bool(
    (live_primary_btn or live_rerun_btn) and live_url_text and live_cache_key is None
)
should_run_live = bool(
    live_url_text
    and live_cache_key is not None
    and (live_rerun_btn or (live_primary_btn and live_cached_result is None))
)

if should_load_live_cache and live_cached_result:
    st.session_state["report"] = live_cached_result
    if picked is not None:
        st.session_state["last_picked"] = picked
    st.session_state["report_source"] = "live_cache"
    st.rerun()

if should_warn_invalid_live_url:
    st.warning("Enter a valid outlet URL, such as https://www.bbc.com.")
    st.stop()

if should_run_live:
    st.markdown("## Live analysis")
    with st.spinner("Running the 9-analyzer pipeline"):
        result = run_live(live_url_text)
    if result:
        save_live_cached_run(live_url_text, result)
        st.session_state["report"] = result
        if picked is not None:
            st.session_state["last_picked"] = picked
        st.session_state["report_source"] = "live"
        st.rerun()
    st.stop()

if picked and picked in cached_outlets:
    if picked != st.session_state.get("last_picked"):
        _apply_report(picked, cached_outlets)

report_to_render: dict | None = st.session_state.get("report")
if not report_to_render:
    st.info("No cached reports were found. Add the cached JSONL file or run a live analysis.")
    st.stop()

source_label = {
    "cached": "Cached system output",
    "live": "Fresh live system output",
    "live_cache": "Saved live system output",
}.get(st.session_state.get("report_source", "cached"), "Fresh live system output")
render_hero(report_to_render, source_label)

tab_overview, tab_evidence, tab_trace, tab_compare = st.tabs(
    ["Overview", "Evidence", "Analyzer Trace", "Compare"]
)
with tab_overview:
    render_overview(report_to_render, benchmark_summary)
with tab_evidence:
    render_evidence(report_to_render)
with tab_trace:
    render_analyzer_trace(report_to_render)
with tab_compare:
    render_comparison(cached_outlets)
