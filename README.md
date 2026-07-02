# SCOPE: Source Credibility and Outlet Profiling Engine

SCOPE is a **multi-agent system** that profiles a news media outlet for **political
bias**, **factual reliability**, and **overall credibility** from just its URL.
Given an outlet, it coordinates specialized agents for article acquisition,
content analysis, external evidence search, scoring, and report generation,
following a [Media Bias/Fact Check (MBFC)](https://mediabiasfactcheck.com/methodology/)–style
methodology — while keeping every intermediate piece of evidence and agent
output inspectable through a structured demonstration interface.

Media profiling is treated not as a single classification call but as a
multi-step analytical workflow — retrieval, extraction, comparison, reasoning,
and explanation — decomposed across agents so that coverage improves and the
profiling process stays transparent and auditable.

On a benchmark of **media outlets**, SCOPE outperforms LLM-only,
article-only, search-only, and media-background-check baselines on most metrics
— achieving the lowest bias and factuality MAE, perfect failed-fact-check
detection, and the lowest factual error rate, at roughly **$0.054 per outlet**.

---

## Quick start

Requires **Python 3.12** (pinned in `.python-version`).

```bash
git clone https://github.com/mbaisbay/SCOPE.git
cd SCOPE

# Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate            # Windows (PowerShell): .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Launch the demo
streamlit run demo_app.py            # opens http://localhost:8501
```

The demo **opens instantly on cached results** — no API key
required to browse the profiles, the benchmark leaderboard, and the evidence
audit.

### Optional: profile a live URL

To run the full pipeline on a new outlet (live scraping + LLM analysis with
GPT-5 mini), set an OpenAI API key:

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```

or export it in your shell (`export OPENAI_API_KEY="sk-..."`; PowerShell:
`$env:OPENAI_API_KEY="sk-..."`). The key is read from the environment only (see
`config.py`); it is never stored in the repository, and `.env` is git-ignored.

---

## Demonstration interface

SCOPE presents each outlet profile as a four-tab report:

- **Overview** — the three MBFC-style verdict badges (bias, factuality,
  credibility), the underlying numeric scores, outlet metadata (country, media
  type, traffic tier, press-freedom rating), and a generated narrative summary
  with collapsible history, ownership, and analysis sections.
- **Evidence** — the full pipeline audit: verbatim article quotes linked to each
  analyzer finding, detected propaganda-technique spans with confidence scores,
  and external fact-check results.
- **Analyzer Trace** — the structured, schema-constrained output of each agent.
- **Compare** — two outlets side by side for direct comparison.

---

## System architecture

Given an outlet URL, a **deterministic orchestrator** coordinates four agent
groups in a fixed order, checks whether the collected evidence is sufficient,
and then computes the bias, factuality, and credibility scores:

1. **Acquisition agent** — scrapes and caches up to 15 articles per outlet
   (HTTP + BeautifulSoup). Candidate URLs are discovered from RSS feeds,
   sitemaps, and homepage links, then scored by a hard-news heuristic
   (politics / economy / world news) and fetched with randomized rate-limiting.
2. **Content-analysis agents** — nine typed, rubric-guided analyzers run over the
   collected articles: six content analyzers (editorial bias, transparency,
   one-sidedness, pseudoscience, fact-check history, sourcing quality) and three
   metadata analyzers (traffic/longevity, media type, opinion). Each returns a
   structured judgment with a label, score, rationale, and supporting evidence.
3. **Evidence-search agent** — uses DuckDuckGo to retrieve external information
   about the outlet (history, ownership, funding, transparency ranking, proofs
   of failed fact-checks). Media-rating aggregators (MBFC, AllSides, NewsGuard)
   are explicitly excluded to prevent gold-label leakage.
4. **Report-generation agent** — synthesizes the intermediate outputs into an
   MBFC-style media profile with citations.

This modular design separates evidence collection, content analysis, scoring,
and report generation, making intermediate decisions inspectable and the final
profile auditable.

---

## Methodology

SCOPE follows the [MBFC](https://mediabiasfactcheck.com/methodology/) rubric.

**Bias** is a weighted score on a left–right scale from −10 to +10:

```
B = 0.35·E + 0.35·S + 0.15·N + 0.15·O
```

where *E* is economic ideology, *S* position on social values, *N* the balance
of straight news reporting, and *O* editorial/opinion bias. The economic and
social axes follow a two-axis political-science framework (Eysenck, 1954;
Bobbio, 1996): a 153-item editorial-stance questionnaire is distilled into the
scoring prompt.

**Factual reliability** is assessed separately:

```
F        = 0.40·C + 0.25·R + 0.25·T + 0.10·M
```

where *C* is failed fact checks, *R* sourcing quality, *T* transparency, and *M*
one-sidedness/omission/propaganda. Articles are additionally scanned for
propaganda and manipulation techniques following the
[Da San Martino et al. (2020)](https://aclanthology.org/2020.semeval-1.186/)
taxonomy (e.g., appeal to fear, flag waving, exaggeration).

**External knowledge** that does not depend on the outlet's own text:

- **Tranco top-1M** domain rank → traffic tier (High ≤10k, Medium ≤100k, Low
  ≤1M, Minimal otherwise).
- **RSF World Press Freedom Index (2025)** + **Freedom House (2024)** →
  per-country freedom tier; limited press freedom applies a −1 credibility
  adjustment, no press freedom a −2 adjustment.

---

## Results

Evaluated on a fixed set of **outlets** sampled from the MBFC catalogue
(round-robin across seven bias categories and six factuality levels), using
**GPT-5 mini**:

| Metric                       | SCOPE  | Note                                  |
| ---------------------------- | :----: | ------------------------------------- |
| Bias MAE ↓                   | **0.65** | vs 1.17 for the LLM-only baseline   |
| Factuality MAE ↓             | **0.67** | lowest of all systems               |
| Failed-fact-check detection ↑| **1.00** | perfect                             |
| FACTScore precision ↑        | **0.48** | highest                             |
| ROUGE-L ↑                    | **0.16** | highest                             |
| Error rate ↓                 | **0.52** | lowest                              |
| Cost per outlet              | **$0.054** | comparable to LLM-only ($0.050)   |

SCOPE predicts the exact bias class for 50% of outlets (90% within one ordinal
class) and the exact factuality class for 42% (90% within one class). It wins on
six of eight metrics against the LLM-only, articles-only, search-only, and
Media-Background-Check baselines; the search baseline leads only on the
recall-oriented metrics (fact recall, METEOR).

---

## What's included

```
SCOPE/
├── demo_app.py                 # Streamlit demonstration interface (entry point)
├── evaluators.py               # SystemRunner — orchestrates the agent pipeline
├── research.py                 # Evidence-search agent (ownership, history, freedom indices)
├── refactored_analyzers.py     # The nine content/metadata analyzers
├── scraper.py                  # Acquisition agent (article scraping)
├── search_backends.py          # DuckDuckGo / pluggable search routing
├── schemas.py                  # Pydantic schemas (structured agent output)
├── methodology.py              # MBFC 2025 scoring logic
├── article_cache.py            # Per-outlet article cache
├── cost_tracker.py             # Token / cost accounting
├── config.py                   # Configuration + API-key loading
├── ideology_question_bank.json # 17-domain / 153-item ideology reference bank (distilled into the bias prompt)
├── 2025.csv                    # RSF Press Freedom Index (reference data)
├── known_media_types.csv       # Media-type lookup table
├── FH_FIW.csv                  # Freedom House "Freedom in the World" scores
├── tranco_top1m.csv            # Tranco top-1M domains (traffic signal)
├── results/
│   ├── summary.json            # Cached benchmark summary
│   └── cached/
│       ├── gpt-5-mini-2025-08-07_system.jsonl   # Cached outlet profiles
│       ├── fact_check_reattribution.json        # Fact-check re-attribution sidecar
│       └── loaded_language_sources.json         # Loaded-language example sidecar
├── .streamlit/config.toml      # Theme
├── requirements.txt
├── pyproject.toml
├── .python-version
├── .env.example
├── .devcontainer/devcontainer.json  # Codespaces config
├── LICENSE                     # Apache-2.0
└── .gitignore
```

The four reference CSVs are used only by the optional **live-profiling** path.
If absent, the app still runs: `tranco_top1m.csv` auto-downloads on demand and
the press-freedom scores are simply omitted (no credibility penalty applied).

---

## Limitations & ethics

SCOPE is evaluated mainly on **U.S.-centric, English-language** outlets, which
are better represented in fact-checking and media-profiling resources;
performance may be less stable for non-Western, non-English, or low-visibility
outlets. Although media-rating platforms are filtered out of web search,
indirect leakage cannot be fully ruled out. Conspiracy and pseudoscience outlets
can be hard to place on a single left–right axis.

These scores describe politically sensitive attributes and could be misused to
dismiss or amplify sources. Every score is backed by inspectable intermediate
evidence and is withheld when the evidence-sufficiency gate is not met. SCOPE is
intended as an **auditable, assistive tool** for researchers, journalists, and
analysts — not as an arbiter of truth.

---

## License

Released under the [Apache License 2.0](LICENSE).

## Citation

Citation details will be added later.
