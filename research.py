"""
Research Module - Web Research and Comprehensive Profiling for MBFC Reports

This module gathers external context about media outlets and orchestrates
all analyzers to produce comprehensive MBFC-style reports.

Components:
1. MediaResearcher: Gathers history, ownership, and external analysis via web search
2. MediaProfiler: Orchestrates all analyzers to produce comprehensive reports

All LLM calls use LangChain's .with_structured_output() for type-safe responses.
"""

import csv
import logging
import os
import re
from datetime import date, datetime
from typing import Optional
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
# DDGS is no longer imported directly — search is delegated to search_backends
# ChatOpenAI is accessed via get_llm() imported from refactored_analyzers

from schemas import (
    CalibratedScores,
    ComprehensiveReportData,
    EditorialBiasResult,
    ExternalAnalysisItem,
    ExternalAnalysisLLMOutput,
    FactCheckAnalysisResult,
    HistoryLLMOutput,
    OneSidednessResult,
    OwnershipLLMOutput,
    PseudoscienceAnalysisResult,
    SourcingAnalysisResult,
    TransparencyResult,
)

from refactored_analyzers import (
    EditorialBiasAnalyzer,
    FactCheckSearcher,
    MediaTypeAnalyzer,
    OneSidednessAnalyzer,
    OpinionAnalyzer,
    PseudoscienceAnalyzer,
    SourcingAnalyzer,
    TrafficLongevityAnalyzer,
    TransparencyAnalyzer,
    get_llm,
)

logger = logging.getLogger(__name__)


# =============================================================================
# MediaResearcher - Web Research with Structured Output
# =============================================================================


class MediaResearcher:
    """
    Gathers external context for comprehensive MBFC-style reports.

    Uses DuckDuckGo search + LLM with structured output to gather:
    - History and founding information
    - Ownership and funding details
    - External criticism and analysis

    All LLM calls use .with_structured_output() for type-safe responses.
    """

    HISTORY_PROMPT = """You are extracting history and identity information about a media outlet from search results.

Extract the following if available:
- Official Name: The full, proper name of the organization (e.g., "The New York Times" instead of "nytimes", "The Associated Press" instead of "apnews", "Wall Street Journal" instead of "wsj")
- Founding year
- Founder name(s)
- Original name (if different from current)
- Key events in the outlet's history (ownership changes, scandals, major milestones)

Be conservative - only extract information that is clearly stated in the search results.
If information is not found, leave fields as null."""

    OWNERSHIP_PROMPT = """You are extracting ownership and funding information about a media outlet.

Extract the following if available:
- Current owner (person or entity)
- Parent company (if applicable)
- Funding model (advertising, subscription, public funding, nonprofit, mixed)
- Headquarters location (city, country)
- Country where the outlet is based (if headquarters is unknown, infer from evidence like currency, company suffix, domain, or content)

Be conservative - only extract information that is clearly stated in the search results.
If information is not found, leave fields as null. For country, you may infer from contextual clues if not explicitly stated."""

    EXTERNAL_ANALYSIS_PROMPT = """You are extracting external analyses and criticism about a media outlet.

Focus on:
- Media watchdog reviews (Ad Fontes, NewsGuard, etc.)
- Academic studies
- Journalism reviews (CJR, Nieman Lab, etc.)
- Major controversies or notable praise

IMPORTANT: Do NOT include Media Bias/Fact Check (MBFC) results — our system uses MBFC methodology independently and we should not cite them as an external analysis.

For each analysis found:
- Identify the source name
- Extract URL if available
- Summarize the key finding
- Categorize sentiment as: positive, negative, neutral, or mixed

Include up to 3-5 most relevant and credible analyses."""

    # Domains to exclude from search results (social media + bias aggregators)
    from config import EXCLUDED_DOMAINS as _EXCLUDED_DOMAINS
    SEARCH_BLACKLIST = {
        "facebook.com",
        "twitter.com",
        "x.com",
        "instagram.com",
        "tiktok.com",
        "pinterest.com",
        "linkedin.com",
        "reddit.com",
        "youtube.com",
    } | {d.lower() for d in _EXCLUDED_DOMAINS}

    # Common about page paths to try when scraping directly
    ABOUT_PAGE_PATHS = [
        "/about",
        "/about-us",
        "/about/",
        "/about-us/",
        "/corporate/about",
        "/company/about",
        "/aboutthebbc",
        "/corporate",
        "/who-we-are",
        "/team",
        "/staff",
        "/contact",
        "/our-story",
        "/masthead",
        "/editorial-team",
        "/advertise",
    ]

    # HTTP headers to mimic a browser
    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }

    # Wikipedia API endpoint and headers
    _WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
    _WIKIPEDIA_HEADERS = {
        "User-Agent": "MediaProfilingBot/1.0 (research tool; contact: media-profiling@example.com)",
        "Accept": "application/json",
    }

    def __init__(
        self,
        model: str = "gpt-5-mini-2025-08-07",
        temperature: float = 0.0,
        search_backend=None,
    ):
        """
        Initialize the MediaResearcher.

        Args:
            model: OpenAI model to use
            temperature: LLM temperature (0 for deterministic)
            search_backend: Optional SearchBackend instance (defaults to DDGSearchBackend)
        """
        self.history_llm = get_llm(model, temperature).with_structured_output(
            HistoryLLMOutput
        )
        self.ownership_llm = get_llm(model, temperature).with_structured_output(
            OwnershipLLMOutput
        )
        self.analysis_llm = get_llm(model, temperature).with_structured_output(
            ExternalAnalysisLLMOutput
        )
        self.name_llm = get_llm(model, temperature)
        if search_backend is not None:
            self.search_backend = search_backend
        else:
            from search_backends import DDGSearchBackend
            self.search_backend = DDGSearchBackend()
        # Cache for about page text (domain -> text) to avoid redundant scraping
        self._about_page_cache: dict[str, str] = {}

    def _extract_domain(self, url: str) -> str:
        """Extract the root domain from a URL."""
        parsed = urlparse(url if url.startswith("http") else f"https://{url}")
        domain = parsed.netloc or parsed.path
        domain = re.sub(r"^www\.", "", domain)
        domain = domain.split("/")[0]
        return domain.lower()

    def _extract_outlet_name(self, url: str) -> str:
        """
        Extract a human-readable outlet name from URL.

        Args:
            url: The outlet's URL

        Returns:
            Human-readable name derived from domain
        """
        domain = self._extract_domain(url)
        # Generate name from domain (e.g., "nytimes.com" -> "Nytimes")
        name = domain.split(".")[0]
        name = name.replace("-", " ").replace("_", " ")
        # Short names (<=4 chars) are likely acronyms — uppercase them (bbc -> BBC, cnn -> CNN, npr -> NPR)
        if len(name) <= 4:
            return name.upper()
        return name.title()

    def _scrape_about_page(self, domain: str) -> str:
        """
        Directly scrape the outlet's about page.

        Strategy:
        1. Check cache (avoid redundant scraping)
        2. Try common about page paths (/about, /about-us, etc.)
        3. If none work, fetch homepage and discover about links from <a> tags

        Args:
            domain: The outlet's domain (e.g., "bbc.com")

        Returns:
            About page text content, or empty string if not found
        """
        # Check cache first
        if domain in self._about_page_cache:
            return self._about_page_cache[domain]

        base_url = f"https://www.{domain}" if not domain.startswith("www.") else f"https://{domain}"

        # Strategy 1: Try common hardcoded paths
        for path in self.ABOUT_PAGE_PATHS:
            text = self._fetch_page_text(urljoin(base_url, path))
            if text:
                self._about_page_cache[domain] = text
                return text

        # Strategy 2: Discover about links from homepage
        try:
            resp = requests.get(base_url, headers=self._HEADERS, timeout=10, allow_redirects=True)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                about_links = set()
                for a in soup.find_all("a", href=True):
                    href = a["href"].lower()
                    # Look for links containing "about" in the path
                    if "about" in href and not href.startswith("mailto:"):
                        full_url = urljoin(base_url, a["href"])
                        # Only follow links on the same domain
                        if domain in full_url:
                            about_links.add(full_url)

                for about_url in list(about_links)[:5]:
                    text = self._fetch_page_text(about_url)
                    if text:
                        self._about_page_cache[domain] = text
                        return text
        except Exception as e:
            logger.debug(f"  - Homepage about link discovery failed: {e}")

        self._about_page_cache[domain] = ""
        return ""

    def _fetch_wikipedia(self, query: str) -> str:
        """
        Fetch Wikipedia extract for a query using the MediaWiki API.

        Tries the query as-is first, then falls back to simpler variants.
        Returns plain text extract (up to 5000 chars), or empty string.
        """
        search_variants = [query]
        # Add domain-based variant: "frontiersin.org" -> "Frontiers"
        if "." in query:
            domain_base = query.split(".")[0]
            if domain_base.lower() not in ("www", "the"):
                search_variants.append(domain_base)

        for search_term in search_variants:
            try:
                # Step 1: Search for the best matching article
                resp = requests.get(
                    self._WIKIPEDIA_API,
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": search_term,
                        "srlimit": 3,
                        "format": "json",
                    },
                    headers=self._WIKIPEDIA_HEADERS,
                    timeout=10,
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
                results = data.get("query", {}).get("search", [])
                if not results:
                    continue

                # Step 2: Get the extract of the best match
                page_title = results[0]["title"]
                resp2 = requests.get(
                    self._WIKIPEDIA_API,
                    params={
                        "action": "query",
                        "titles": page_title,
                        "prop": "extracts",
                        "exintro": False,
                        "explaintext": True,
                        "exlimit": 1,
                        "format": "json",
                    },
                    headers=self._WIKIPEDIA_HEADERS,
                    timeout=10,
                )
                if resp2.status_code != 200:
                    continue
                pages = resp2.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    extract = page.get("extract", "")
                    if extract and len(extract) > 100:
                        logger.info(f"  - Found Wikipedia article: '{page_title}' for query '{search_term}'")
                        return extract[:5000]
            except Exception as e:
                logger.debug(f"  - Wikipedia fetch failed for '{search_term}': {e}")

        return ""

    def _fetch_page_text(self, url: str) -> str:
        """
        Fetch a URL and return its text content.

        Args:
            url: URL to fetch

        Returns:
            Cleaned text content, or empty string if failed
        """
        try:
            resp = requests.get(url, headers=self._HEADERS, timeout=10, allow_redirects=True)
            if resp.status_code == 200 and "text/html" in resp.headers.get("content-type", ""):
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "header", "footer"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)
                if len(text) > 200:
                    logger.info(f"  - Found about page at {url}")
                    return text[:5000]
        except Exception as e:
            logger.debug(f"  - Page fetch failed for {url}: {e}")
        return ""

    _WIKIDATA_API = "https://www.wikidata.org/w/api.php"
    _WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

    def _fetch_wikidata(self, outlet_name: str, domain: str = "") -> dict:
        """Fetch structured facts from Wikidata for a media outlet.

        Queries Wikidata for properties like:
        - P127: owned by
        - P571: inception (founding date)
        - P112: founded by
        - P159: headquarters location
        - P749: parent organization

        Returns dict with keys: owner, founded, founder, headquarters, parent_org
        (all strings or None).
        """
        result = {
            "owner": None,
            "founded": None,
            "founder": None,
            "headquarters": None,
            "parent_org": None,
        }

        # Step 1: Search Wikidata for the entity
        search_terms = [outlet_name]
        if domain:
            brand = domain.split(".")[0].replace("-", " ").title()
            if brand.lower() != outlet_name.lower():
                search_terms.append(brand)

        entity_id = None
        for term in search_terms:
            try:
                resp = requests.get(
                    self._WIKIDATA_API,
                    params={
                        "action": "wbsearchentities",
                        "search": term,
                        "language": "en",
                        "limit": 3,
                        "format": "json",
                    },
                    headers=self._WIKIPEDIA_HEADERS,
                    timeout=10,
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
                entities = data.get("search", [])
                if entities:
                    entity_id = entities[0]["id"]
                    logger.info(f"  - Wikidata entity found: {entity_id} for '{term}'")
                    break
            except Exception as e:
                logger.debug(f"  - Wikidata search failed for '{term}': {e}")

        if not entity_id:
            return result

        # Step 2: Fetch the entity's claims
        try:
            resp = requests.get(
                self._WIKIDATA_API,
                params={
                    "action": "wbgetentities",
                    "ids": entity_id,
                    "props": "claims",
                    "format": "json",
                },
                headers=self._WIKIPEDIA_HEADERS,
                timeout=10,
            )
            if resp.status_code != 200:
                return result
            entity_data = resp.json().get("entities", {}).get(entity_id, {})
            claims = entity_data.get("claims", {})
        except Exception as e:
            logger.debug(f"  - Wikidata claims fetch failed: {e}")
            return result

        def get_label(qid: str) -> str:
            """Resolve a Wikidata QID to its English label."""
            try:
                r = requests.get(
                    self._WIKIDATA_API,
                    params={
                        "action": "wbgetentities",
                        "ids": qid,
                        "props": "labels",
                        "languages": "en",
                        "format": "json",
                    },
                    headers=self._WIKIPEDIA_HEADERS,
                    timeout=5,
                )
                if r.status_code == 200:
                    labels = r.json().get("entities", {}).get(qid, {}).get("labels", {})
                    return labels.get("en", {}).get("value", "")
            except Exception:
                pass
            return ""

        def extract_value(prop: str) -> str | None:
            """Extract the first value of a Wikidata property."""
            if prop not in claims:
                return None
            claim_list = claims[prop]
            if not claim_list:
                return None
            mainsnak = claim_list[0].get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            if datavalue.get("type") == "wikibase-entityid":
                qid = datavalue.get("value", {}).get("id", "")
                return get_label(qid) if qid else None
            elif datavalue.get("type") == "time":
                time_val = datavalue.get("value", {}).get("time", "")
                # Extract year from "+1920-01-01T00:00:00Z"
                if time_val:
                    year_match = re.search(r'(\d{4})', time_val)
                    return year_match.group(1) if year_match else None
            elif datavalue.get("type") == "string":
                return datavalue.get("value")
            return None

        # Extract properties
        result["owner"] = extract_value("P127")       # owned by
        result["founded"] = extract_value("P571")      # inception
        result["founder"] = extract_value("P112")      # founded by
        result["headquarters"] = extract_value("P159")  # HQ location
        result["parent_org"] = extract_value("P749")    # parent org

        found = {k: v for k, v in result.items() if v}
        if found:
            logger.info(f"  - Wikidata extracted: {found}")

        return result

    def resolve_outlet_name(self, url: str, domain: str = "") -> str:
        """
        Resolve the official outlet name using URL heuristics + LLM fallback.

        Strategy:
        1. Extract name from URL (quick heuristic)
        2. If the name looks like a raw domain slug (e.g., "apnews", "foxnews"),
           scrape the about page and use LLM to extract the official name

        Args:
            url: The outlet's URL
            domain: Optional domain for about page scraping

        Returns:
            Best available outlet name
        """
        domain = domain or self._extract_domain(url)
        url_name = self._extract_outlet_name(url)
        domain_base = domain.split(".")[0].lower()

        # If the URL-derived name looks like a clean, recognizable name, use it
        # Short names (acronyms like BBC, CNN, NPR) are already good
        if len(domain_base) <= 4:
            # Already handled as acronym — but still try LLM for official name
            pass
        # Check if name is a concatenated slug (no spaces, looks like domain slug)
        # e.g., "apnews", "foxnews", "nytimes", "washingtonpost"
        # These need LLM resolution to get "The Associated Press", "Fox News", etc.

        # Try to get official name from the about page
        about_text = self._scrape_about_page(domain)
        if about_text:
            try:
                prompt = (
                    f'What is the full, official name of the media organization at {domain}? '
                    f'Based on this about page text, return ONLY the official name '
                    f'(e.g., "The Associated Press" not "apnews", '
                    f'"British Broadcasting Corporation" not "bbc", '
                    f'"Fox News" not "foxnews"). '
                    f'If you cannot determine it, return "{url_name}".\n\n'
                    f'About page text:\n{about_text[:3000]}'
                )
                response = self.name_llm.invoke([
                    {"role": "user", "content": prompt}
                ])
                official_name = response.content.strip().strip('"').strip("'")
                # Sanity check: name should be reasonable length
                if 2 < len(official_name) < 100:
                    logger.info(f"  - Resolved outlet name: '{url_name}' -> '{official_name}'")
                    return official_name
            except Exception as e:
                logger.warning(f"  - Outlet name resolution failed: {e}")

        return url_name

    def _search(self, query: str, max_results: int = 5) -> str:
        """
        Perform a search and return combined snippets.

        Uses the configured search backend (DDG by default, OpenAI for hybrid mode).

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            Combined search snippets
        """
        try:
            import time as _time
            _search_t0 = _time.time()
            # Request more results to account for blacklist filtering
            results = self.search_backend.search(query, max_results=max_results + 5)
            logger.debug(f"  - Search for '{query[:60]}...' returned {len(results)} results in {_time.time()-_search_t0:.1f}s")
            if results:
                snippets = []
                for r in results:
                    url = r.get("url", "")
                    # Filter out blacklisted domains
                    if url:
                        result_domain = self._extract_domain(url)
                        if result_domain in self.SEARCH_BLACKLIST:
                            continue
                    title = r.get("title", "")
                    body = r.get("body", "")
                    if title or body:
                        snippets.append(f"{title}: {body} (URL: {url})")
                    if len(snippets) >= max_results:
                        break
                if snippets:
                    logger.debug(f"  - Kept {len(snippets)} results after filtering")
                else:
                    logger.debug(f"  - All {len(results)} results filtered out or empty")
                return "\n\n".join(snippets)
            else:
                logger.debug(f"  - Search returned no results for: {query[:80]}")
            return ""
        except Exception as e:
            logger.warning(f"Search failed for '{query[:60]}...': {e}")
            return ""

    def research_history(self, outlet_name: str, domain: str = "") -> HistoryLLMOutput:
        """
        Research outlet history and founding information.

        Strategy: Gather evidence from MULTIPLE sources and combine them,
        rather than stopping at the first hit. This produces richer context
        for the LLM extraction step.

        Sources (all attempted, results combined):
        1. Outlet's own about page (direct scrape)
        2. Wikipedia API (direct, no DDG needed)
        3. DDG search with quoted name
        4. DDG search with relaxed (unquoted) name + domain
        5. DDG search with domain only

        Args:
            outlet_name: Human-readable outlet name
            domain: Optional domain for disambiguation (e.g., "bbc.com")

        Returns:
            HistoryLLMOutput with extracted history
        """
        evidence_parts = []

        # Source 0: Wikidata structured facts (highest quality for founding info)
        wikidata = self._fetch_wikidata(outlet_name, domain)
        wikidata_facts = []
        if wikidata.get("founded"):
            wikidata_facts.append(f"Founded: {wikidata['founded']}")
        if wikidata.get("founder"):
            wikidata_facts.append(f"Founder: {wikidata['founder']}")
        if wikidata.get("headquarters"):
            wikidata_facts.append(f"Headquarters: {wikidata['headquarters']}")
        if wikidata.get("owner"):
            wikidata_facts.append(f"Owner: {wikidata['owner']}")
        if wikidata.get("parent_org"):
            wikidata_facts.append(f"Parent Organization: {wikidata['parent_org']}")
        if wikidata_facts:
            evidence_parts.append(f"=== WIKIDATA (STRUCTURED) ===\n" + "\n".join(wikidata_facts))

        # Source 1: Wikipedia API (reliable, fast, no DDG needed) — PRIMARY SOURCE
        wiki_text = self._fetch_wikipedia(outlet_name)
        if not wiki_text and domain:
            wiki_text = self._fetch_wikipedia(domain)
        if wiki_text:
            evidence_parts.append(f"=== WIKIPEDIA ===\n{wiki_text}")

        # Source 2: Scrape the outlet's own about page directly
        if domain:
            about_text = self._scrape_about_page(domain)
            if about_text:
                evidence_parts.append(f"=== ABOUT PAGE ({domain}) ===\n{about_text}")

        # Source 3: DDG search with quoted name
        name_variants = f'"{outlet_name}"'
        if domain:
            domain_base = domain.split(".")[0]
            if domain_base.lower() != outlet_name.lower().replace(" ", ""):
                name_variants = f'"{outlet_name}" OR "{domain_base}"'
        search_text = self._search(f'{name_variants} founded history about')
        if search_text:
            evidence_parts.append(f"=== SEARCH RESULTS ===\n{search_text}")

        # Source 4: Relaxed (unquoted) search if quoted search failed
        if not search_text:
            relaxed_query = f'{outlet_name} {domain} founded history about us'
            search_text = self._search(relaxed_query)
            if search_text:
                evidence_parts.append(f"=== SEARCH RESULTS (relaxed) ===\n{search_text}")

        # Source 5: Domain-only fallback
        if not search_text and domain:
            domain_search = self._search(f'{domain} history founded owner media')
            if domain_search:
                evidence_parts.append(f"=== SEARCH RESULTS (domain) ===\n{domain_search}")

        if not evidence_parts:
            logger.info(f"  - No evidence found for {outlet_name} history, returning empty")
            return HistoryLLMOutput(
                summary="No history information found via web search.",
                confidence=0.0,
            )

        combined = "\n\n".join(evidence_parts)
        user_prompt = f"""Extract history information for "{outlet_name}" from these sources:

{combined[:4000]}"""

        try:
            result: HistoryLLMOutput = self.history_llm.invoke(
                [
                    {"role": "system", "content": self.HISTORY_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return result
        except Exception as e:
            logger.error(f"History research failed: {e}")
            return HistoryLLMOutput(
                summary=f"History research failed: {str(e)}",
                confidence=0.0
            )

    def research_ownership(self, outlet_name: str, domain: str = "") -> OwnershipLLMOutput:
        """
        Research ownership and funding information.

        Strategy: Gather evidence from MULTIPLE sources and combine them.
        Wikipedia is especially valuable for ownership info (parent company,
        headquarters, funding model).

        Sources (all attempted, results combined):
        1. Wikipedia API (often has owner/parent/HQ info)
        2. DDG search with quoted name + ownership keywords
        3. DDG search with relaxed (unquoted) name + domain
        4. DDG search with domain only

        Args:
            outlet_name: Human-readable outlet name
            domain: Optional domain for disambiguation

        Returns:
            OwnershipLLMOutput with extracted ownership info
        """
        evidence_parts = []

        # Source 0: Wikidata structured facts (best quality for ownership)
        wikidata = self._fetch_wikidata(outlet_name, domain)
        wikidata_facts = []
        if wikidata.get("owner"):
            wikidata_facts.append(f"Owner: {wikidata['owner']}")
        if wikidata.get("parent_org"):
            wikidata_facts.append(f"Parent Organization: {wikidata['parent_org']}")
        if wikidata.get("headquarters"):
            wikidata_facts.append(f"Headquarters: {wikidata['headquarters']}")
        if wikidata.get("founded"):
            wikidata_facts.append(f"Founded: {wikidata['founded']}")
        if wikidata.get("founder"):
            wikidata_facts.append(f"Founder: {wikidata['founder']}")
        if wikidata_facts:
            evidence_parts.append(f"=== WIKIDATA (STRUCTURED) ===\n" + "\n".join(wikidata_facts))

        # Source 1: Wikipedia API (often has ownership/parent/HQ info)
        wiki_text = self._fetch_wikipedia(outlet_name)
        if not wiki_text and domain:
            wiki_text = self._fetch_wikipedia(domain)
        if not wiki_text and domain:
            # Try brand name extracted from domain (e.g., "frontiersin" → "Frontiers")
            brand = domain.split(".")[0].replace("-", " ").title()
            wiki_text = self._fetch_wikipedia(brand)
        if wiki_text:
            evidence_parts.append(f"=== WIKIPEDIA ===\n{wiki_text}")

        # Source 1b: Scrape about page directly (often has ownership/funding info)
        if domain:
            about_text = self._scrape_about_page(domain)
            if about_text:
                evidence_parts.append(f"=== ABOUT PAGE ===\n{about_text[:2000]}")

        # Source 2: DDG search with quoted name
        search_text = self._search(
            f'"{outlet_name}" ownership owner parent company funded by headquarters'
        )
        if search_text:
            evidence_parts.append(f"=== SEARCH RESULTS ===\n{search_text}")

        # Source 3: Relaxed (unquoted) search if quoted search failed
        if not search_text:
            relaxed = f'{outlet_name} {domain} owner parent company headquarters'
            search_text = self._search(relaxed)
            if search_text:
                evidence_parts.append(f"=== SEARCH RESULTS (relaxed) ===\n{search_text}")

        # Source 4: Domain-only fallback
        if not search_text and domain:
            domain_search = self._search(f'{domain} ownership owner funded headquarters')
            if domain_search:
                evidence_parts.append(f"=== SEARCH RESULTS (domain) ===\n{domain_search}")

        # Source 5: Newsletter-specific search (for Substack/independent outlets)
        if not search_text or len(evidence_parts) <= 2:
            newsletter_search = self._search(
                f'"{outlet_name}" author journalist founded subscription Substack Patreon'
            )
            if newsletter_search:
                evidence_parts.append(f"=== SEARCH RESULTS (newsletter) ===\n{newsletter_search}")

        if not evidence_parts:
            logger.info(f"  - No evidence found for {outlet_name} ownership, returning empty")
            return OwnershipLLMOutput(
                notes="No ownership information found via web search.",
                confidence=0.0,
            )

        combined = "\n\n".join(evidence_parts)
        user_prompt = f"""Extract ownership and funding information for "{outlet_name}" from these sources:

{combined[:6000]}"""

        try:
            result: OwnershipLLMOutput = self.ownership_llm.invoke(
                [
                    {"role": "system", "content": self.OWNERSHIP_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return result
        except Exception as e:
            logger.error(f"Ownership research failed: {e}")
            return OwnershipLLMOutput(
                notes=f"Ownership research failed: {str(e)}",
                confidence=0.0
            )

    def research_external_analysis(self, outlet_name: str, domain: str = "") -> ExternalAnalysisLLMOutput:
        """
        Research external analyses and criticism.

        Args:
            outlet_name: Human-readable outlet name
            domain: Optional domain for disambiguation

        Returns:
            ExternalAnalysisLLMOutput with external analyses
        """
        # Narrowed query: journalism reviews, academic critiques, ownership
        # controversies, corrections policy — avoid pulling in MBFC/AllSides ratings
        query = f'"{outlet_name}" journalism review criticism controversy corrections policy accuracy'
        snippets = self._search(query, max_results=8)

        # Fallback: include domain
        if not snippets and domain:
            query = f'{domain} journalism review criticism controversy corrections policy'
            snippets = self._search(query, max_results=8)

        if not snippets:
            # No search results — return low-confidence empty result.
            # Do NOT fall back to LLM general knowledge (anti-contamination).
            logger.info(f"  - No search results found for {outlet_name} external analysis, returning empty")
            return ExternalAnalysisLLMOutput(
                analyses=[],
                confidence=0.0,
            )
        else:
            user_prompt = f"""Extract external analyses and criticism for "{outlet_name}" from these search results:

SEARCH RESULTS:
{snippets[:4000]}"""

        try:
            result: ExternalAnalysisLLMOutput = self.analysis_llm.invoke(
                [
                    {"role": "system", "content": self.EXTERNAL_ANALYSIS_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return result
        except Exception as e:
            logger.error(f"External analysis research failed: {e}")
            return ExternalAnalysisLLMOutput(
                analyses=[],
                confidence=0.0
            )

    def research_all(
        self,
        outlet_name: str,
        domain: str = "",
    ) -> tuple["HistoryLLMOutput", "OwnershipLLMOutput", "ExternalAnalysisLLMOutput"]:
        """Batch history, ownership, and external analysis into one search call.

        If the search backend supports batch_search(), all three research
        queries are combined into a single API call, reducing latency and
        cost when using OpenAI web_search.  Falls back to sequential calls
        for backends without batch_search() (e.g., DDG).

        Returns:
            Tuple of (history, ownership, external_analysis) LLM outputs.
        """
        has_batch = hasattr(self.search_backend, "batch_search")

        if not has_batch:
            # Fall back to sequential individual calls
            history = self.research_history(outlet_name, domain)
            ownership = self.research_ownership(outlet_name, domain)
            external = self.research_external_analysis(outlet_name, domain)
            return history, ownership, external

        # Build queries for all three research dimensions
        name_variants = f'"{outlet_name}"'
        if domain:
            domain_base = domain.split(".")[0]
            if domain_base.lower() != outlet_name.lower():
                name_variants = f'"{outlet_name}" OR "{domain_base}"'

        history_query = f'{name_variants} founded history about'
        ownership_query = f'"{outlet_name}" ownership owner parent company funded by headquarters'
        external_query = f'"{outlet_name}" journalism review criticism controversy corrections policy accuracy'

        queries = [history_query, ownership_query, external_query]

        logger.info(f"  - Batch searching history+ownership+external for {outlet_name}...")
        batch_results = self.search_backend.batch_search(queries, max_results_per_query=8)

        # Fetch Wikipedia once — used for both history and ownership
        wiki_text = self._fetch_wikipedia(outlet_name)
        if not wiki_text and domain:
            wiki_text = self._fetch_wikipedia(domain)

        # Process history results — combine about page + wiki + search
        history_parts = []
        if domain:
            about_text = self._scrape_about_page(domain)
            if about_text:
                history_parts.append(f"=== ABOUT PAGE ({domain}) ===\n{about_text}")
        if wiki_text:
            history_parts.append(f"=== WIKIPEDIA ===\n{wiki_text}")
        history_search = self._format_snippets(
            batch_results.get(history_query, []), max_results=5
        )
        if history_search:
            history_parts.append(f"=== SEARCH RESULTS ===\n{history_search}")

        if not history_parts:
            history_user_prompt = (
                f'Extract history information for "{outlet_name}" (domain: {domain}).\n\n'
                f"No search results were available. Return minimal data with confidence 0.0. "
                f"Do NOT use general training knowledge about this outlet."
            )
        else:
            combined_history = "\n\n".join(history_parts)
            history_user_prompt = (
                f'Extract history information for "{outlet_name}" from these sources:\n\n'
                f"{combined_history[:4000]}"
            )

        # Process ownership results — combine about page + wiki + search
        ownership_parts = []
        if about_text:
            ownership_parts.append(f"=== ABOUT PAGE ({domain}) ===\n{about_text[:2000]}")
        if wiki_text:
            ownership_parts.append(f"=== WIKIPEDIA ===\n{wiki_text}")
        ownership_search = self._format_snippets(
            batch_results.get(ownership_query, []), max_results=5
        )
        if ownership_search:
            ownership_parts.append(f"=== SEARCH RESULTS ===\n{ownership_search}")

        if not ownership_parts:
            ownership_user_prompt = (
                f'Extract ownership and funding information for "{outlet_name}" (domain: {domain}).\n\n'
                f"No search results were available. Return minimal data with confidence 0.0. "
                f"Do NOT use general training knowledge about this outlet."
            )
        else:
            combined_ownership = "\n\n".join(ownership_parts)
            ownership_user_prompt = (
                f'Extract ownership and funding information for "{outlet_name}" from these sources:\n\n'
                f"{combined_ownership[:4000]}"
            )

        # Process external analysis results
        external_snippets = self._format_snippets(
            batch_results.get(external_query, []), max_results=8
        )
        if not external_snippets:
            external_user_prompt = (
                f'Extract external analyses and criticism for "{outlet_name}" (domain: {domain}).\n\n'
                f"No search results were available. Return empty analyses with confidence 0.0. "
                f"Do NOT use general training knowledge about this outlet."
            )
        else:
            external_user_prompt = (
                f'Extract external analyses and criticism for "{outlet_name}" from these search results:\n\n'
                f"SEARCH RESULTS:\n{external_snippets[:4000]}"
            )

        # Run all three LLM parsing calls in parallel
        from concurrent.futures import ThreadPoolExecutor

        def _invoke_history():
            try:
                return self.history_llm.invoke([
                    {"role": "system", "content": self.HISTORY_PROMPT},
                    {"role": "user", "content": history_user_prompt},
                ])
            except Exception as e:
                logger.error(f"History research failed: {e}")
                return HistoryLLMOutput(summary=f"History research failed: {e}", confidence=0.0)

        def _invoke_ownership():
            try:
                return self.ownership_llm.invoke([
                    {"role": "system", "content": self.OWNERSHIP_PROMPT},
                    {"role": "user", "content": ownership_user_prompt},
                ])
            except Exception as e:
                logger.error(f"Ownership research failed: {e}")
                return OwnershipLLMOutput(notes=f"Ownership research failed: {e}", confidence=0.0)

        def _invoke_external():
            try:
                return self.analysis_llm.invoke([
                    {"role": "system", "content": self.EXTERNAL_ANALYSIS_PROMPT},
                    {"role": "user", "content": external_user_prompt},
                ])
            except Exception as e:
                logger.error(f"External analysis research failed: {e}")
                return ExternalAnalysisLLMOutput(analyses=[], confidence=0.0)

        with ThreadPoolExecutor(max_workers=3) as executor:
            history_future = executor.submit(_invoke_history)
            ownership_future = executor.submit(_invoke_ownership)
            external_future = executor.submit(_invoke_external)

            history = history_future.result()
            ownership = ownership_future.result()
            external = external_future.result()

        return history, ownership, external

    def _format_snippets(self, results: list[dict], max_results: int = 5) -> str:
        """Format search result dicts into snippet text, applying blacklist filter."""
        snippets = []
        for r in results:
            url = r.get("url", "")
            if url:
                result_domain = self._extract_domain(url)
                if result_domain in self.SEARCH_BLACKLIST:
                    continue
            title = r.get("title", "")
            body = r.get("body", "")
            if title or body:
                snippets.append(f"{title}: {body} (URL: {url})")
            if len(snippets) >= max_results:
                break
        return "\n\n".join(snippets)


# =============================================================================
# MediaProfiler - Comprehensive Analysis Orchestrator
# =============================================================================


class MediaProfiler:
    """
    Orchestrates all analyzers to produce comprehensive MBFC-style reports.

    Combines results from:
    - TrafficLongevityAnalyzer: Domain age and traffic tier
    - MediaTypeAnalyzer: Media type classification
    - OpinionAnalyzer: News vs Opinion classification
    - EditorialBiasAnalyzer: Political bias detection
    - FactCheckSearcher: Fact-checker search results
    - SourcingAnalyzer: Source quality assessment
    - PseudoscienceAnalyzer: Pseudoscience detection
    - MediaResearcher: History, ownership, external analysis

    Produces a ComprehensiveReportData object with all analysis results.
    """

    def __init__(
        self,
        model: str = "gpt-5-mini-2025-08-07",
        temperature: float = 0.0,
        search_backend=None,
        use_calibration: bool = True,
    ):
        """
        Initialize all analyzers.

        Args:
            model: OpenAI model to use for all analyzers
            temperature: LLM temperature (0 for deterministic)
            search_backend: Optional SearchBackend instance for web searches
                           (defaults to DDGSearchBackend). Passed to analyzers
                           that perform web searches: TrafficLongevityAnalyzer,
                           MediaTypeAnalyzer, FactCheckSearcher, MediaResearcher.
            use_calibration: Whether to apply LLM-based score calibration after
                           formula computation (default True).
        """
        self._model = model
        self._temperature = temperature
        self._use_calibration = use_calibration
        self.traffic_analyzer = TrafficLongevityAnalyzer(
            model=model, temperature=temperature, search_backend=search_backend
        )
        self.media_type_analyzer = MediaTypeAnalyzer(
            model=model, temperature=temperature, search_backend=search_backend
        )
        self.opinion_analyzer = OpinionAnalyzer(model=model, temperature=temperature)
        self.editorial_bias_analyzer = EditorialBiasAnalyzer(model=model, temperature=temperature)
        self.fact_check_searcher = FactCheckSearcher(
            model=model, temperature=temperature, search_backend=search_backend,
            attribute_claim_source=True,
        )
        self.sourcing_analyzer = SourcingAnalyzer(model=model, temperature=temperature)
        self.pseudoscience_analyzer = PseudoscienceAnalyzer(model=model, temperature=temperature)
        self.transparency_analyzer = TransparencyAnalyzer(
            model=model, temperature=temperature, search_backend=search_backend
        )
        self.one_sidedness_analyzer = OneSidednessAnalyzer(model=model, temperature=temperature)
        self.researcher = MediaResearcher(
            model=model, temperature=temperature, search_backend=search_backend
        )

    def _lookup_freedom_rating(self, headquarters: str | None) -> dict:
        """Look up combined press freedom score (RSF + Freedom House).

        Uses RSF Press Freedom Index as primary source, and Freedom House
        as secondary. When both are available, the combined score is their
        average. When only one is available, that one is used directly.

        Args:
            headquarters: Headquarters string (e.g., "London, United Kingdom")

        Returns:
            Dict with keys: country, freedom_score, freedom_label, freedom_rank (or all None)
        """
        from config import FREEDOM_INDEX_FILE, FREEDOM_HOUSE_FILE, FREEDOM_LABELS, COUNTRY_NAME_ALIASES

        result = {"country": None, "freedom_score": None, "freedom_label": None, "freedom_rank": None}

        if not headquarters:
            return result

        # Extract country name from headquarters (usually "City, Country")
        parts = [p.strip() for p in headquarters.split(",")]
        country_name = parts[-1] if parts else None
        if not country_name:
            return result

        # Normalize via aliases
        country_name = COUNTRY_NAME_ALIASES.get(country_name, country_name)
        result["country"] = country_name

        rsf_score = None
        rsf_rank = None
        fh_score = None

        # --- RSF Press Freedom Index ---
        csv_path = os.path.join(os.path.dirname(__file__), FREEDOM_INDEX_FILE)
        if os.path.exists(csv_path):
            try:
                with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f, delimiter=";")
                    for row in reader:
                        row_country = row.get("Country_EN", "").strip()
                        if row_country.lower() == country_name.lower():
                            score_str = row.get("Score 2025", "").replace(",", ".")
                            rank_str = row.get("Rank", "")
                            try:
                                rsf_score = float(score_str)
                                rsf_rank = int(rank_str) if rank_str else None
                            except (ValueError, TypeError):
                                pass
                            break
            except Exception as e:
                logger.error(f"Failed to load RSF freedom index: {e}")
        else:
            logger.warning(f"RSF freedom index file not found: {csv_path}")

        # --- Freedom House (optional secondary source) ---
        # FH_FIW.csv uses World Bank Data360 format: filter INDICATOR=FH_FIW_TOTAL,
        # use most recent TIME_PERIOD, match via REF_AREA_LABEL, score from OBS_VALUE
        fh_path = os.path.join(os.path.dirname(__file__), FREEDOM_HOUSE_FILE)
        if os.path.exists(fh_path):
            try:
                with open(fh_path, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    best_year = 0
                    for row in reader:
                        indicator = row.get("INDICATOR", "").strip('"')
                        if indicator != "FH_FIW_TOTAL":
                            continue
                        row_country = row.get("REF_AREA_LABEL", "").strip('"').strip()
                        # Normalize FH country names (e.g., "Gambia, The" → "Gambia")
                        fh_country = COUNTRY_NAME_ALIASES.get(row_country, row_country)
                        if fh_country.lower() != country_name.lower():
                            continue
                        # Use the most recent year available
                        try:
                            year = int(row.get("TIME_PERIOD", "0").strip('"'))
                        except (ValueError, TypeError):
                            continue
                        if year > best_year:
                            best_year = year
                            try:
                                fh_score = float(row.get("OBS_VALUE", "").strip('"'))
                            except (ValueError, TypeError):
                                pass
            except Exception as e:
                logger.debug(f"Freedom House data not available: {e}")

        # --- Combine RSF + Freedom House ---
        if rsf_score is not None and fh_score is not None:
            combined_score = (rsf_score + fh_score) / 2.0
        elif rsf_score is not None:
            combined_score = rsf_score
        elif fh_score is not None:
            combined_score = fh_score
        else:
            return result

        result["freedom_score"] = round(combined_score, 2)
        result["freedom_rank"] = rsf_rank

        # Determine label from combined score
        for low, high, label in FREEDOM_LABELS:
            if low <= combined_score <= high:
                result["freedom_label"] = label
                break
        else:
            result["freedom_label"] = "Unknown"

        return result

    @staticmethod
    def _sanitize_null(value: str | None) -> str | None:
        """Convert LLM-generated 'null' strings to actual None."""
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in ("null", "none", "n/a", ""):
            return None
        return value

    def _extract_domain(self, url: str) -> str:
        """Extract the root domain from a URL."""
        parsed = urlparse(url if url.startswith("http") else f"https://{url}")
        domain = parsed.netloc or parsed.path
        domain = re.sub(r"^www\.", "", domain)
        domain = domain.split("/")[0]
        return domain.lower()

    @staticmethod
    def _bias_score_to_label(score: float) -> str:
        """Convert numeric bias score to MBFC-style label (mirrors EditorialBiasAnalyzer._score_to_label)."""
        if score <= -8.0:
            return "Extreme Left"
        elif score <= -7.0:
            return "Far Left"
        elif score <= -5.0:
            return "Left"
        elif score <= -2.0:
            return "Left-Center"
        elif score <= 1.9:
            return "Least Biased"
        elif score <= 4.9:
            return "Right-Center"
        elif score <= 6.9:
            return "Right"
        elif score <= 7.9:
            return "Far Right"
        else:
            return "Extreme Right"

    @staticmethod
    def _estimate_bias_from_research(
        external_analyses, history, ownership,
    ) -> float:
        """Estimate bias from research signals when no editorial analysis is available.

        Scans external analyses, history summary, and ownership notes for
        ideological keywords. Returns a rough bias score instead of 0.0.
        """
        # Collect all text from research outputs
        texts = []
        if external_analyses and getattr(external_analyses, 'analyses', None):
            for item in external_analyses.analyses:
                if getattr(item, 'summary', None):
                    texts.append(item.summary.lower())
        if history and hasattr(history, 'summary') and history.summary:
            texts.append(history.summary.lower())
        if ownership and hasattr(ownership, 'notes') and ownership.notes:
            texts.append(ownership.notes.lower())

        combined = " ".join(texts)
        if not combined.strip():
            return 0.0  # genuinely unknown

        # Keyword-based signal detection
        far_right_signals = [
            "conspiracy", "far-right", "far right", "extreme right",
            "white nationalist", "white supremac", "alt-right",
            "qanon", "infowars", "propaganda", "disinformation",
            "state-controlled", "state controlled", "kremlin",
            "anti-vaxx", "anti-vaccine", "pseudoscience",
        ]
        right_signals = [
            "conservative", "right-wing", "right wing", "right-leaning",
            "republican", "libertarian", "pro-trump",
        ]
        far_left_signals = [
            "far-left", "far left", "extreme left", "marxist",
            "communist", "revolutionary", "anarchist",
        ]
        left_signals = [
            "progressive", "left-wing", "left wing", "left-leaning",
            "liberal", "democrat", "socialist",
        ]

        score = 0.0
        for kw in far_right_signals:
            if kw in combined:
                score += 3.0
        for kw in right_signals:
            if kw in combined:
                score += 1.5
        for kw in far_left_signals:
            if kw in combined:
                score -= 3.0
        for kw in left_signals:
            if kw in combined:
                score -= 1.5

        # Clamp to [-10, +10]
        return max(-10.0, min(10.0, score))

    @staticmethod
    def _estimate_factuality_from_research(
        external_analyses, history, ownership,
    ) -> float | None:
        """Estimate factuality adjustment from research signals.

        Returns a signal score (0=trustworthy, 10=unreliable) or None if
        no clear signals found.
        """
        texts = []
        if external_analyses and getattr(external_analyses, 'analyses', None):
            for item in external_analyses.analyses:
                if getattr(item, 'summary', None):
                    texts.append(item.summary.lower())
        if history and hasattr(history, 'summary') and history.summary:
            texts.append(history.summary.lower())

        combined = " ".join(texts)
        if not combined.strip():
            return None

        # Credible signals → lower factuality score (better)
        credible_signals = [
            "award-winning", "pulitzer", "peabody", "respected",
            "authoritative", "reliable", "credible", "established",
            "wire service", "news agency", "peer-reviewed",
            "academic", "research journal", "scientific journal",
        ]
        # Unreliable signals → higher factuality score (worse)
        unreliable_signals = [
            "misinformation", "disinformation", "conspiracy",
            "pseudoscience", "unreliable", "fake news", "propaganda",
            "misleading", "debunked", "discredited", "fringe",
            "junk science", "anti-science", "hoax",
        ]

        credible_count = sum(1 for kw in credible_signals if kw in combined)
        unreliable_count = sum(1 for kw in unreliable_signals if kw in combined)

        if credible_count == 0 and unreliable_count == 0:
            return None

        # Map to 0-10 scale: credible→0, unreliable→10
        if unreliable_count > 0 and credible_count == 0:
            return min(8.0 + unreliable_count, 10.0)
        elif credible_count > 0 and unreliable_count == 0:
            return max(1.0 - credible_count * 0.3, 0.0)
        else:
            # Mixed signals — lean slightly unreliable
            return 5.0

    def _score_to_factuality_label(self, score: float) -> str:
        """Convert factuality score to MBFC-style label.

        Per MBFC methodology: 0.0 only → Very High; 0.1–1.9 → High.
        """
        if score == 0.0:
            return "Very High"
        elif score <= 1.9:
            return "High"
        elif score <= 4.4:
            return "Mostly Factual"
        elif score <= 6.4:
            return "Mixed"
        elif score <= 8.4:
            return "Low"
        else:
            return "Very Low"

    def _calculate_credibility_score(
        self,
        factuality_label: str,
        factuality_score: float,
        bias_label: str,
        traffic_tier: str,
        domain_age_years: float | None = None,
        freedom_label: str | None = None,
        promotes_pseudoscience: bool = False,
        is_questionable: bool = False,
    ) -> tuple[float, str]:
        """Calculate credibility using MBFC 2025 point-based system.

        Points:
          Factual: Very High(4), High(3), Mostly Factual(2), Mixed(1), Low/Very Low(0)
          Bias: Least Biased(3), L-C/R-C(2), L/R(1), Extreme/Questionable(0)
          Traffic: High(2), Medium(1), Low/Minimal(0)
          Longevity: +1 if source exists 10+ years
          Freedom: Limited Freedom(-1), Total Oppression(-2)

        Levels: 6+ = HIGH, 3-5 = MEDIUM, 0-2 = LOW

        Exceptions (per MBFC methodology):
          - Questionable/Conspiracy/Pseudoscience → automatic LOW
          - Mostly Factual with score 3.6–4.5 → automatic MEDIUM
        """
        from config import CREDIBILITY_POINTS

        # Factual points
        factual_points = CREDIBILITY_POINTS["factual"].get(factuality_label, 0)

        # Bias points — map extreme labels
        bias_key = bias_label
        if bias_key in ("Extreme Left", "Extreme Right"):
            bias_key = "Extreme"
        if bias_key in ("Far Left", "Far Right"):
            # Far Left/Right are sub-labels of Left/Right — same credibility points
            bias_key = bias_key.replace("Far ", "")
        bias_points = CREDIBILITY_POINTS["bias"].get(bias_key, 0)

        # Traffic points — handle enum-style values like "HIGH" -> "High"
        traffic_map = {"HIGH": "High", "MEDIUM": "Medium", "LOW": "Minimal", "MINIMAL": "Minimal", "UNKNOWN": "Minimal"}
        traffic_key = traffic_map.get(traffic_tier.upper(), traffic_tier.capitalize()) if traffic_tier else "Minimal"
        traffic_points = CREDIBILITY_POINTS["traffic"].get(traffic_key, 0)

        # Longevity bonus
        longevity_bonus = 1 if domain_age_years and domain_age_years >= 10 else 0

        # Press freedom penalty
        freedom_penalty = 0
        if freedom_label:
            freedom_penalty = CREDIBILITY_POINTS["freedom_penalty"].get(freedom_label, 0)

        total_points = factual_points + bias_points + traffic_points + longevity_bonus + freedom_penalty
        total_points = max(0, total_points)  # Floor at 0

        logger.debug(
            f"  - Credibility: factual={factual_points}({factuality_label}) "
            f"bias={bias_points}({bias_label}) traffic={traffic_points}({traffic_tier}) "
            f"longevity={longevity_bonus} freedom={freedom_penalty} = {total_points}"
        )

        # Exception rules (override point-based) — per MBFC methodology
        # 1. Pseudoscience/Conspiracy → automatic LOW
        if promotes_pseudoscience:
            return float(total_points), "Low Credibility"

        # 2. Questionable sources → automatic LOW
        if is_questionable:
            return float(total_points), "Low Credibility"

        # 3. Mostly Factual with score 3.6–4.5 → automatic MEDIUM
        #    (not all Mostly Factual — only the lower end per methodology)
        if factuality_label == "Mostly Factual" and 3.6 <= factuality_score <= 4.5:
            return float(total_points), "Medium Credibility"

        # Point-based levels
        if total_points >= 6:
            label = "High Credibility"
        elif total_points >= 3:
            label = "Medium Credibility"
        else:
            label = "Low Credibility"

        return float(total_points), label

    def profile(
        self,
        url: str,
        articles: list[dict[str, str]],
        outlet_name: Optional[str] = None,
        parallel: bool = True,
    ) -> ComprehensiveReportData:
        """
        Perform comprehensive profiling of a media outlet.

        Args:
            url: The outlet's URL
            articles: List of article dicts with 'title' and 'text' keys
            outlet_name: Optional human-readable name (auto-detected if not provided)
            parallel: If True, run independent analyzers concurrently (default).
                      Set to False for sequential execution (useful for debugging).

        Returns:
            ComprehensiveReportData with all analysis results
        """
        if parallel:
            return self._profile_parallel(url, articles, outlet_name)
        return self._profile_sequential(url, articles, outlet_name)

    def _profile_sequential(
        self,
        url: str,
        articles: list[dict[str, str]],
        outlet_name: Optional[str] = None,
    ) -> ComprehensiveReportData:
        """Original sequential profiling implementation (for debugging)."""
        domain = self._extract_domain(url)
        if not outlet_name:
            logger.info("  - Resolving outlet name...")
            outlet_name = self.researcher.resolve_outlet_name(url, domain=domain)

        logger.info(f"Profiling (sequential): {outlet_name} ({domain})")
        import time as _time
        _profile_t0 = _time.time()

        # 1. Traffic and metadata analysis
        logger.info("  - Analyzing traffic and longevity...")
        traffic_data = self.traffic_analyzer.analyze(url)

        logger.info("  - Classifying media type...")
        media_type_result = self.media_type_analyzer.analyze(url)

        # 2. Evidence sufficiency gate (MBFC requires 10 headlines, 5 full stories)
        from config import MIN_HEADLINES, MIN_FULL_STORIES
        headline_count = len(articles)
        full_story_count = sum(1 for a in articles if len(a.get("text", "")) >= 200)
        evidence_sufficient = (headline_count >= MIN_HEADLINES and full_story_count >= MIN_FULL_STORIES)
        insufficient_reason = None
        if not evidence_sufficient:
            parts = []
            if headline_count < MIN_HEADLINES:
                parts.append(f"only {headline_count}/{MIN_HEADLINES} headlines")
            if full_story_count < MIN_FULL_STORIES:
                parts.append(f"only {full_story_count}/{MIN_FULL_STORIES} full stories")
            insufficient_reason = "Insufficient evidence: " + ", ".join(parts)
            logger.warning(f"  - {insufficient_reason} — scores may be unreliable")

        news_articles = []
        opinion_articles = []
        editorial_bias_result: Optional[EditorialBiasResult] = None
        sourcing_result: Optional[SourcingAnalysisResult] = None
        pseudoscience_result: Optional[PseudoscienceAnalysisResult] = None
        one_sidedness_result: Optional[OneSidednessResult] = None

        if articles:
            logger.info(f"  - Classifying {len(articles)} articles (news vs opinion)...")
            for article in articles:
                try:
                    classification = self.opinion_analyzer.analyze(
                        article.get("title", ""), article.get("text", "")[:1000]
                    )
                    if classification.article_type.value in ("Opinion", "Satire"):
                        opinion_articles.append(article)
                    else:
                        news_articles.append(article)
                except Exception:
                    news_articles.append(article)

            logger.info(
                f"  - Classified: {len(news_articles)} news, {len(opinion_articles)} opinion"
            )

            logger.info(f"  - Analyzing {len(articles)} articles for bias...")
            # Fetch about page text as fallback for zero-article scenarios
            about_text_for_bias = self.researcher._scrape_about_page(domain) if not articles else None
            editorial_bias_result = self.editorial_bias_analyzer.analyze(
                articles, url, outlet_name,
                news_articles=news_articles,
                opinion_articles=opinion_articles,
                about_page_text=about_text_for_bias,
            )

            logger.info("  - Analyzing sourcing quality...")
            sourcing_result = self.sourcing_analyzer.analyze(articles)

            logger.info("  - Checking for pseudoscience...")
            pseudoscience_result = self.pseudoscience_analyzer.analyze(
                articles, url, outlet_name
            )

            logger.info("  - Evaluating one-sidedness/propaganda...")
            one_sidedness_result = self.one_sidedness_analyzer.analyze(articles)

        logger.info("  - Searching fact-checkers...")
        fact_check_result = self.fact_check_searcher.analyze(url, outlet_name)

        logger.info("  - Evaluating transparency...")
        transparency_result = self.transparency_analyzer.analyze(
            url, outlet_name, articles=articles,
        )

        logger.info("  - Researching history, ownership, and external analyses...")
        history, ownership, external_analyses = self.researcher.research_all(
            outlet_name, domain=domain
        )

        if history.official_name:
            logger.info(f"  - Updating outlet name from '{outlet_name}' to '{history.official_name}'")
            outlet_name = history.official_name

        return self._build_report(
            url, domain, outlet_name, articles,
            traffic_data, media_type_result,
            editorial_bias_result, sourcing_result, pseudoscience_result,
            one_sidedness_result, fact_check_result, transparency_result,
            history, ownership, external_analyses,
            _profile_t0,
        )

    def _profile_parallel(
        self,
        url: str,
        articles: list[dict[str, str]],
        outlet_name: Optional[str] = None,
    ) -> ComprehensiveReportData:
        """Parallel profiling — runs independent analyzers concurrently."""
        from concurrent.futures import ThreadPoolExecutor
        from config import MAX_PARALLEL_LLM_CALLS

        domain = self._extract_domain(url)
        import time as _time
        _profile_t0 = _time.time()

        # If no outlet_name provided, we need to resolve it first (fast, ~1 LLM call)
        if not outlet_name:
            logger.info("  - Resolving outlet name...")
            outlet_name = self.researcher.resolve_outlet_name(url, domain=domain)

        logger.info(f"Profiling (parallel, max_workers={MAX_PARALLEL_LLM_CALLS}): {outlet_name} ({domain})")

        # Phase 1: Submit all independent tasks
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_LLM_CALLS) as executor:
            # URL-based analyzers (no article dependency)
            logger.info("  - [Phase 1] Launching parallel analyzers...")
            fut_traffic = executor.submit(self.traffic_analyzer.analyze, url)
            fut_media_type = executor.submit(self.media_type_analyzer.analyze, url)
            fut_fact_check = executor.submit(self.fact_check_searcher.analyze, url, outlet_name)
            fut_transparency = executor.submit(
                self.transparency_analyzer.analyze, url, outlet_name, articles,
            )
            fut_research = executor.submit(
                self.researcher.research_all, outlet_name, domain,
            )

            # Article-based analyzers (need articles, but NOT opinion classification)
            fut_opinion = None
            fut_sourcing = None
            fut_pseudoscience = None
            fut_one_sidedness = None

            if articles:
                fut_opinion = executor.submit(
                    self.opinion_analyzer.analyze_batch, articles,
                )
                fut_sourcing = executor.submit(
                    self.sourcing_analyzer.analyze, articles,
                )
                fut_pseudoscience = executor.submit(
                    self.pseudoscience_analyzer.analyze, articles, url, outlet_name,
                )
                fut_one_sidedness = executor.submit(
                    self.one_sidedness_analyzer.analyze, articles,
                )

            # Collect Phase 1 results
            traffic_data = fut_traffic.result()
            logger.info("  - [Done] Traffic analysis")
            media_type_result = fut_media_type.result()
            logger.info("  - [Done] Media type classification")
            fact_check_result = fut_fact_check.result()
            logger.info("  - [Done] Fact check search")
            transparency_result = fut_transparency.result()
            logger.info("  - [Done] Transparency analysis")
            history, ownership, external_analyses = fut_research.result()
            logger.info("  - [Done] History/ownership/external research")

            sourcing_result = None
            pseudoscience_result = None
            one_sidedness_result = None
            editorial_bias_result = None

            if articles:
                opinion_classifications = fut_opinion.result()
                logger.info("  - [Done] Opinion classification (batched)")
                sourcing_result = fut_sourcing.result()
                logger.info("  - [Done] Sourcing analysis")
                pseudoscience_result = fut_pseudoscience.result()
                logger.info("  - [Done] Pseudoscience check")
                one_sidedness_result = fut_one_sidedness.result()
                logger.info("  - [Done] One-sidedness analysis")

                # Phase 2: Editorial bias (depends on opinion classification)
                news_articles = []
                opinion_articles = []
                for article, classification in zip(articles, opinion_classifications):
                    try:
                        if classification.article_type.value in ("Opinion", "Satire"):
                            opinion_articles.append(article)
                        else:
                            news_articles.append(article)
                    except Exception:
                        news_articles.append(article)

                logger.info(
                    f"  - Classified: {len(news_articles)} news, {len(opinion_articles)} opinion"
                )
                logger.info("  - [Phase 2] Running editorial bias analysis...")
                editorial_bias_result = self.editorial_bias_analyzer.analyze(
                    articles, url, outlet_name,
                    news_articles=news_articles,
                    opinion_articles=opinion_articles,
                )
                logger.info("  - [Done] Editorial bias analysis")
            else:
                # No articles: run bias analysis with about page fallback
                logger.info("  - [Phase 2] No articles — running editorial bias with about page fallback...")
                about_text_for_bias = self.researcher._scrape_about_page(domain)
                editorial_bias_result = self.editorial_bias_analyzer.analyze(
                    [], url, outlet_name,
                    about_page_text=about_text_for_bias,
                )
                logger.info("  - [Done] Editorial bias analysis (about page fallback)")

        # Update outlet_name if LLM found the official name
        if history.official_name:
            logger.info(f"  - Updating outlet name from '{outlet_name}' to '{history.official_name}'")
            outlet_name = history.official_name

        return self._build_report(
            url, domain, outlet_name, articles,
            traffic_data, media_type_result,
            editorial_bias_result, sourcing_result, pseudoscience_result,
            one_sidedness_result, fact_check_result, transparency_result,
            history, ownership, external_analyses,
            _profile_t0,
        )

    def _calibrate_scores_with_llm(
        self,
        outlet_name: str,
        url: str,
        formula_bias: float,
        formula_factuality: float,
        editorial_bias_result,
        fact_check_result,
        sourcing_result,
        transparency_result,
        one_sidedness_result,
        pseudoscience_result,
        external_analyses,
        history,
        ownership,
        evidence_sufficient: bool,
    ) -> CalibratedScores:
        """Calibrate formula-based scores using holistic LLM judgment.

        The formula-driven approach compresses scores toward the center and
        defaults to 0.0 bias when editorial analysis fails. This method
        passes all evidence to an LLM with calibration reference points
        to produce better-calibrated final scores.
        """
        # Build evidence summary
        evidence_parts = [f"Outlet: {outlet_name} ({url})"]

        # Editorial bias analysis
        if editorial_bias_result:
            evidence_parts.append(
                f"EDITORIAL BIAS ANALYSIS (from article content):\n"
                f"  Economic Score: {editorial_bias_result.economic_score:.1f}/±10\n"
                f"  Social Score: {editorial_bias_result.social_score:.1f}/±10\n"
                f"  News Reporting Score: {editorial_bias_result.news_reporting_score:.1f}/±10\n"
                f"  Editorial Bias Score: {editorial_bias_result.editorial_bias_score:.1f}/±10\n"
                f"  Composite Bias: {editorial_bias_result.bias_score:.1f} ({editorial_bias_result.mbfc_label})\n"
                f"  Summary: {editorial_bias_result.reasoning[:300] if editorial_bias_result.reasoning else 'N/A'}"
            )
        else:
            evidence_parts.append(
                "EDITORIAL BIAS ANALYSIS: Not available (no articles could be analyzed). "
                "You MUST estimate bias from the research signals below."
            )

        # Fact-check findings
        if fact_check_result:
            evidence_parts.append(
                f"FACT-CHECK FINDINGS:\n"
                f"  Total checks found: {fact_check_result.total_checks_count}\n"
                f"  Failed checks: {fact_check_result.failed_checks_count}\n"
                f"  Score: {fact_check_result.score:.1f}/10"
            )

        # Sourcing quality
        if sourcing_result:
            evidence_parts.append(
                f"SOURCING QUALITY:\n"
                f"  Score: {sourcing_result.score:.1f}/10 (0=perfect, 10=no sourcing)\n"
                f"  Avg sources/article: {sourcing_result.avg_sources_per_article:.1f}, Has hyperlinks: {sourcing_result.has_hyperlinks}"
            )

        # Transparency
        if transparency_result:
            evidence_parts.append(
                f"TRANSPARENCY:\n"
                f"  Score: {transparency_result.score:.1f}/10 (0=fully transparent, 10=none)\n"
                f"  Summary: {transparency_result.reasoning[:200] if transparency_result.reasoning else 'N/A'}"
            )

        # One-sidedness
        if one_sidedness_result:
            evidence_parts.append(
                f"ONE-SIDEDNESS/PROPAGANDA:\n"
                f"  Score: {one_sidedness_result.score:.1f}/10 (0=balanced, 10=extreme)\n"
                f"  Summary: {one_sidedness_result.reasoning[:200] if one_sidedness_result.reasoning else 'N/A'}"
            )

        # Pseudoscience
        if pseudoscience_result:
            evidence_parts.append(
                f"PSEUDOSCIENCE:\n"
                f"  Promotes pseudoscience: {pseudoscience_result.promotes_pseudoscience}\n"
                f"  Respects scientific consensus: {pseudoscience_result.respects_scientific_consensus}"
            )

        # External research
        if external_analyses and hasattr(external_analyses, 'analyses'):
            analyses_text = []
            for ea in external_analyses.analyses[:5]:
                src = getattr(ea, 'source_name', 'Unknown')
                summary = getattr(ea, 'summary', '')[:200]
                analyses_text.append(f"  - {src}: {summary}")
            if analyses_text:
                evidence_parts.append(
                    f"EXTERNAL RESEARCH:\n" + "\n".join(analyses_text)
                )

        # History and ownership
        if history and history.summary:
            evidence_parts.append(f"HISTORY: {history.summary[:600]}")
        if ownership:
            owner = getattr(ownership, 'owner', None) or getattr(ownership, 'parent_company', None)
            funding = getattr(ownership, 'funding_model', None)
            notes = getattr(ownership, 'notes', None)
            parts = []
            if owner:
                parts.append(f"Owner: {owner}")
            if funding:
                parts.append(f"Funding: {funding}")
            if notes:
                parts.append(f"Notes: {notes[:200]}")
            if parts:
                evidence_parts.append(f"OWNERSHIP: {'; '.join(parts)}")

        # Add conditional escalation warnings
        if pseudoscience_result and getattr(pseudoscience_result, 'promotes_pseudoscience', False):
            evidence_parts.append(
                "⚠ WARNING: This outlet promotes pseudoscience. "
                "Factuality should be 7.0+ minimum."
            )
        if history and history.summary:
            history_lower = history.summary.lower()
            if any(kw in history_lower for kw in ['fake news', 'conspiracy', 'unreliable', 'questionable']):
                evidence_parts.append(
                    "⚠ WARNING: Historical record indicates this is an unreliable/questionable source. "
                    "Factuality should be 6.5+ (Low or Very Low)."
                )
        if (fact_check_result and fact_check_result.failed_checks_count == 0
                and sourcing_result and sourcing_result.score < 3.0):
            evidence_parts.append(
                "✓ NOTE: Clean fact-check record with good sourcing suggests "
                "High or Very High factuality (0.0-1.9)."
            )

        evidence_text = "\n\n".join(evidence_parts)

        system_prompt = (
            "You are calibrating media bias and factuality scores for a media outlet. "
            "A formula computed initial estimates from individual analyzer scores, but "
            "formulas compress results toward the center. Your job is to produce "
            "well-calibrated final scores using the full evidence.\n\n"
            "BIAS SCORING (-10 to +10):\n"
            "  -10.0 to -8.0: Extreme Left\n"
            "  -7.9 to -5.0: Left\n"
            "  -4.9 to -2.0: Left-Center\n"
            "  -1.9 to +1.9: Least Biased\n"
            "  +2.0 to +4.9: Right-Center\n"
            "  +5.0 to +7.9: Right\n"
            "  +8.0 to +10.0: Extreme Right\n\n"
            "FACTUALITY SCORING (0 to 10, LOWER IS BETTER):\n"
            "  0.0 only: Very High (zero failed fact checks, perfect sourcing, full transparency)\n"
            "  0.1-1.9: High\n"
            "  2.0-4.4: Mostly Factual\n"
            "  4.5-6.4: Mixed\n"
            "  6.5-8.4: Low\n"
            "  8.5-10.0: Very Low\n\n"
            "CALIBRATION REFERENCE POINTS (use these to anchor your scoring):\n"
            "- AP News: Bias ≈ 0, Factuality ≈ 0.5 (Very High)\n"
            "- Fox News: Bias ≈ +6, Factuality ≈ 5.5 (Mixed)\n"
            "- The New York Times: Bias ≈ -3.5, Factuality ≈ 1.5 (High)\n"
            "- High Times: Bias ≈ -6.4, Factuality ≈ 4.9 (Mixed)\n"
            "- Breitbart: Bias ≈ +7, Factuality ≈ 6.5 (Low)\n"
            "- Education Next: Bias ≈ +3.8, Factuality ≈ 2.4 (Mostly Factual)\n"
            "- InfoWars: Bias ≈ +9, Factuality ≈ 9.5 (Very Low)\n"
            "- Colorado Newsline: Bias ≈ -5.9, Factuality ≈ 1.5 (High)\n"
            "- Reuters: Bias ≈ -0.5, Factuality ≈ 0.0 (Very High)\n\n"
            "CRITICAL RULES:\n"
            "1. Do NOT default to 0.0 bias when evidence is unclear — use ALL available "
            "signals (ownership, history, external research) to estimate bias direction.\n"
            "2. Use the FULL range of scores. Very High factuality outlets (wire services, "
            "academic journals) should score near 0. Conspiracy/pseudoscience sites should "
            "score 8+.\n"
            "3. The formula scores are starting points — adjust them based on your holistic "
            "assessment of ALL the evidence.\n"
            "4. Do NOT reference MBFC, AllSides, Ad Fontes, or any bias-rating service.\n\n"
            "BIAS INFERENCE FROM RESEARCH SIGNALS (when editorial analysis is weak or absent):\n"
            "- History says 'supports Trump', 'right-leaning', 'conservative' → bias +5 to +9\n"
            "- History says 'socialist', 'left-wing', 'progressive advocacy' → bias -5 to -9\n"
            "- History says 'state-run media' of authoritarian regime → bias extreme in relevant direction\n"
            "- History says 'conspiracy theories', 'fake news' → bias extreme + factuality 8+\n"
            "- A bias score of 0.0 from the formula usually means no articles were available — "
            "it does NOT mean the outlet is centrist. You MUST override it using other signals.\n\n"
            "FACTUALITY CALIBRATION (the formula compresses toward 'Mostly Factual' — fight this):\n"
            "- Score 0.0-0.9 (Very High): Wire services, academic/scientific journals, "
            "outlets with 0 failed fact checks AND good sourcing AND transparent ownership\n"
            "- Score 0.1-1.9 (High): Major newspapers with strong sourcing and rare or no fact-check failures\n"
            "- Score 2.0-4.4 (Mostly Factual): Only ~15% of outlets truly belong here — "
            "do NOT dump everything in this range\n"
            "- Score 6.5-8.4 (Low): Multiple failed fact checks, poor sourcing, history of misinformation\n"
            "- Score 8.5-10.0 (Very Low): Fake news sites, conspiracy outlets, "
            "those described as 'unreliable' or 'fake news' in their history\n"
        )

        user_prompt = (
            f"Formula-computed scores:\n"
            f"  Bias: {formula_bias:.2f}\n"
            f"  Factuality: {formula_factuality:.2f}\n"
            f"  Evidence sufficient: {evidence_sufficient}\n\n"
            f"EVIDENCE:\n{evidence_text}\n\n"
            f"Based on ALL the evidence above, produce calibrated bias and factuality scores. "
            f"If the formula scores seem reasonable, keep them close. If the evidence suggests "
            f"the formula missed something (e.g., bias defaulted to 0.0 for a clearly biased "
            f"outlet), adjust significantly."
        )

        try:
            calibration_llm = get_llm(self._model, self._temperature).with_structured_output(
                CalibratedScores
            )
            result: CalibratedScores = calibration_llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
            logger.info(
                f"  - Calibration: bias {formula_bias:.2f}→{result.bias_score:.2f}, "
                f"factuality {formula_factuality:.2f}→{result.factuality_score:.2f}"
            )
            return result
        except Exception as e:
            logger.error(f"  - Calibration LLM failed: {e}; using formula scores")
            return CalibratedScores(
                bias_score=formula_bias,
                factuality_score=formula_factuality,
                bias_reasoning="Calibration failed; using formula scores",
                factuality_reasoning="Calibration failed; using formula scores",
            )

    def _build_report(
        self,
        url, domain, outlet_name, articles,
        traffic_data, media_type_result,
        editorial_bias_result, sourcing_result, pseudoscience_result,
        one_sidedness_result, fact_check_result, transparency_result,
        history, ownership, external_analyses,
        _profile_t0,
    ) -> ComprehensiveReportData:
        """Build the ComprehensiveReportData from analyzer results (shared by sequential/parallel)."""
        import time as _time
        from config import MIN_HEADLINES, MIN_FULL_STORIES

        # Evidence sufficiency gate
        headline_count = len(articles)
        full_story_count = sum(1 for a in articles if len(a.get("text", "")) >= 200)
        evidence_sufficient = (headline_count >= MIN_HEADLINES and full_story_count >= MIN_FULL_STORIES)
        insufficient_reason = None
        if not evidence_sufficient:
            parts = []
            if headline_count < MIN_HEADLINES:
                parts.append(f"only {headline_count}/{MIN_HEADLINES} headlines")
            if full_story_count < MIN_FULL_STORIES:
                parts.append(f"only {full_story_count}/{MIN_FULL_STORIES} full stories")
            insufficient_reason = "Insufficient evidence: " + ", ".join(parts)

        # --- BIAS: Weighted 4-category composite ---
        if editorial_bias_result:
            bias_score = editorial_bias_result.bias_score
            bias_label = editorial_bias_result.mbfc_label
            economic_score = editorial_bias_result.economic_score
            social_score = editorial_bias_result.social_score
            news_reporting_score = editorial_bias_result.news_reporting_score
            editorial_score = editorial_bias_result.editorial_bias_score
        else:
            # Fallback: estimate bias from external research signals instead of
            # defaulting to 0.0 (Least Biased) which causes massive centrality bias.
            bias_score = self._estimate_bias_from_research(
                external_analyses, history, ownership
            )
            bias_label = self._bias_score_to_label(bias_score)
            economic_score = 0.0
            social_score = 0.0
            news_reporting_score = 0.0
            editorial_score = 0.0
            logger.info(
                f"  - No editorial bias result; estimated bias={bias_score:.1f} "
                f"({bias_label}) from research signals"
            )

        # --- FACTUALITY: 4-category weighted (40/25/25/10) ---
        fact_check_score = fact_check_result.score
        sourcing_score = sourcing_result.score if sourcing_result else 5.0
        # Step 2c: When transparency analysis fails (no articles), use a less
        # punitive default. The old default of 5.0 pushed everything toward
        # "Mostly Factual". Use 3.0 (moderate) as a gentler unknown-state.
        transparency_score = transparency_result.score if transparency_result else 3.0
        one_sidedness_score = one_sidedness_result.score if one_sidedness_result else 3.0

        if pseudoscience_result and pseudoscience_result.promotes_pseudoscience:
            fact_check_score = max(fact_check_score, 5.0)

        factuality_score = (
            fact_check_score * 0.40
            + sourcing_score * 0.25
            + transparency_score * 0.25
            + one_sidedness_score * 0.10
        )
        factuality_score = min(10.0, factuality_score)

        # Capture intermediate values for recalculation/audit
        formula_factuality_score = factuality_score
        formula_bias_score = bias_score

        # Step 2b: Adjust factuality using research signals (external analyses,
        # history, ownership). This helps distinguish VHigh outlets (Reuters,
        # Smithsonian) from VLow outlets (conspiracy sites) when the formula
        # alone clusters everything around "Mostly Factual".
        research_factuality = self._estimate_factuality_from_research(
            external_analyses, history, ownership
        )
        if research_factuality is not None:
            # Blend formula score with research signal
            factuality_score = factuality_score * 0.6 + research_factuality * 0.4
            factuality_score = max(0.0, min(10.0, factuality_score))
            logger.info(
                f"  - Factuality adjusted by research signals: "
                f"formula→{factuality_score:.1f}, signal→{research_factuality:.1f}"
            )

        # Step 2d: LLM-based score calibration (holistic adjustment)
        if self._use_calibration:
            calibrated = self._calibrate_scores_with_llm(
                outlet_name=outlet_name,
                url=url,
                formula_bias=bias_score,
                formula_factuality=factuality_score,
                editorial_bias_result=editorial_bias_result,
                fact_check_result=fact_check_result,
                sourcing_result=sourcing_result,
                transparency_result=transparency_result,
                one_sidedness_result=one_sidedness_result,
                pseudoscience_result=pseudoscience_result,
                external_analyses=external_analyses,
                history=history,
                ownership=ownership,
                evidence_sufficient=evidence_sufficient,
            )
            bias_score = calibrated.bias_score
            bias_label = self._bias_score_to_label(bias_score)
            factuality_score = calibrated.factuality_score

        factuality_label = self._score_to_factuality_label(factuality_score)

        freedom_data = self._lookup_freedom_rating(self._sanitize_null(ownership.headquarters))

        # Fallback: use LLM-extracted country if headquarters parsing failed
        if not freedom_data["country"] and getattr(ownership, "country", None):
            fallback_freedom = self._lookup_freedom_rating(f", {ownership.country}")
            if fallback_freedom["country"]:
                freedom_data = fallback_freedom

        promotes_pseudoscience = (
            pseudoscience_result.promotes_pseudoscience
            if pseudoscience_result else False
        )
        is_extreme_bias = abs(bias_score) >= 8.0
        is_low_factuality = factuality_label in ("Low", "Very Low")
        lacks_transparency = transparency_score >= 8.0
        has_heavy_propaganda = one_sidedness_score >= 7.0
        is_questionable = (
            is_extreme_bias
            or is_low_factuality
            or (lacks_transparency and has_heavy_propaganda)
        )

        # Pro-Science detection: scientific outlet + respects consensus + low bias
        # When pseudoscience_result is None (no articles scraped), trust the
        # editorial bias analyzer's is_pro_science flag alone — a scientific
        # journal shouldn't fail just because we couldn't scrape articles.
        editorial_pro_science = getattr(editorial_bias_result, 'is_pro_science', False)
        respects_consensus = (
            pseudoscience_result.respects_scientific_consensus
            if pseudoscience_result is not None
            else True  # assume True when no articles to check
        )
        is_pro_science = (
            editorial_pro_science
            and respects_consensus
            and not promotes_pseudoscience
            and abs(bias_score) < 2.0
        )

        if promotes_pseudoscience:
            source_category = "Conspiracy/Pseudoscience"
        elif is_questionable:
            source_category = "Questionable"
        elif is_pro_science:
            source_category = "Pro-Science"
            bias_label = "Pro-Science"
        else:
            source_category = "News"

        credibility_score, credibility_label = self._calculate_credibility_score(
            factuality_label=factuality_label,
            factuality_score=factuality_score,
            bias_label=bias_label,
            traffic_tier=traffic_data.traffic_tier.value,
            domain_age_years=traffic_data.age_years,
            freedom_label=freedom_data.get("freedom_label"),
            promotes_pseudoscience=promotes_pseudoscience,
            is_questionable=is_questionable,
        )

        report = ComprehensiveReportData(
            target_url=url,
            target_domain=domain,
            outlet_name=outlet_name,
            bias_label=bias_label,
            bias_score=bias_score,
            factuality_label=factuality_label,
            factuality_score=factuality_score,
            credibility_label=credibility_label,
            credibility_score=credibility_score,
            source_category=source_category,
            economic_score=economic_score,
            social_score=social_score,
            news_reporting_score=news_reporting_score,
            editorial_score=editorial_score,
            media_type=media_type_result.media_type.value,
            traffic_tier=traffic_data.traffic_tier.value,
            domain_age_years=traffic_data.age_years,
            editorial_bias_result=editorial_bias_result,
            fact_check_result=fact_check_result,
            sourcing_result=sourcing_result,
            pseudoscience_result=pseudoscience_result,
            transparency_result=transparency_result,
            one_sidedness_result=one_sidedness_result,
            history_summary=history.summary,
            founding_year=history.founding_year,
            founder=history.founder,
            original_name=history.original_name,
            key_events=history.key_events or [],
            owner=self._sanitize_null(ownership.owner) or self._sanitize_null(ownership.parent_company),
            parent_company=self._sanitize_null(ownership.parent_company),
            funding_model=self._sanitize_null(ownership.funding_model),
            headquarters=self._sanitize_null(ownership.headquarters),
            ownership_notes=self._sanitize_null(ownership.notes),
            external_analyses=[
                a for a in external_analyses.analyses
                if not any(kw in (a.source_name or "").lower() for kw in ("mbfc", "media bias/fact check", "media bias fact check"))
                and not any(kw in (a.source_url or "").lower() for kw in ("mediabiasfactcheck",))
            ],
            **freedom_data,
            articles_index=[
                {"number": i + 1, "title": a.get("title", "Untitled"), "url": a.get("url", "")}
                for i, a in enumerate(articles)
            ],
            headline_count=headline_count,
            full_story_count=full_story_count,
            evidence_sufficient=evidence_sufficient,
            insufficient_evidence_reason=insufficient_reason,
            formula_factuality_score=formula_factuality_score,
            research_factuality_signal=research_factuality,
            formula_bias_score=formula_bias_score,
            analysis_date=datetime.now().strftime("%Y-%m-%d"),
            articles_analyzed=len(articles),
        )

        _profile_elapsed = _time.time() - _profile_t0
        logger.info(f"  - Profiling complete for {outlet_name} in {_profile_elapsed:.1f}s (bias={bias_score}, factuality={factuality_score:.1f})")
        return report

    def generate_report_text(self, report: ComprehensiveReportData) -> str:
        """
        Generate a human-readable MBFC-style report.

        Args:
            report: ComprehensiveReportData from profile()

        Returns:
            Formatted text report
        """
        lines = []

        # Header
        lines.append("=" * 70)
        lines.append(f"MEDIA BIAS/FACT CHECK REPORT: {report.outlet_name.upper()}")
        lines.append("=" * 70)
        lines.append(f"URL: {report.target_url}")
        lines.append(f"Analysis Date: {report.analysis_date}")
        lines.append("")

        # Quick Summary
        lines.append("QUICK SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Bias Rating:        {report.bias_label}")
        lines.append(f"  Factuality Rating:  {report.factuality_label}")
        lines.append(f"  Credibility:        {report.credibility_label}")
        lines.append(f"  Media Type:         {report.media_type}")
        lines.append(f"  Traffic:            {report.traffic_tier}")
        if report.domain_age_years:
            lines.append(f"  Domain Age:         {report.domain_age_years:.1f} years")
        lines.append("")

        # History
        lines.append("HISTORY")
        lines.append("-" * 40)
        if report.founding_year:
            lines.append(f"  Founded: {report.founding_year}")
        if report.founder:
            lines.append(f"  Founder(s): {report.founder}")
        if report.original_name:
            lines.append(f"  Original Name: {report.original_name}")
        if report.key_events:
            lines.append("  Key Events:")
            for event in report.key_events:
                lines.append(f"    - {event}")
        if report.history_summary:
            lines.append(f"\n  {report.history_summary}")
        lines.append("")

        # Ownership
        lines.append("FUNDED BY / OWNERSHIP")
        lines.append("-" * 40)
        if report.owner:
            lines.append(f"  Owner: {report.owner}")
        if report.parent_company and report.parent_company != report.owner:
            lines.append(f"  Parent Company: {report.parent_company}")
        if report.funding_model:
            lines.append(f"  Funding: {report.funding_model}")
        if report.headquarters:
            lines.append(f"  Headquarters: {report.headquarters}")
        if report.ownership_notes:
            lines.append(f"\n  {report.ownership_notes}")
        lines.append("")

        # Bias Analysis
        lines.append("BIAS ANALYSIS")
        lines.append("-" * 40)
        lines.append(f"  Overall Bias: {report.bias_label} (score: {report.bias_score:+.1f})")
        if report.editorial_bias_result:
            eb = report.editorial_bias_result
            if eb.uses_loaded_language:
                lines.append(f"  Uses Loaded Language: Yes")
                if eb.loaded_language_examples:
                    examples = ", ".join(eb.loaded_language_examples[:3])
                    lines.append(f"    Examples: {examples}")
            if eb.policy_positions:
                lines.append("  Policy Positions Detected:")
                for pos in eb.policy_positions[:3]:
                    lines.append(f"    - {pos.domain.value}: {pos.leaning.value}")
            lines.append(f"\n  Analysis: {eb.reasoning}")
        lines.append("")

        # Factuality Analysis
        lines.append("FACTUALITY ANALYSIS")
        lines.append("-" * 40)
        lines.append(f"  Factuality Rating: {report.factuality_label} (score: {report.factuality_score:.1f}/10)")

        if report.fact_check_result:
            fc = report.fact_check_result
            lines.append(f"\n  Fact Check Search Results:")
            lines.append(f"    Total Fact Checks Found: {fc.total_checks_count}")
            lines.append(f"    Failed Fact Checks: {fc.failed_checks_count}")
            if fc.findings:
                lines.append("    Recent Findings:")
                for finding in fc.findings[:3]:
                    lines.append(f"      - [{finding.verdict.value}] {finding.claim_summary[:60]}...")

        if report.sourcing_result:
            sr = report.sourcing_result
            lines.append(f"\n  Sourcing Quality:")
            lines.append(f"    Score: {sr.score:.1f}/10")
            lines.append(f"    Unique Sources: {sr.unique_domains}")
            lines.append(f"    Has Primary Sources: {'Yes' if sr.has_primary_sources else 'No'}")
            lines.append(f"    Has Wire Services: {'Yes' if sr.has_wire_services else 'No'}")
        lines.append("")

        # Pseudoscience
        if report.pseudoscience_result:
            lines.append("PSEUDOSCIENCE CHECK")
            lines.append("-" * 40)
            ps = report.pseudoscience_result
            lines.append(f"  Promotes Pseudoscience: {'Yes' if ps.promotes_pseudoscience else 'No'}")
            lines.append(f"  Respects Scientific Consensus: {'Yes' if ps.respects_scientific_consensus else 'No'}")
            if ps.categories_found:
                cats = ", ".join(c.value for c in ps.categories_found[:3])
                lines.append(f"  Categories Found: {cats}")
            lines.append(f"\n  Assessment: {ps.reasoning}")
            lines.append("")

        # External Analyses
        if report.external_analyses:
            lines.append("EXTERNAL ANALYSES")
            lines.append("-" * 40)
            for analysis in report.external_analyses[:3]:
                sentiment_emoji = {
                    "positive": "+",
                    "negative": "-",
                    "neutral": "~",
                    "mixed": "?"
                }.get(analysis.sentiment, "?")
                lines.append(f"  [{sentiment_emoji}] {analysis.source_name}")
                lines.append(f"      {analysis.summary}")
            lines.append("")

        # Footer
        lines.append("=" * 70)
        lines.append(f"Articles Analyzed: {report.articles_analyzed}")
        lines.append("Generated by Media Profiling System")
        lines.append("=" * 70)

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================


def research_outlet(url: str, outlet_name: Optional[str] = None) -> dict:
    """
    Convenience function to research an outlet without full profiling.

    Args:
        url: The outlet's URL
        outlet_name: Optional human-readable name

    Returns:
        Dict with history, ownership, and external analyses
    """
    researcher = MediaResearcher()
    domain = researcher._extract_domain(url)
    outlet_name = outlet_name or researcher.resolve_outlet_name(url, domain=domain)

    history = researcher.research_history(outlet_name, domain=domain)
    ownership = researcher.research_ownership(outlet_name, domain=domain)
    external = researcher.research_external_analysis(outlet_name, domain=domain)

    return {
        "outlet_name": outlet_name,
        "history": history.model_dump(),
        "ownership": ownership.model_dump(),
        "external_analyses": external.model_dump(),
    }


def profile_outlet(
    url: str,
    articles: list[dict[str, str]],
    outlet_name: Optional[str] = None,
) -> ComprehensiveReportData:
    """
    Convenience function to profile a media outlet.

    Args:
        url: The outlet's URL
        articles: List of article dicts with 'title' and 'text' keys
        outlet_name: Optional human-readable name

    Returns:
        ComprehensiveReportData with all analysis results
    """
    profiler = MediaProfiler()
    return profiler.profile(url, articles, outlet_name)


# =============================================================================
# CLI / Testing
# =============================================================================

if __name__ == "__main__":
    import sys

    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 70)
    print("MEDIA PROFILER - DEMO")
    print("=" * 70)

    # Test with sample articles
    sample_articles = [
        {
            "title": "Climate Change Policy Faces Opposition",
            "text": """
            The administration's new climate policy has drawn sharp criticism from
            industry groups who claim it will devastate the economy. Environmental
            advocates, however, argue the measures don't go far enough to address
            the urgent threat of global warming. According to a new report from the
            IPCC, immediate action is needed to prevent catastrophic warming.
            Critics on the right have called the policy "radical" and "job-killing,"
            while progressive groups say it represents a step in the right direction.
            The EPA cited studies from nature.gov and the Department of Energy
            in defending the new regulations.
            """
        },
        {
            "title": "Healthcare Reform Debate Intensifies",
            "text": """
            As healthcare costs continue to rise, lawmakers are divided on solutions.
            Progressive members of Congress are pushing for expanded Medicare coverage,
            while conservatives argue for market-based reforms. A new study from the
            Kaiser Family Foundation found that healthcare spending now accounts for
            nearly 20% of GDP. The American Medical Association has expressed concerns
            about both approaches, citing potential impacts on physician autonomy.
            """
        },
    ]

    # Test with a domain
    test_url = "https://www.bbc.com"

    print(f"\nProfiling: {test_url}")
    print("-" * 50)

    profiler = MediaProfiler()
    report = profiler.profile(test_url, sample_articles)

    # Generate and print text report
    report_text = profiler.generate_report_text(report)
    print("\n" + report_text)

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
