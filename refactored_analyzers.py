"""
Refactored analyzers using deterministic data retrieval and structured LLM outputs.

This module replaces heuristic-based methods with:
- LangChain's .with_structured_output() for type-safe LLM responses
- python-whois for deterministic domain age data
- DuckDuckGo search for external information gathering

Classes:
    OpinionAnalyzer: Classifies articles using LLM content analysis (no URL heuristics)
    TrafficLongevityAnalyzer: Gets domain age from WHOIS + traffic from search/LLM
    MediaTypeAnalyzer: Classifies media type using Wikipedia search + LLM
"""

import logging
import re
from datetime import date, datetime
from typing import Optional
from urllib.parse import urlparse

import os

try:
    import whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "python-whois not installed. WHOIS lookups will be skipped. "
        "Install with: pip install python-whois"
    )

WHOIS_DEBUG = os.environ.get("WHOIS_DEBUG", "0") == "1"
# DDGS is no longer imported directly — search is delegated to search_backends
from langchain_openai import ChatOpenAI

from schemas import (
    ArticleClassification,
    ArticleType,
    BatchArticleClassification,
    BiasDirection,
    ClaimSource,
    EditorialBiasLLMOutput,
    EditorialBiasResult,
    FactCheckAnalysisResult,
    FactCheckFinding,
    FactCheckLLMOutput,
    FactCheckSource,
    FactCheckVerdict,
    MediaType,
    MediaTypeClassification,
    MediaTypeLLMOutput,
    MediaTypeSource,
    PolicyDomain,
    PolicyPosition,
    PseudoscienceAnalysisResult,
    PseudoscienceCategory,
    PseudoscienceIndicator,
    PseudoscienceLLMOutput,
    PseudoscienceSeverity,
    SourceAssessment,
    SourceQuality,
    SourcingAnalysisResult,
    SourcingLLMOutput,
    TransparencyLLMOutput,
    TransparencyResult,
    OneSidednessLLMOutput,
    OneSidednessResult,
    TrafficData,
    TrafficEstimate,
    TrafficSource,
    TrafficTier,
)

logger = logging.getLogger(__name__)


# =============================================================================
# LLM Configuration
# =============================================================================


def get_llm(model: str = "gpt-5-mini-2025-08-07", temperature: float = 0.0) -> ChatOpenAI:
    """
    Get a configured LLM instance.

    Args:
        model: The OpenAI model to use
        temperature: Temperature setting (0 for deterministic)

    Returns:
        Configured ChatOpenAI instance
    """
    from cost_tracker import LangChainUsageCallback
    return ChatOpenAI(
        model=model, temperature=temperature,
        callbacks=[LangChainUsageCallback()],
    )


# =============================================================================
# OpinionAnalyzer
# =============================================================================


class OpinionAnalyzer:
    """
    Analyzes article content to classify it as News, Opinion, Satire, or PR.

    This analyzer uses pure content analysis via LLM - it does NOT rely on
    URL patterns or title heuristics. Classification is based on:
    - Writing style and tone
    - Use of first-person vs third-person
    - Presence of subjective language
    - Factual reporting vs commentary patterns

    Attributes:
        llm: The LangChain LLM with structured output binding
        max_text_chars: Maximum characters of text to analyze (default 1000)
    """

    SYSTEM_PROMPT = """You are an expert media analyst specializing in distinguishing
between different types of journalistic content. Your task is to classify articles
based on their actual content, writing style, and journalistic intent.

Classification Guidelines:

NEWS:
- Objective, fact-based reporting
- Third-person perspective
- Balanced presentation of multiple viewpoints
- Attribution to sources
- Inverted pyramid structure (most important facts first)
- Minimal adjectives and loaded language

OPINION:
- First-person perspective or clear editorial voice
- Subjective analysis and commentary
- Author's personal views and judgments
- Persuasive language and arguments
- May include "I think", "we should", "in my view"
- Editorial, op-ed, column, or analysis pieces

SATIRE:
- Exaggerated or absurd scenarios
- Ironic or sarcastic tone
- Humorous intent
- Clearly implausible claims played straight
- Parody of news formats

PR (Press Release / Promotional):
- Promotional language about a company/product/person
- One-sided positive framing
- Corporate speak and marketing language
- Quotes primarily from the subject being promoted
- Announcement-style structure

Analyze the CONTENT and STYLE, not the URL or publication name."""

    def __init__(
        self,
        model: str = "gpt-5-mini-2025-08-07",
        temperature: float = 0.0,
        max_text_chars: int = 1000,
    ):
        """
        Initialize the OpinionAnalyzer.

        Args:
            model: OpenAI model to use
            temperature: LLM temperature (0 for deterministic)
            max_text_chars: Maximum characters of article text to analyze
        """
        self.model = model
        self.llm = get_llm(model, temperature).with_structured_output(
            ArticleClassification
        )
        self.max_text_chars = max_text_chars

    def analyze(self, title: str, text: str) -> ArticleClassification:
        """
        Classify an article based on its title and text content.

        Args:
            title: The article headline/title
            text: The article body text

        Returns:
            ArticleClassification with type, confidence, and reasoning
        """
        # Truncate text to max_text_chars
        text_snippet = text[: self.max_text_chars] if text else ""

        user_prompt = f"""Classify the following article:

TITLE: {title}

TEXT (first {self.max_text_chars} characters):
{text_snippet}

Based on the writing style, tone, and content, classify this article."""

        try:
            result: ArticleClassification = self.llm.invoke(
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return result

        except Exception as e:
            logger.error(f"OpinionAnalyzer failed: {e}")
            # Return a safe default with low confidence
            return ArticleClassification(
                article_type=ArticleType.NEWS,
                confidence=0.0,
                reasoning=f"Classification failed due to error: {str(e)}",
            )

    def analyze_batch(
        self, articles: list[dict[str, str]]
    ) -> list[ArticleClassification]:
        """
        Classify multiple articles in a single batched LLM call.

        Sends all articles in one prompt and gets back all classifications
        at once, reducing N LLM calls to 1.

        Args:
            articles: List of dicts with 'title' and 'text' keys

        Returns:
            List of ArticleClassification results (same order as input)
        """
        if not articles:
            return []

        if len(articles) == 1:
            return [self.analyze(
                title=articles[0].get("title", ""),
                text=articles[0].get("text", ""),
            )]

        # Build a single prompt with all articles numbered
        article_sections = []
        for i, article in enumerate(articles):
            title = article.get("title", "Untitled")
            text = article.get("text", "")
            text_snippet = text[: self.max_text_chars] if text else ""
            article_sections.append(
                f"--- ARTICLE {i + 1} ---\n"
                f"TITLE: {title}\n"
                f"TEXT (first {self.max_text_chars} characters):\n{text_snippet}"
            )

        combined = "\n\n".join(article_sections)
        user_prompt = (
            f"Classify each of the following {len(articles)} articles. "
            f"Return exactly {len(articles)} classifications in the same order.\n\n"
            f"{combined}\n\n"
            f"Based on the writing style, tone, and content, classify each article."
        )

        try:
            batch_llm = get_llm(
                self.model,
                0.0,
            ).with_structured_output(BatchArticleClassification)

            result: BatchArticleClassification = batch_llm.invoke(
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )

            # Validate we got the right number of results
            if len(result.classifications) == len(articles):
                return result.classifications

            # If count mismatch, pad or truncate
            logger.warning(
                f"OpinionAnalyzer batch returned {len(result.classifications)} "
                f"classifications for {len(articles)} articles, falling back to individual calls"
            )
        except Exception as e:
            logger.warning(f"OpinionAnalyzer batch failed ({e}), falling back to individual calls")

        # Fallback: individual calls (original behavior)
        results = []
        for article in articles:
            result = self.analyze(
                title=article.get("title", ""),
                text=article.get("text", ""),
            )
            results.append(result)
        return results


# =============================================================================
# Tranco List Configuration
# =============================================================================

# Default Tranco list location and URL
TRANCO_DEFAULT_PATH = "tranco_top1m.csv"
TRANCO_DOWNLOAD_URL = "https://tranco-list.eu/top-1m.csv.zip"  # Stable URL for latest top 1M list

# Default tier thresholds based on Tranco rank
DEFAULT_TRANCO_THRESHOLDS = {
    "HIGH": 10_000,      # Rank < 10,000 = HIGH traffic
    "MEDIUM": 100_000,   # Rank < 100,000 = MEDIUM traffic
    "LOW": 1_000_000,    # Rank < 1,000,000 = LOW traffic
}


# =============================================================================
# TrafficLongevityAnalyzer
# =============================================================================


class TrafficLongevityAnalyzer:
    """
    Analyzes domain traffic and longevity using a hybrid deterministic + LLM approach.

    This analyzer implements a two-tier strategy:
    1. **Deterministic (Tranco)**: First checks the Tranco Top 1M list for instant,
       reproducible ranking data. This replaces the deprecated Alexa Rank.
    2. **LLM Fallback**: For domains not in Tranco, searches for traffic data
       via DuckDuckGo and uses LLM to parse the results.

    Domain age is always retrieved deterministically via python-whois.

    NOTE: MBFC uses SimilarWeb page views for traffic tiers (Under 250K=Minimal,
    250K-3M=Medium, Over 3M=High). We use Tranco ranking as a proxy (rank<10K=High,
    10K-100K=Medium, >100K=Low/Minimal). This is an approximation — Tranco measures
    domain popularity differently from SimilarWeb monthly page views.

    Attributes:
        llm: LangChain LLM with structured output for traffic parsing
        search: DuckDuckGo search instance
        tranco_data: Dict mapping domain -> rank (loaded from Tranco list)
        tranco_loaded: Whether Tranco list is available
        thresholds: Dict mapping tier names to rank cutoffs
    """

    TRAFFIC_PARSE_PROMPT = """You are analyzing search results to estimate website traffic.

Based on the search snippet provided, determine the traffic tier:

HIGH: Major websites with millions of monthly visits (>10M)
      - Nationally/globally recognized brands
      - Major news outlets, social media, e-commerce giants

MEDIUM: Established websites with hundreds of thousands to millions of visits (100K-10M)
        - Regional news outlets
        - Popular niche websites
        - Well-known blogs or specialty sites

LOW: Smaller websites with tens of thousands of visits (10K-100K)
     - Local news sites
     - Small business websites
     - Niche community sites

MINIMAL: Very small websites with under 10K visits
         - Personal blogs
         - New or obscure websites
         - Sites with little web presence

UNKNOWN: If the search results don't provide enough information

Look for indicators like:
- Explicit traffic numbers (e.g., "10M monthly visits")
- Tranco/Similarweb/Semrush rankings
- Descriptions of reach/popularity
- Comparisons to known sites"""

    def __init__(
        self,
        model: str = "gpt-5-mini-2025-08-07",
        temperature: float = 0.0,
        tranco_path: Optional[str] = None,
        auto_download_tranco: bool = True,
        thresholds: Optional[dict[str, int]] = None,
        search_backend=None,
    ):
        """
        Initialize the TrafficLongevityAnalyzer.

        Args:
            model: OpenAI model to use
            temperature: LLM temperature (0 for deterministic)
            tranco_path: Path to Tranco CSV file (default: tranco_top1m.csv)
            auto_download_tranco: Whether to auto-download Tranco if missing
            thresholds: Custom tier thresholds dict (keys: HIGH, MEDIUM, LOW)
            search_backend: Optional SearchBackend instance (defaults to DDGSearchBackend)
        """
        self.llm = get_llm(model, temperature).with_structured_output(TrafficEstimate)
        if search_backend is not None:
            self.search_backend = search_backend
        else:
            from search_backends import DDGSearchBackend
            self.search_backend = DDGSearchBackend()
        self.thresholds = thresholds or DEFAULT_TRANCO_THRESHOLDS.copy()

        # Initialize Tranco data
        self.tranco_data: dict[str, int] = {}
        self.tranco_loaded = False
        self._tranco_path = tranco_path or TRANCO_DEFAULT_PATH

        # Try to load Tranco list
        self._load_tranco_list(auto_download=auto_download_tranco)

    def _load_tranco_list(self, auto_download: bool = True) -> bool:
        """
        Load the Tranco top 1M list into memory.

        Args:
            auto_download: Whether to download if file doesn't exist

        Returns:
            True if loaded successfully, False otherwise
        """
        import os

        tranco_path = self._tranco_path

        # Check if file exists and is not a git-lfs pointer
        is_lfs_pointer = False
        if os.path.exists(tranco_path):
            try:
                with open(tranco_path, "r", encoding="utf-8") as f:
                    first_line = f.readline()
                if first_line.startswith("version https://git-lfs"):
                    is_lfs_pointer = True
                    logger.info(f"Tranco file at {tranco_path} is a git-lfs pointer, need to download actual data")
            except Exception:
                pass

        if not os.path.exists(tranco_path) or is_lfs_pointer:
            if auto_download:
                logger.info(f"Tranco list not found at {tranco_path}, downloading...")
                if not self._download_tranco_list(tranco_path):
                    logger.warning("Failed to download Tranco list, will use LLM fallback only")
                    return False
            else:
                logger.warning(f"Tranco list not found at {tranco_path}, will use LLM fallback only")
                return False

        # Load CSV into dict (rank -> domain mapping, we need domain -> rank)
        try:
            with open(tranco_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or "," not in line:
                        continue
                    parts = line.split(",", 1)
                    if len(parts) == 2:
                        rank_str, domain = parts
                        try:
                            rank = int(rank_str)
                            # Store domain -> rank mapping
                            self.tranco_data[domain.lower()] = rank
                        except ValueError:
                            continue

            self.tranco_loaded = len(self.tranco_data) > 0
            if self.tranco_loaded:
                logger.info(f"Loaded Tranco list with {len(self.tranco_data):,} domains")
            return self.tranco_loaded

        except Exception as e:
            logger.error(f"Failed to load Tranco list: {e}")
            return False

    def _download_tranco_list(self, save_path: str) -> bool:
        """
        Download the Tranco top 1M list.

        Args:
            save_path: Path to save the downloaded file

        Returns:
            True if download successful, False otherwise
        """
        import urllib.request
        import zipfile
        import io

        try:
            logger.info(f"Downloading Tranco list from {TRANCO_DOWNLOAD_URL}...")

            # Download the file
            with urllib.request.urlopen(TRANCO_DOWNLOAD_URL, timeout=60) as response:
                content = response.read()

            # Check if it's a zip file (Tranco sometimes serves zipped)
            if content[:2] == b"PK":  # ZIP file magic bytes
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    # Extract the first CSV file
                    for name in zf.namelist():
                        if name.endswith(".csv"):
                            with zf.open(name) as csv_file:
                                content = csv_file.read()
                            break

            # Save to disk
            with open(save_path, "wb") as f:
                f.write(content)

            logger.info(f"Successfully downloaded Tranco list to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download Tranco list: {e}")
            return False

    def _get_tranco_rank(self, domain: str) -> Optional[int]:
        """
        Look up a domain's rank in the Tranco list.

        Args:
            domain: The domain to look up (e.g., "bbc.com")

        Returns:
            Rank (1 = most popular) or None if not found
        """
        if not self.tranco_loaded:
            return None

        # Normalize domain
        domain_lower = domain.lower().strip()

        # Direct lookup
        if domain_lower in self.tranco_data:
            return self.tranco_data[domain_lower]

        # Try with www prefix
        if not domain_lower.startswith("www."):
            www_domain = f"www.{domain_lower}"
            if www_domain in self.tranco_data:
                return self.tranco_data[www_domain]

        return None

    def _rank_to_tier(self, rank: int) -> TrafficTier:
        """
        Convert a Tranco rank to a traffic tier.

        Args:
            rank: The Tranco rank (1 = most popular)

        Returns:
            TrafficTier based on configured thresholds
        """
        if rank < self.thresholds["HIGH"]:
            return TrafficTier.HIGH
        elif rank < self.thresholds["MEDIUM"]:
            return TrafficTier.MEDIUM
        elif rank < self.thresholds["LOW"]:
            return TrafficTier.LOW
        else:
            return TrafficTier.MINIMAL

    def _extract_domain(self, url: str) -> str:
        """Extract the root domain from a URL."""
        parsed = urlparse(url if url.startswith("http") else f"https://{url}")
        domain = parsed.netloc or parsed.path
        # Remove www. prefix for consistency
        domain = re.sub(r"^www\.", "", domain)
        # Remove any path components
        domain = domain.split("/")[0]
        return domain.lower()

    def _get_whois_data(self, domain: str) -> tuple[Optional[date], bool, Optional[str]]:
        """
        Get domain creation date from WHOIS.
        Compatible with both 'whois' and 'python-whois' packages.
        Set WHOIS_DEBUG=1 environment variable for verbose debug output.
        """
        if not WHOIS_AVAILABLE:
            msg = "python-whois not installed — skipping WHOIS lookup"
            logger.info(f"[WHOIS] {msg} for {domain}")
            return None, False, msg

        try:
            # Try python-whois syntax first
            if hasattr(whois, 'whois'):
                if WHOIS_DEBUG:
                    logger.info(f"[WHOIS DEBUG] Using whois.whois() for {domain}")
                w = whois.whois(domain)
            # Try standard whois syntax second
            elif hasattr(whois, 'query'):
                if WHOIS_DEBUG:
                    logger.info(f"[WHOIS DEBUG] Using whois.query() for {domain}")
                w = whois.query(domain)
            else:
                return None, False, "Unknown whois library installed"

            if WHOIS_DEBUG:
                logger.info(f"[WHOIS DEBUG] Raw response for {domain}: {w}")

            # Extract creation date safely
            creation_date = w.creation_date

            if WHOIS_DEBUG:
                logger.info(f"[WHOIS DEBUG] Raw creation_date for {domain}: {creation_date} (type: {type(creation_date).__name__})")

            # Handle list of dates (some registrars return multiple)
            if isinstance(creation_date, list):
                creation_date = creation_date[0]

            # Convert datetime to date if needed
            if isinstance(creation_date, datetime):
                creation_date = creation_date.date()

            # Handle string dates (common in 'whois' package)
            if isinstance(creation_date, str):
                from dateutil import parser
                try:
                    creation_date = parser.parse(creation_date).date()
                except Exception as parse_err:
                    if WHOIS_DEBUG:
                        logger.warning(f"[WHOIS DEBUG] Failed to parse date string '{creation_date}': {parse_err}")

            if creation_date:
                if WHOIS_DEBUG:
                    logger.info(f"[WHOIS DEBUG] Resolved creation_date for {domain}: {creation_date}")
                return creation_date, True, None

            return None, False, "No creation date in WHOIS response"

        except Exception as e:
            # Catch-all for other errors
            error_msg = f"WHOIS error ({type(e).__name__}): {str(e)}"
            logger.warning(f"{error_msg} for {domain}")
            if WHOIS_DEBUG:
                logger.exception(f"[WHOIS DEBUG] Full traceback for {domain}")
            return None, False, error_msg

    def _calculate_age_years(self, creation_date: Optional[date]) -> Optional[float]:
        """Calculate domain age in years from creation date."""
        if not creation_date:
            return None
        today = date.today()
        delta = today - creation_date
        return round(delta.days / 365.25, 2)

    def _search_traffic_info(self, domain: str) -> Optional[str]:
        """
        Search for traffic information about a domain.

        Uses an improved query targeting traffic aggregator sites.

        Args:
            domain: The domain to search for

        Returns:
            Search result snippet or None
        """
        # Improved query per Gemini's suggestion - targets multiple traffic data sources
        query = f"{domain} traffic stats similarweb hypestat semrush"
        try:
            results = self.search_backend.search(query, max_results=5)
            if results:
                snippets = []
                for r in results[:5]:
                    title = r.get("title", "")
                    body = r.get("body", "")
                    snippets.append(f"{title}: {body}")
                return "\n".join(snippets)
            return None
        except Exception as e:
            logger.warning(f"Traffic search failed for {domain}: {e}")
            return None

    def _parse_traffic_with_llm(self, domain: str, snippet: str) -> TrafficEstimate:
        """
        Use LLM to parse traffic tier from search snippet.

        Args:
            domain: The domain being analyzed
            snippet: Search result snippet to parse

        Returns:
            TrafficEstimate with tier and reasoning
        """
        user_prompt = f"""Analyze the following search results for {domain} and estimate the traffic tier:

SEARCH RESULTS:
{snippet}

Determine the traffic tier based on any traffic data, rankings, or popularity indicators found."""

        try:
            result: TrafficEstimate = self.llm.invoke(
                [
                    {"role": "system", "content": self.TRAFFIC_PARSE_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return result

        except Exception as e:
            logger.error(f"Traffic LLM parsing failed: {e}")
            return TrafficEstimate(
                traffic_tier=TrafficTier.UNKNOWN,
                monthly_visits_estimate=None,
                confidence=0.0,
                reasoning=f"LLM parsing failed: {str(e)}",
            )

    def analyze(self, url_or_domain: str) -> TrafficData:
        """
        Analyze traffic and longevity for a domain using hybrid approach.

        Strategy:
        1. Always get WHOIS data for domain age (deterministic)
        2. Check Tranco list first (deterministic, instant, high confidence)
        3. If not in Tranco, fall back to DuckDuckGo search + LLM parsing

        Args:
            url_or_domain: URL or domain to analyze

        Returns:
            TrafficData with WHOIS and traffic information
        """
        domain = self._extract_domain(url_or_domain)

        # 1. Get deterministic WHOIS data
        creation_date, whois_success, whois_error = self._get_whois_data(domain)
        age_years = self._calculate_age_years(creation_date)

        # 2. Try Tranco lookup first (deterministic)
        tranco_rank = self._get_tranco_rank(domain)

        if tranco_rank is not None:
            # Found in Tranco - use deterministic ranking
            traffic_tier = self._rank_to_tier(tranco_rank)
            return TrafficData(
                domain=domain,
                creation_date=creation_date,
                age_years=age_years,
                traffic_tier=traffic_tier,
                monthly_visits_estimate=None,  # Tranco doesn't provide this
                traffic_confidence=1.0,  # Deterministic = 100% confidence
                traffic_source=TrafficSource.TRANCO,
                tranco_rank=tranco_rank,
                whois_success=whois_success,
                whois_error=whois_error,
                traffic_search_snippet=None,
            )

        # 3. Fall back to LLM-based estimation
        traffic_snippet = self._search_traffic_info(domain)

        if traffic_snippet:
            traffic_estimate = self._parse_traffic_with_llm(domain, traffic_snippet)
            return TrafficData(
                domain=domain,
                creation_date=creation_date,
                age_years=age_years,
                traffic_tier=traffic_estimate.traffic_tier,
                monthly_visits_estimate=traffic_estimate.monthly_visits_estimate,
                traffic_confidence=traffic_estimate.confidence,
                traffic_source=TrafficSource.LLM,
                tranco_rank=None,
                whois_success=whois_success,
                whois_error=whois_error,
                traffic_search_snippet=traffic_snippet[:500] if traffic_snippet else None,
            )

        # 4. No data available - return fallback
        return TrafficData(
            domain=domain,
            creation_date=creation_date,
            age_years=age_years,
            traffic_tier=TrafficTier.UNKNOWN,
            monthly_visits_estimate=None,
            traffic_confidence=0.0,
            traffic_source=TrafficSource.FALLBACK,
            tranco_rank=None,
            whois_success=whois_success,
            whois_error=whois_error,
            traffic_search_snippet=None,
        )

    def get_tranco_stats(self) -> dict:
        """
        Get statistics about the loaded Tranco list.

        Returns:
            Dict with Tranco list stats
        """
        return {
            "loaded": self.tranco_loaded,
            "total_domains": len(self.tranco_data),
            "thresholds": self.thresholds,
            "path": self._tranco_path,
        }


# =============================================================================
# MediaTypeAnalyzer Configuration
# =============================================================================

# Default known media types lookup file
KNOWN_MEDIA_TYPES_PATH = "known_media_types.csv"


# =============================================================================
# MediaTypeAnalyzer
# =============================================================================


class MediaTypeAnalyzer:
    """
    Classifies media outlet type using a hybrid deterministic + LLM approach.

    This analyzer implements a two-tier strategy:
    1. **Deterministic (Lookup)**: First checks the known_media_types.csv for instant,
       reproducible classification of major outlets.
    2. **LLM Fallback**: For unknown domains, searches for information and uses
       LLM to parse the results.

    Attributes:
        llm: LangChain LLM with structured output for parsing
        search: DuckDuckGo search instance
        known_types: Dict mapping domain -> MediaType (from lookup file)
        lookup_loaded: Whether lookup table is available
    """

    SYSTEM_PROMPT = """You are classifying the type of media outlet based on search results.

Media Type Definitions:

TV: Television broadcast networks and channels
    - Examples: CNN, BBC, Fox News, NBC, ABC, MSNBC, Al Jazeera

NEWSPAPER: Traditional print newspapers (may have online presence)
    - Examples: New York Times, Washington Post, The Guardian, Wall Street Journal

WEBSITE: Online-only news or content sites (digital native)
    - Examples: Vox, BuzzFeed News, The Daily Wire, Axios, Politico, HuffPost

MAGAZINE: Periodical publications (weekly/monthly)
    - Examples: Time, The Economist, The Atlantic, Newsweek, The New Yorker

RADIO: Radio broadcast networks
    - Examples: NPR, BBC Radio, Voice of America

NEWS_AGENCY: Wire services that provide content to other outlets
    - Examples: Reuters, Associated Press (AP), AFP, UPI

BLOG: Personal or small group blogs
    - Usually individual authors, informal style

PODCAST: Audio-first media
    - Examples: The Daily, Pod Save America

STREAMING: Streaming-first media services
    - Examples: Netflix, YouTube (when used as primary platform)

UNKNOWN: Cannot determine from available information

Classify based on the PRIMARY format of the outlet, not secondary formats.
For example, NYT is a NEWSPAPER even though they have a website and podcasts."""

    def __init__(
        self,
        model: str = "gpt-5-mini-2025-08-07",
        temperature: float = 0.0,
        lookup_path: Optional[str] = None,
        search_backend=None,
    ):
        """
        Initialize the MediaTypeAnalyzer.

        Args:
            model: OpenAI model to use
            temperature: LLM temperature (0 for deterministic)
            lookup_path: Path to known_media_types.csv (default: known_media_types.csv)
            search_backend: Optional SearchBackend instance (defaults to DDGSearchBackend)
        """
        self.llm = get_llm(model, temperature).with_structured_output(MediaTypeLLMOutput)
        if search_backend is not None:
            self.search_backend = search_backend
        else:
            from search_backends import DDGSearchBackend
            self.search_backend = DDGSearchBackend()

        # Initialize lookup data
        self.known_types: dict[str, MediaType] = {}
        self.lookup_loaded = False
        self._lookup_path = lookup_path or KNOWN_MEDIA_TYPES_PATH

        # Load lookup table
        self._load_known_types()

    def _load_known_types(self) -> bool:
        """
        Load the known media types lookup table.

        Returns:
            True if loaded successfully, False otherwise
        """
        import os
        import csv

        lookup_path = self._lookup_path

        if not os.path.exists(lookup_path):
            logger.warning(f"Known media types file not found at {lookup_path}, will use LLM only")
            return False

        try:
            with open(lookup_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    # Skip comments and empty lines
                    if not row or row[0].startswith("#"):
                        continue
                    if len(row) >= 2:
                        domain = row[0].strip().lower()
                        media_type_str = row[1].strip()

                        # Map string to MediaType enum
                        try:
                            media_type = MediaType(media_type_str)
                            self.known_types[domain] = media_type
                        except ValueError:
                            # Try case-insensitive match
                            for mt in MediaType:
                                if mt.value.lower() == media_type_str.lower():
                                    self.known_types[domain] = mt
                                    break

            self.lookup_loaded = len(self.known_types) > 0
            if self.lookup_loaded:
                logger.info(f"Loaded {len(self.known_types)} known media types from {lookup_path}")
            return self.lookup_loaded

        except Exception as e:
            logger.error(f"Failed to load known media types: {e}")
            return False

    def _lookup_media_type(self, domain: str) -> Optional[MediaType]:
        """
        Look up a domain's media type in the known types table.

        Args:
            domain: The domain to look up (e.g., "bbc.com")

        Returns:
            MediaType if found, None otherwise
        """
        if not self.lookup_loaded:
            return None

        # Normalize domain
        domain_lower = domain.lower().strip()

        # Direct lookup
        if domain_lower in self.known_types:
            return self.known_types[domain_lower]

        # Try with www prefix
        if not domain_lower.startswith("www."):
            www_domain = f"www.{domain_lower}"
            if www_domain in self.known_types:
                return self.known_types[www_domain]

        # Try without subdomain (e.g., news.bbc.com -> bbc.com)
        parts = domain_lower.split(".")
        if len(parts) > 2:
            base_domain = ".".join(parts[-2:])
            if base_domain in self.known_types:
                return self.known_types[base_domain]

        return None

    def _extract_domain(self, url: str) -> str:
        """Extract the root domain from a URL."""
        parsed = urlparse(url if url.startswith("http") else f"https://{url}")
        domain = parsed.netloc or parsed.path
        # Remove www. prefix for consistency
        domain = re.sub(r"^www\.", "", domain)
        # Remove any path components
        domain = domain.split("/")[0]
        return domain.lower()

    def _extract_site_name(self, url_or_domain: str) -> str:
        """
        Extract a clean site name from URL or domain.

        Args:
            url_or_domain: URL or domain string

        Returns:
            Clean site name for searching
        """
        parsed = urlparse(
            url_or_domain if url_or_domain.startswith("http") else f"https://{url_or_domain}"
        )
        domain = parsed.netloc or parsed.path
        domain = re.sub(r"^www\.", "", domain)

        # Extract site name (remove TLD for cleaner search)
        site_name = domain.split(".")[0]

        # Title case for better search results
        return site_name.replace("-", " ").replace("_", " ").title()

    def _search_media_type(self, site_name: str, domain: str) -> Optional[str]:
        """
        Search for media type information using improved query.

        Args:
            site_name: Clean site name
            domain: Full domain for search

        Returns:
            Search result snippet or None
        """
        # Improved query per PI's suggestion - more direct question format
        query = f'"{domain}" type of media outlet newspaper television website magazine'

        try:
            results = self.search_backend.search(query, max_results=5)

            if not results:
                # Fallback query - Wikipedia focused
                query = f"{site_name} wikipedia media company"
                results = self.search_backend.search(query, max_results=3)

            if results:
                snippets = []
                for r in results[:5]:
                    title = r.get("title", "")
                    body = r.get("body", "")
                    snippets.append(f"{title}: {body}")
                return "\n".join(snippets)

            return None

        except Exception as e:
            logger.warning(f"Media type search failed for {site_name}: {e}")
            return None

    def _parse_with_llm(self, site_name: str, domain: str, snippet: str) -> MediaTypeLLMOutput:
        """
        Use LLM to parse media type from search snippet.

        Args:
            site_name: Clean site name
            domain: Full domain
            snippet: Search result snippet to parse

        Returns:
            MediaTypeLLMOutput with type, confidence, and reasoning
        """
        user_prompt = f"""Classify the media type for: {site_name} ({domain})

SEARCH RESULTS:
{snippet}

Based on these search results, what type of media outlet is this?"""

        try:
            result: MediaTypeLLMOutput = self.llm.invoke(
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return result

        except Exception as e:
            logger.error(f"MediaTypeAnalyzer LLM call failed: {e}")
            return MediaTypeLLMOutput(
                media_type=MediaType.UNKNOWN,
                confidence=0.0,
                reasoning=f"LLM classification failed: {str(e)}",
            )

    def analyze(self, url_or_domain: str) -> MediaTypeClassification:
        """
        Classify the media type for a given URL or domain using hybrid approach.

        Strategy:
        1. Check known_media_types.csv first (deterministic, instant)
        2. If not found, search and use LLM to classify

        Args:
            url_or_domain: URL or domain to classify

        Returns:
            MediaTypeClassification with type, confidence, source, and reasoning
        """
        domain = self._extract_domain(url_or_domain)
        site_name = self._extract_site_name(url_or_domain)

        # 1. Try deterministic lookup first
        known_type = self._lookup_media_type(domain)

        if known_type is not None:
            return MediaTypeClassification(
                media_type=known_type,
                confidence=1.0,  # Deterministic = 100% confidence
                source=MediaTypeSource.LOOKUP,
                source_snippet=None,
                reasoning=f"Found in known media types database as {known_type.value}",
            )

        # 2. Fall back to search + LLM
        search_snippet = self._search_media_type(site_name, domain)

        if search_snippet:
            llm_result = self._parse_with_llm(site_name, domain, search_snippet)
            return MediaTypeClassification(
                media_type=llm_result.media_type,
                confidence=llm_result.confidence,
                source=MediaTypeSource.LLM,
                source_snippet=search_snippet[:500] if search_snippet else None,
                reasoning=llm_result.reasoning,
            )

        # 3. No data available - return fallback
        return MediaTypeClassification(
            media_type=MediaType.UNKNOWN,
            confidence=0.0,
            source=MediaTypeSource.FALLBACK,
            source_snippet=None,
            reasoning="Could not find information about this media outlet",
        )

    def get_lookup_stats(self) -> dict:
        """
        Get statistics about the loaded lookup table.

        Returns:
            Dict with lookup table stats
        """
        # Count by type
        type_counts = {}
        for media_type in self.known_types.values():
            type_counts[media_type.value] = type_counts.get(media_type.value, 0) + 1

        return {
            "loaded": self.lookup_loaded,
            "total_domains": len(self.known_types),
            "by_type": type_counts,
            "path": self._lookup_path,
        }


# =============================================================================
# FactCheckSearcher Configuration
# =============================================================================

# Sites to search for fact checks — imported from central config.
# mediabiasfactcheck.com is excluded (anti-contamination).
from config import FACTCHECK_SEARCH_SITES as FACTCHECK_SITES
from config import FACTCHECK_ATTRIBUTE_CLAIM_SOURCE

# Mapping of verdicts to their "failed" status
FAILED_VERDICTS = {
    FactCheckVerdict.FALSE,
    FactCheckVerdict.MOSTLY_FALSE,
    FactCheckVerdict.PANTS_ON_FIRE,
    FactCheckVerdict.MISLEADING,
}


# =============================================================================
# FactCheckSearcher
# =============================================================================


class FactCheckSearcher:
    """
    Searches fact-checker sites for fact checks about a media outlet.

    This analyzer replaces keyword heuristics with direct search on reputable
    fact-checking sites. It searches 5 major fact-checkers and uses an LLM
    to parse the results into structured findings.

    Strategy:
    1. Search each fact-checker site: `site:{site} "{domain}" OR "{outlet_name}"`
    2. Combine all search snippets
    3. Pass to LLM to extract fact check findings (verdicts, claims)
    4. Calculate score based on failed checks count

    Score Calculation:
    - 0 failed checks = 0.0 (excellent)
    - 1-2 failed checks = 2.0-4.0
    - 3-5 failed checks = 5.0-7.0
    - 6+ failed checks = 8.0-10.0 (very poor)

    Attributes:
        llm: LangChain LLM with structured output for parsing
        search: DuckDuckGo search instance
    """

    SYSTEM_PROMPT = """You are an expert at parsing fact-check search results.

Your task is to extract fact check findings from search snippets. For each fact check found:

1. Identify the fact-checking organization (PolitiFact, Snopes, etc.)
2. Summarize the claim that was checked
3. Determine the verdict given

Verdict Categories (map to these):
- TRUE: Claim is accurate
- MOSTLY_TRUE: Claim is mostly accurate with minor issues
- HALF_TRUE: Claim is partly accurate, partly misleading
- MIXED: Contains both accurate and inaccurate elements
- MOSTLY_FALSE: Claim has some truth but is largely inaccurate
- FALSE: Claim is not accurate
- PANTS_ON_FIRE: Claim is extremely false/ridiculous (PolitiFact term)
- MISLEADING: Technically accurate but missing context
- UNPROVEN: Insufficient evidence to verify
- NOT_RATED: Mentioned but no clear verdict given

Count as "failed" fact checks: FALSE, MOSTLY_FALSE, PANTS_ON_FIRE, MISLEADING

Be conservative - only extract fact checks that are clearly about the media outlet or its reporting.
If a snippet is ambiguous or doesn't clearly contain a fact check verdict, skip it.
If the search results are about fact-checking an outlet's OVERALL reliability rating, that's relevant.
If results are about fact checks OF specific claims MADE BY the outlet, those are also relevant."""

    # Appended to SYSTEM_PROMPT only when claim-source attribution is enabled.
    # Forces the model to attribute WHO published each checked claim, so failures of
    # claims merely *about* the outlet (hoaxes, impersonation) are not counted as the
    # outlet's own failures.
    ATTRIBUTION_ADDENDUM = """

CLAIM-SOURCE ATTRIBUTION (required for every finding):
For each fact check, decide who originally PUBLISHED the checked claim, and set:
- claim_source:
    - "published_by_outlet": the outlet itself authored/published the claim (it is the publisher).
    - "about_outlet": the claim is ABOUT the outlet, IMPERSONATES it, or is FALSELY ATTRIBUTED
      to it (e.g. a fake post, hoax screenshot, or "a claim circulating that {outlet} said X"),
      but the outlet did NOT publish it.
    - "third_party": the claim merely mentions the outlet (e.g. the outlet is cited as a source
      while a different speaker's claim is being checked).
    - "unknown": you cannot tell from the evidence.
- published_by_outlet: true ONLY when claim_source == "published_by_outlet".
- claim_source_domain: the domain that actually published the claim, if identifiable (else null).
- attribution_confidence: 0.0-1.0.

Be strict: a fact check counts as the outlet's FAILURE only when published_by_outlet is true.
When in doubt, prefer "unknown" over "published_by_outlet". Example: a viral post falsely
attributed to the BBC that PolitiFact rates FALSE is "about_outlet" (published_by_outlet=false),
NOT a BBC failure.

When counting failed_count, count ONLY findings with published_by_outlet == true AND a failing
verdict (FALSE, MOSTLY_FALSE, PANTS_ON_FIRE, MISLEADING)."""

    def __init__(
        self,
        model: str = "gpt-5-mini-2025-08-07",
        temperature: float = 0.0,
        sites: list[str] | None = None,
        search_backend=None,
        attribute_claim_source: bool | None = None,
    ):
        """
        Initialize the FactCheckSearcher.

        Args:
            model: OpenAI model to use
            temperature: LLM temperature (0 for deterministic)
            sites: List of fact-checker sites to search (default: FACTCHECK_SITES)
            search_backend: Optional SearchBackend instance (defaults to DDGSearchBackend)
            attribute_claim_source: When True, classify who published each checked claim
                and count a failure only when the outlet itself published it. Defaults to
                config.FACTCHECK_ATTRIBUTE_CLAIM_SOURCE (off) when None.
        """
        self.llm = get_llm(model, temperature).with_structured_output(FactCheckLLMOutput)
        if search_backend is not None:
            self.search_backend = search_backend
        else:
            from search_backends import DDGSearchBackend
            self.search_backend = DDGSearchBackend()
        self.sites = sites or FACTCHECK_SITES.copy()
        self.attribute_claim_source = (
            FACTCHECK_ATTRIBUTE_CLAIM_SOURCE
            if attribute_claim_source is None
            else attribute_claim_source
        )

    def _extract_domain(self, url: str) -> str:
        """Extract the root domain from a URL."""
        parsed = urlparse(url if url.startswith("http") else f"https://{url}")
        domain = parsed.netloc or parsed.path
        domain = re.sub(r"^www\.", "", domain)
        domain = domain.split("/")[0]
        return domain.lower()

    def _extract_outlet_name(self, domain: str) -> str:
        """
        Extract a human-readable outlet name from domain.

        Args:
            domain: The domain (e.g., "nytimes.com")

        Returns:
            Outlet name (e.g., "New York Times")
        """
        # Common mappings
        known_names = {
            "nytimes.com": "New York Times",
            "washingtonpost.com": "Washington Post",
            "wsj.com": "Wall Street Journal",
            "bbc.com": "BBC",
            "cnn.com": "CNN",
            "foxnews.com": "Fox News",
            "msnbc.com": "MSNBC",
            "infowars.com": "InfoWars",
            "breitbart.com": "Breitbart",
            "dailywire.com": "Daily Wire",
            "theguardian.com": "The Guardian",
            "reuters.com": "Reuters",
            "apnews.com": "Associated Press",
        }

        if domain in known_names:
            return known_names[domain]

        # Generate from domain
        name = domain.split(".")[0]
        return name.replace("-", " ").replace("_", " ").title()

    def _search_fact_checks(self, domain: str, outlet_name: str) -> str:
        """
        Search all fact-checker sites for fact checks about the outlet.

        Args:
            domain: The domain to search for
            outlet_name: Human-readable outlet name

        Returns:
            Combined search snippets from all sites
        """
        all_snippets = []

        for site in self.sites:
            # Query format: site:politifact.com "nytimes.com" OR "New York Times"
            query = f'site:{site} "{domain}" OR "{outlet_name}"'

            try:
                results = self.search_backend.search(query, max_results=3)

                for r in results:
                    title = r.get("title", "")
                    body = r.get("body", "")
                    url = r.get("url", "")
                    snippet = f"[{site}] {title}: {body}"
                    if url:
                        snippet += f" (URL: {url})"
                    all_snippets.append(snippet)

            except Exception as e:
                logger.warning(f"Fact check search failed for {site}: {e}")
                continue

        return "\n\n".join(all_snippets) if all_snippets else ""

    def _parse_with_llm(self, domain: str, outlet_name: str, snippets: str) -> FactCheckLLMOutput:
        """
        Use LLM to parse fact check findings from search snippets.

        Args:
            domain: The domain being analyzed
            outlet_name: Human-readable outlet name
            snippets: Combined search snippets

        Returns:
            FactCheckLLMOutput with findings and counts
        """
        user_prompt = f"""Analyze fact check search results for: {outlet_name} ({domain})

SEARCH RESULTS:
{snippets}

Extract all fact check findings related to this outlet. Count how many have negative verdicts
(FALSE, MOSTLY_FALSE, PANTS_ON_FIRE, MISLEADING)."""

        system_prompt = self.SYSTEM_PROMPT
        if self.attribute_claim_source:
            system_prompt += self.ATTRIBUTION_ADDENDUM

        try:
            result: FactCheckLLMOutput = self.llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return result

        except Exception as e:
            logger.error(f"FactCheckSearcher LLM call failed: {e}")
            return FactCheckLLMOutput(
                findings=[],
                failed_count=0,
                total_count=0,
                confidence=0.0,
                reasoning=f"LLM parsing failed: {str(e)}",
            )

    def _calculate_score(self, failed_count: int, total_count: int) -> tuple[float, bool]:
        """Calculate MBFC-style fact check score (0=excellent, 10=poor).

        Per MBFC 2025 methodology, the score maps linearly to failed fact checks:
          0 failed = 0, 1 failed = 1, 2 failed = 2, ..., 10+ failed = 10

        Returns:
            Tuple of (score, coverage_sufficient).
            coverage_sufficient is False when no fact-check evidence was found
            at all — meaning the score is a default, not an observed measurement.
        """
        # If we searched approved sources and found checks with NONE failed → excellent.
        if total_count > 0 and failed_count == 0:
            return 0.0, True

        # If we found NO fact-check evidence at all, flag insufficient coverage.
        # "No fact checks found" ≠ "0 failed fact checks". Return an uncertain
        # score (3.0) instead of 0.0 to avoid systematically underrating outlets
        # that simply weren't checked by IFCN fact-checkers.
        if total_count == 0:
            return 3.0, False

        # Linear 1:1 mapping, capped at 10
        return min(float(failed_count), 10.0), True

    def analyze(self, url_or_domain: str, outlet_name: str | None = None) -> FactCheckAnalysisResult:
        """
        Search for and analyze fact checks about a media outlet.

        Args:
            url_or_domain: URL or domain to analyze
            outlet_name: Optional human-readable outlet name (auto-generated if not provided)

        Returns:
            FactCheckAnalysisResult with findings and score
        """
        domain = self._extract_domain(url_or_domain)
        outlet_name = outlet_name or self._extract_outlet_name(domain)

        # Search all fact-checker sites
        snippets = self._search_fact_checks(domain, outlet_name)

        if not snippets:
            # No search results at all — insufficient coverage, not "0 failures"
            score, coverage = self._calculate_score(0, 0)
            return FactCheckAnalysisResult(
                domain=domain,
                outlet_name=outlet_name,
                failed_checks_count=0,
                total_checks_count=0,
                score=score,
                source=FactCheckSource.FALLBACK,
                findings=[],
                search_snippets=None,
                confidence=0.0,
                reasoning="No fact check results found for this outlet",
                coverage_sufficient=coverage,
            )

        # Parse with LLM
        llm_output = self._parse_with_llm(domain, outlet_name, snippets)

        failed_count = llm_output.failed_count
        about_outlet_count = 0
        if self.attribute_claim_source:
            # Count a failure ONLY when the outlet itself published the checked claim.
            # Claims about / falsely attributed to the outlet are surfaced separately
            # and excluded from the failure count and score.
            failed_count = sum(
                1
                for f in llm_output.findings
                if f.published_by_outlet and f.verdict in FAILED_VERDICTS
            )
            about_outlet_count = sum(
                1 for f in llm_output.findings if f.claim_source == ClaimSource.ABOUT_OUTLET
            )

        # Calculate score
        score, coverage = self._calculate_score(failed_count, llm_output.total_count)

        return FactCheckAnalysisResult(
            domain=domain,
            outlet_name=outlet_name,
            failed_checks_count=failed_count,
            total_checks_count=llm_output.total_count,
            score=score,
            source=FactCheckSource.SEARCH,
            findings=llm_output.findings,
            search_snippets=snippets[:1000] if snippets else None,
            confidence=llm_output.confidence,
            reasoning=llm_output.reasoning,
            coverage_sufficient=coverage,
            about_outlet_count=about_outlet_count,
        )


# =============================================================================
# SourcingAnalyzer
# =============================================================================


class SourcingAnalyzer:
    """
    Analyzes sourcing quality by examining BOTH cited links AND textual attributions.
    """

    SYSTEM_PROMPT = """You are an expert at evaluating news source quality and attribution standards.

Your task is to analyze news articles to determine how well they source their claims.
You must look for two things:
1. **Hyperlinks**: Domains linked directly in the text.
2. **Textual Citations**: Explicit mentions of sources (e.g., "According to The New York Times", "A study by Harvard University").

### QUALITY TIERS (Assess identified sources):
- **PRIMARY**: Official docs, court filings, direct research studies, government data (.gov).
- **WIRE_SERVICE**: Reuters, AP, AFP, UPI.
- **MAJOR_OUTLET**: Established legacy media (NYT, BBC, WSJ, WaPo).
- **CREDIBLE**: Regional papers, specialized trade journals.
- **QUESTIONABLE**: State propaganda, conspiracy sites, tabloids, unverified blogs.

### VAGUE SOURCING (The "Weasel Words" Check):
You must also detect **Vague Sourcing** or "Anonymous Authority".
- BAD: "Critics say...", "Experts agree...", "British scientists claim...", "Sources close to the matter..." (without explaining why they are anonymous).
- BAD: "Many people are saying...", "It is reported that..." (Passive voice without agent).

### INSTRUCTIONS:
1. Analyze the provided source links and article snippets.
2. Extract named sources found in the text that weren't hyperlinked.
3. Identify if the text relies heavily on vague/weasel sourcing.
4. Provide a final sourcing score (0=Excellent/High Transparency, 10=Poor/No Sourcing).
"""

    def __init__(
        self,
        model: str = "gpt-5-mini-2025-08-07",
        temperature: float = 0.0,
    ):
        self.llm = get_llm(model, temperature).with_structured_output(SourcingLLMOutput)

    def _extract_links(self, text: str) -> list[str]:
        """Extract all URLs from article text."""
        url_pattern = r'https?://[^\s<>"\')\]]+[^\s<>"\')\].,;:!?]'
        return re.findall(url_pattern, text)

    def _extract_domains(self, urls: list[str]) -> list[str]:
        """Extract unique domains from URLs."""
        excluded_domains = {
            "twitter.com", "x.com", "facebook.com", "instagram.com",
            "youtube.com", "tiktok.com", "linkedin.com", "reddit.com",
            "t.co", "google.com"
        }
        domains = set()
        for url in urls:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                domain = re.sub(r"^www\.", "", domain)
                if domain and domain not in excluded_domains:
                    domains.add(domain)
            except Exception:
                continue
        return list(domains)

    def analyze(self, articles: list[dict[str, str]]) -> SourcingAnalysisResult:
        """
        Analyze sourcing quality using links and text analysis.
        """
        # 1. Gather Link Evidence
        all_links = []
        combined_text_snippets = []
        
        for i, article in enumerate(articles):
            text = article.get("text", "")
            if not text:
                continue
                
            # Extract links
            links = self._extract_links(text)
            all_links.extend(links)
            
            # Prepare text snippet for LLM (First 2000 chars is usually where sourcing happens)
            snippet = text[:2000].replace("\n", " ")
            combined_text_snippets.append(f"ARTICLE {i+1}: {snippet}")

        unique_domains = self._extract_domains(all_links)
        
        # 2. Prepare Prompt for LLM
        # We give the LLM the hard links we found, PLUS the text to find non-linked citations
        domains_str = ", ".join(unique_domains) if unique_domains else "None detected via regex"
        # Use all reviewed articles — MBFC methodology requires the full review set
        text_context = "\n\n".join(combined_text_snippets)

        user_prompt = f"""Analyze the sourcing in these articles.

DETECTED HYPERLINKS (already extracted):
{domains_str}

ARTICLE TEXT SNIPPETS (Look for textual citations and vague sourcing here):
{text_context}

1. Did you find valid named sources in the text that were NOT linked? (e.g. "According to the AP")
2. Is there frequent use of vague sourcing? (e.g. "Scientists say", "Critics claim")
3. Assess the quality of the specific sources found."""

        try:
            # 3. Invoke LLM
            llm_output: SourcingLLMOutput = self.llm.invoke(
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )

            # 4. Calculate Final Score Logic
            # Penalize for vague sourcing if quality score is otherwise good
            final_score = llm_output.overall_quality_score

            # If vague sourcing is found, bump score toward Mixed range but cap at 5.0
            if llm_output.vague_sourcing_detected and final_score < 5.0:
                final_score = min(final_score + 1.5, 5.0)

            # Calculate stats — use LLM-detected sources as fallback when regex finds none
            link_count = len(all_links)
            llm_source_count = len(llm_output.sources_assessed) if llm_output.sources_assessed else 0

            # MBFC hard floor: if zero hyperlinks AND zero named citations → score ≥ 5.0
            # "No sourcing" is at minimum a Mixed-level issue per methodology
            if link_count == 0 and llm_source_count == 0:
                final_score = max(5.0, final_score)

            # Cap score at 10 (Poor)
            final_score = min(10.0, final_score)
            effective_source_count = link_count if link_count > 0 else llm_source_count
            avg_sources = effective_source_count / len(articles) if articles else 0.0

            # Format reasoning to include vague sourcing info if present
            reasoning = llm_output.overall_assessment
            if llm_output.vague_sourcing_detected and llm_output.vague_sourcing_examples:
                examples = ", ".join(llm_output.vague_sourcing_examples[:2])
                reasoning += f" (Note: Detected vague sourcing: '{examples}')"

            return SourcingAnalysisResult(
                score=final_score,
                avg_sources_per_article=round(avg_sources, 2),
                total_sources_found=effective_source_count,
                unique_domains=len(unique_domains) if unique_domains else llm_source_count,
                has_hyperlinks=len(all_links) > 0,
                source_assessments=llm_output.sources_assessed,
                has_primary_sources=llm_output.has_primary_sources,
                has_wire_services=llm_output.has_wire_services,
                confidence=llm_output.confidence,
                reasoning=reasoning
            )

        except Exception as e:
            logger.error(f"SourcingAnalyzer LLM call failed: {e}")
            # Fallback ONLY on error
            return SourcingAnalysisResult(
                score=5.0,
                avg_sources_per_article=0.0,
                total_sources_found=len(all_links),
                unique_domains=len(unique_domains),
                has_hyperlinks=len(all_links) > 0,
                source_assessments=[],
                has_primary_sources=False,
                has_wire_services=False,
                confidence=0.0,
                reasoning=f"Analysis failed: {str(e)}"
            )
# =============================================================================
# EditorialBiasAnalyzer
# =============================================================================


class EditorialBiasAnalyzer:
    """
    Analyzes editorial/political bias using LLM content analysis.

    This analyzer replaces keyword matching and lexicon-based approaches with
    comprehensive LLM analysis of article content. It uses the MBFC methodology
    encoded directly in the system prompt.

    The analyzer evaluates:
    1. Policy positions across major domains (economic, social, environmental, etc.)
    2. Use of politically loaded language
    3. Story selection bias patterns
    4. Overall editorial slant

    Bias Scale:
    - -10 to -7: Extreme Left / Left
    - -7 to -3: Left-Center
    - -3 to +3: Center
    - +3 to +7: Right-Center
    - +7 to +10: Right / Extreme Right

    Attributes:
        llm: LangChain LLM with structured output for bias analysis
    """

    # Comprehensive system prompt encoding MBFC methodology
    SYSTEM_PROMPT = """You are an expert media analyst specializing in detecting editorial and political bias.
Your task is to analyze article content and determine the outlet's political leaning.

IMPORTANT: This analysis uses an AMERICAN political perspective. Left/Right designations
are based on the US political spectrum.

## BIAS SCALE
Use a scale from -10 (far left) to +10 (far right), with 0 being perfectly centrist:
- Extreme Left (-10 to -8): Proposes revolutionary change, overthrow of capitalism, or violent resistance. Ignores democratic processes.
- Left (-8 to -5): Strong progressive/socialist democrat stance. Advocates for major systemic change within democratic framework (e.g., Green New Deal, Universal Healthcare).
- Left-Center (-5 to -2): Standard liberal/democrat positions.
- Center (-2 to +2): Balanced coverage, minimal editorial slant, presents multiple viewpoints
- Right-Center (+2 to +5): Leans conservative but with some moderate positions
- Right (+5 to +8): Consistently favors conservative policies
- Extreme Right (+8 to +10): Advocates radical conservative/nationalist positions

## POLICY DOMAIN INDICATORS

### ECONOMIC POLICY
LEFT indicators:
- Supports income equality, higher taxes on wealthy
- Favors government spending on social programs
- Supports stronger business regulations
- Advocates minimum wage increases, wealth redistribution
- Pro-union, worker protections

RIGHT indicators:
- Supports lower taxes, less regulation
- Favors reduced government spending
- Prefers free-market solutions
- Opposes minimum wage mandates
- Pro-business, lower corporate taxes

### SOCIAL ISSUES
LEFT indicators:
- Supports abortion rights
- Favors LGBTQ+ rights and protections
- Advocates for diversity, equity, inclusion initiatives
- Supports criminal justice reform
- Favors gun control measures

RIGHT indicators:
- Opposes or seeks to restrict abortion
- Traditional marriage advocacy
- Opposes DEI initiatives
- Tough on crime positions
- Strong Second Amendment support

### ENVIRONMENTAL POLICY
LEFT indicators:
- Climate change is urgent, human-caused crisis
- Supports strong environmental regulations
- Favors renewable energy transition
- Supports international climate agreements

RIGHT indicators:
- Skepticism about climate urgency
- Prioritizes economic impact of regulations
- Supports fossil fuel industry
- Skeptical of international climate agreements

### HEALTHCARE
LEFT indicators:
- Supports universal healthcare
- Favors government involvement in healthcare
- Views healthcare as a right

RIGHT indicators:
- Prefers private healthcare solutions
- Opposes government-run healthcare
- Views healthcare as market service

### IMMIGRATION
LEFT indicators:
- Supports pathways to citizenship
- Opposes harsh enforcement measures
- Favors less restrictive immigration

RIGHT indicators:
- Emphasizes border security
- Opposes amnesty programs
- Favors more restrictive immigration

### GUN RIGHTS
LEFT indicators:
- Supports background checks, waiting periods
- Favors assault weapon restrictions
- Emphasizes gun violence prevention

RIGHT indicators:
- Strong Second Amendment advocacy
- Opposes gun restrictions
- Emphasizes self-defense rights

## LOADED LANGUAGE DETECTION
Identify politically loaded terms that reveal bias:

LEFT-LEANING loaded terms: "regime", "far-right", "extremist", "racist", "xenophobic",
"fascist", "climate denier", "voter suppression", "white supremacy", "wealth inequality"

RIGHT-LEANING loaded terms: "radical left", "socialist", "woke", "cancel culture",
"mainstream media", "fake news", "open borders", "defund police", "critical race theory"

## STORY SELECTION BIAS
Note if the outlet appears to:
- Selectively cover stories that favor one political side
- Ignore stories that would be unfavorable to their preferred side
- Frame neutral events with partisan spin

## MBFC 4-CATEGORY BIAS SCORING (REQUIRED)
You MUST provide separate scores for ALL FOUR categories below.
The final bias rating is computed as a weighted average:
  Bias = Economic(35%) + Social(35%) + News Reporting(15%) + Editorial(15%)

### Category 1: ECONOMIC SYSTEM SCORE (-10 to +10)
Score the outlet's economic ideology:
  -10: Communism (no corporatism, full government ownership)
  -7.5: Socialism (minimal corporatism, high regulation)
  -5: Democratic Socialism (reduced corporatism, strongly regulated capitalism)
  -2.5: Regulated Market Economy (moderate corporatism, balanced regulation)
  0: Centrism (balanced regulation and corporate influence)
  +2.5: Moderately Regulated Capitalism (slight lean toward corporatism)
  +5: Classical Liberalism (moderate to high corporatism, lower regulation)
  +7.5: Libertarianism (low government intervention, high corporate influence)
  +10: Radical Laissez-Faire Capitalism (minimal to no regulation)

### Category 2: SOCIAL PROGRESSIVE vs TRADITIONAL CONSERVATIVE SCORE (-10 to +10)
Score the outlet's stance on social issues (abortion, immigration, climate, LGBTQ+, equity):
  -10: Strong Progressive Liberalism
  -7.5: Progressive Liberalism
  -5: Moderate Progressive
  -2.5: Mild Progressive
  0: Balanced
  +2.5: Mild Conservative
  +5: Moderate Conservative
  +7.5: Traditional Conservative
  +10: Strong Traditional Conservative

### Category 3: STRAIGHT NEWS REPORTING BALANCE SCORE (-10 to +10)
Score how well the outlet reports all sides in its STRAIGHT NEWS stories (not opinion):
  -10: Exclusively promotes left-leaning perspectives
  -5: Often leans left but occasionally includes counterpoints
  0: Equally represents all perspectives
  +5: Often leans right but occasionally includes counterpoints
  +10: Exclusively promotes right-leaning perspectives

### Category 4: EDITORIAL/OP-ED BIAS SCORE (-10 to +10)
Score bias in opinion pieces, editorials, and use of loaded emotional language:
  -10: Editorials exclusively promote left views with highly manipulative language
  -5: Often leans left with some emotional framing
  0: Editorials present perspectives fairly, avoid loaded language
  +5: Often leans right with some emotional framing
  +10: Editorials exclusively promote right views with highly manipulative language

## NEWSLETTER / SINGLE-AUTHOR COMMENTARY SITES
For newsletter or single-author commentary outlets (e.g., Substack newsletters, independent blogs):
- ALL content is effectively opinion/editorial — there is no "straight news" to balance against
- Category 3 (news_reporting_score) should reflect the author's ideological framing, NOT be scored as 0 (balanced)
- Category 4 (editorial_bias_score) should carry extra weight since the entire outlet is editorial opinion
- These outlets often have strong, consistent ideological positions that should be scored accordingly
- Do NOT default to centrist scores just because the outlet is small or independent

## EXTREMISM DETECTION (CRITICAL)
If the outlet promotes ANY of the following, you MUST score it in the extreme range
(-8 to -10 or +8 to +10). Do NOT give moderate scores for fringe outlets:
- Conspiracy theories (QAnon, deep state, NWO, chemtrails, 9/11 truth, etc.)
- Pseudoscience (anti-vaccine, climate denial with no scientific basis, etc.)
- Propaganda for authoritarian regimes
- White supremacy, ethno-nationalism, or racial separatism
- Revolutionary communism or anarchism advocating violence
- Explicit calls for overthrow of democratic institutions

## STATE PROPAGANDA DIRECTION (CRITICAL - READ CAREFULLY)
Following MBFC methodology, state-controlled propaganda outlets from authoritarian regimes
(Russia, China, Iran, North Korea, Syria, Venezuela, Cuba, etc.) are classified as RIGHT-LEANING
(positive scores, typically +7 to +10), NOT left-leaning, even when their content contains:
- Anti-capitalist or anti-Western rhetoric
- Criticism of U.S. foreign policy or imperialism
- Advocacy for wealth redistribution or socialist economics
- Anti-NATO or anti-EU narratives

WHY: MBFC classifies these outlets based on their SERVICE TO AUTHORITARIAN POWER, not their
surface-level economic rhetoric. State propaganda serves nationalist, authoritarian interests
which align with the RIGHT on the MBFC scale. Anti-Western rhetoric from Russian or Chinese
state media is a geopolitical strategy, not a genuine progressive/leftist position.

Examples of RIGHT-scored state propaganda outlets per MBFC:
- RT (Russia Today): RIGHT — Russian state media
- New Eastern Outlook: RIGHT — linked to Russian intelligence (SVR)
- Sputnik News: RIGHT — Russian state media
- CGTN / China Daily: RIGHT — Chinese state media
- PressTV: RIGHT — Iranian state media
- TeleSUR: RIGHT — Venezuelan state media

If you detect that an outlet is state-controlled or state-linked from an authoritarian regime,
score ALL FOUR bias categories as POSITIVE (right-leaning), regardless of anti-capitalist content.

## ANALYSIS INSTRUCTIONS
1. Analyze the actual CONTENT, not the outlet's reputation
2. Look for patterns across multiple articles if provided
3. Identify specific policy positions expressed
4. Note use of loaded/emotional language
5. Consider both explicit statements and implicit framing
6. Do NOT default to moderate scores when content clearly shows extreme ideology
7. Distinguish between NEWS reporting and OPINION pieces

## CRITICAL: ALWAYS POPULATE POLICY POSITIONS
You MUST extract policy positions even when the outlet takes a neutral/balanced stance:
- If an article discusses climate change factually without taking sides, output a PolicyPosition with domain="Environmental Policy", leaning="Center", and indicators like "Reports factually on climate science without advocacy."
- If an article covers immigration policy neutrally, output a PolicyPosition with domain="Immigration", leaning="Center", and indicators describing the balanced framing.
- Do NOT return an empty list for policy_positions unless the articles are entirely devoid of political or social topics (e.g., only sports scores or recipes).
- For each major topic covered in the articles, create a PolicyPosition documenting the outlet's stance (which may be Center/balanced).

## CRITICAL: CITE SOURCE ARTICLES
For each PolicyPosition, you MUST populate the 'source_articles' field with the article identifiers (e.g., "Article 1: [title]") where the evidence was found.
- Every indicator MUST be traceable to a specific article.
- Use the article number and title as shown in the input (e.g., "Article 3: Russia warns of consequences after NATO expansion").

## CRITICAL: BE EXPLICIT AND DIRECT
Do NOT use vague descriptions. State the outlet's positions directly:
- BAD: "The outlet appears to have some progressive leanings on social issues."
- GOOD: "The outlet advocates for expanded LGBTQ+ protections and frames abortion restrictions as threats to women's rights (Article 2: 'Supreme Court ruling threatens reproductive freedom')."
- BAD: "Economic coverage tends to favor certain perspectives."
- GOOD: "The outlet supports higher taxes on the wealthy, frames income inequality as a crisis, and advocates for stronger union protections (Article 5: 'Workers deserve a living wage, unions say')."
When the outlet is centrist, still be explicit:
- GOOD: "The outlet presents both pro-business and pro-labor arguments without taking sides, quoting economists from both perspectives (Article 1: 'Fed rate decision splits analysts')."

## PRO-SCIENCE DETECTION
If the outlet is a peer-reviewed scientific journal, academic publisher, or evidence-based scientific publication:
- Set is_pro_science=True
- These outlets publish research-based content that respects scientific consensus
- Examples: Nature, Science, The Lancet, Frontiers journals, PLOS, BMJ
- Do NOT set is_pro_science=True for general news outlets that merely report on science\""""

    def __init__(
        self,
        model: str = "gpt-5-mini-2025-08-07",
        temperature: float = 0.0,
    ):
        """
        Initialize the EditorialBiasAnalyzer.

        Args:
            model: OpenAI model to use
            temperature: LLM temperature (0 for deterministic)
        """
        self.llm = get_llm(model, temperature).with_structured_output(EditorialBiasLLMOutput)

    def _extract_domain(self, url: str) -> str:
        """Extract the root domain from a URL."""
        parsed = urlparse(url if url.startswith("http") else f"https://{url}")
        domain = parsed.netloc or parsed.path
        domain = re.sub(r"^www\.", "", domain)
        domain = domain.split("/")[0]
        return domain.lower()

    def _score_to_label(self, score: float) -> str:
        """Convert numeric score to MBFC-style label based on MBFC 2025 methodology.

        Per MBFC: Far Left at -7.0+, Far Right at +7.0+
        """
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

    def _analyze_with_llm(self, articles: list[dict[str, str]]) -> EditorialBiasLLMOutput:
        """
        Use LLM to analyze editorial bias in articles.

        Args:
            articles: List of article dicts with 'title' and 'text' keys

        Returns:
            EditorialBiasLLMOutput with bias assessment
        """
        # Format articles for analysis, including type label when available
        articles_text = []
        for i, article in enumerate(articles, 1):
            title = article.get("title", "Untitled")
            url = article.get("url", "")
            text = article.get("text", "")[:2000]  # Limit text length
            url_line = f"\nURL: {url}" if url else ""
            type_label = article.get("_type_label", "")
            type_line = f"\nType: {type_label}" if type_label else ""
            articles_text.append(f"ARTICLE {i}:{type_line}\nTitle: {title}{url_line}\nText: {text}\n")

        combined_text = "\n---\n".join(articles_text)

        # Check if articles have type labels (pre-classified by OpinionAnalyzer)
        has_type_labels = any(a.get("_type_label") for a in articles)
        type_instruction = ""
        if has_type_labels:
            type_instruction = (
                "\nIMPORTANT: Articles are labeled as STRAIGHT NEWS or OPINION/EDITORIAL. "
                "Score news_reporting_score (Category 3) based ONLY on articles labeled STRAIGHT NEWS. "
                "Score editorial_bias_score (Category 4) based ONLY on articles labeled OPINION/EDITORIAL. "
                "Score economic_score and social_score on ALL articles regardless of type.\n"
            )

        user_prompt = f"""Analyze the following articles for editorial/political bias.
IMPORTANT: If the articles are not in English, translate their core meaning to English internally before analyzing.
IMPORTANT: For each policy position, cite the specific article(s) where you found the evidence using the format "Article N: [Title]".
{type_instruction}
{combined_text}

You MUST provide ALL of the following:
1. economic_score (-10 to +10): Score the outlet's economic ideology
2. social_score (-10 to +10): Score the outlet's social progressive vs conservative stance
3. news_reporting_score (-10 to +10): Score how balanced the straight news reporting is
4. editorial_bias_score (-10 to +10): Score editorial/op-ed bias and loaded language
5. bias_score: Your best overall combined estimate (-10 to +10)
6. Positions on specific policy domains if detectable — cite source articles for each
7. Use of loaded language (with examples and source article references)
8. Any story selection bias patterns"""

        try:
            result: EditorialBiasLLMOutput = self.llm.invoke(
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return result

        except Exception as e:
            logger.error(f"EditorialBiasAnalyzer LLM call failed: {e}")
            return EditorialBiasLLMOutput(
                overall_bias=BiasDirection.CENTER,
                bias_score=0.0,
                economic_score=0.0,
                social_score=0.0,
                news_reporting_score=0.0,
                editorial_bias_score=0.0,
                policy_positions=[],
                uses_loaded_language=False,
                loaded_language_examples=[],
                story_selection_bias=None,
                confidence=0.0,
                reasoning=f"LLM analysis failed: {str(e)}",
            )

    def analyze(
        self,
        articles: list[dict[str, str]],
        url_or_domain: str | None = None,
        outlet_name: str | None = None,
        news_articles: list[dict[str, str]] | None = None,
        opinion_articles: list[dict[str, str]] | None = None,
        about_page_text: str | None = None,
    ) -> EditorialBiasResult:
        """
        Analyze editorial bias in articles from a media outlet.

        Per MBFC methodology, straight-news balance and editorial bias should
        be scored on different article subsets. When news_articles and
        opinion_articles are provided (pre-classified by OpinionAnalyzer),
        the LLM prompt labels them so it can score categories 3 and 4
        on the correct content. Falls back to the full article list if
        pre-classified subsets are not provided.

        Args:
            articles: Full list of article dicts (used for economic/social scoring)
            url_or_domain: Optional URL or domain for context
            outlet_name: Optional human-readable outlet name
            news_articles: Pre-classified straight-news articles (for category 3)
            opinion_articles: Pre-classified opinion/editorial articles (for category 4)
            about_page_text: Optional about page text for fallback when no articles available

        Returns:
            EditorialBiasResult with comprehensive bias analysis
        """
        domain = self._extract_domain(url_or_domain) if url_or_domain else "unknown"

        if not articles:
            # If we have about page text, use it as fallback evidence
            if about_page_text and len(about_page_text.strip()) > 50:
                logger.info(f"No articles for {outlet_name}, using about page text as fallback evidence")
                articles = [{"title": f"About Page - {outlet_name or domain}", "text": about_page_text[:3000]}]
                # Continue to LLM analysis below with reduced confidence
            else:
                return EditorialBiasResult(
                    domain=domain,
                    outlet_name=outlet_name,
                    overall_bias=BiasDirection.CENTER,
                    bias_score=0.0,
                    mbfc_label="Least Biased",
                    policy_positions=[],
                    uses_loaded_language=False,
                    loaded_language_examples=[],
                    story_selection_bias=None,
                    articles_analyzed=0,
                    confidence=0.0,
                    reasoning="No articles provided for analysis",
                )

        # Build article input with news/opinion classification labels
        if news_articles is not None or opinion_articles is not None:
            # Pre-classified: label each subset so the LLM scores category 3
            # (news_reporting_score) on news articles and category 4
            # (editorial_bias_score) on opinion/editorial articles.
            labeled_articles = []
            for a in (news_articles or []):
                labeled = dict(a)
                labeled["_type_label"] = "STRAIGHT NEWS"
                labeled_articles.append(labeled)
            for a in (opinion_articles or []):
                labeled = dict(a)
                labeled["_type_label"] = "OPINION/EDITORIAL"
                labeled_articles.append(labeled)
            # If neither subset has articles, fall back to all articles
            analysis_input = labeled_articles if labeled_articles else articles
        else:
            analysis_input = articles

        # Analyze with LLM
        llm_output = self._analyze_with_llm(analysis_input)

        # Compute MBFC weighted bias score from 4 subcategories
        # Weights: Economic 35%, Social 35%, News Reporting 15%, Editorial 15%
        weighted_bias_score = (
            llm_output.economic_score * 0.35
            + llm_output.social_score * 0.35
            + llm_output.news_reporting_score * 0.15
            + llm_output.editorial_bias_score * 0.15
        )
        # Clamp to [-10, +10]
        weighted_bias_score = max(-10.0, min(10.0, weighted_bias_score))

        # Step 1b: Power-law score amplification to combat centrality bias.
        # LLMs systematically underestimate extreme scores, giving -5 to +5
        # when gold expects -10 to +10. Apply power-law spreading to stretch
        # scores away from center while preserving direction and keeping
        # moderate scores relatively stable.
        if llm_output.confidence >= 0.5:
            sign = 1.0 if weighted_bias_score >= 0 else -1.0
            magnitude = abs(weighted_bias_score)
            if magnitude > 0:
                # Power-law: f(x) = 10 * (x/10)^0.8
                # This maps: 1→1.6, 3→4.0, 5→5.7, 7→7.3, 9→9.2
                stretched = 10.0 * (magnitude / 10.0) ** 0.8
                weighted_bias_score = sign * min(stretched, 10.0)

        # Convert weighted score to MBFC label
        mbfc_label = self._score_to_label(weighted_bias_score)

        return EditorialBiasResult(
            domain=domain,
            outlet_name=outlet_name,
            overall_bias=llm_output.overall_bias,
            bias_score=weighted_bias_score,
            economic_score=llm_output.economic_score,
            social_score=llm_output.social_score,
            news_reporting_score=llm_output.news_reporting_score,
            editorial_bias_score=llm_output.editorial_bias_score,
            mbfc_label=mbfc_label,
            policy_positions=llm_output.policy_positions,
            uses_loaded_language=llm_output.uses_loaded_language,
            loaded_language_examples=llm_output.loaded_language_examples,
            story_selection_bias=llm_output.story_selection_bias,
            ideology_summary=llm_output.ideology_summary,
            economy_summary=llm_output.economy_summary,
            is_pro_science=llm_output.is_pro_science,
            articles_analyzed=len(articles),
            confidence=llm_output.confidence,
            reasoning=llm_output.reasoning,
        )


# =============================================================================
# PseudoscienceAnalyzer
# =============================================================================


class PseudoscienceAnalyzer:
    """
    Detects pseudoscience and conspiracy content using LLM analysis.

    This analyzer replaces dictionary/keyword matching with comprehensive
    LLM-based content analysis. It identifies pseudoscientific claims and
    assesses how the outlet treats scientific consensus.

    Categories detected:
    - Health pseudoscience (anti-vax, alternative medicine, etc.)
    - Climate/environmental misinformation
    - Paranormal/supernatural claims
    - Conspiracy theories
    - Other pseudoscience

    Severity levels:
    - PROMOTES: Actively promotes pseudoscience as fact
    - PRESENTS_UNCRITICALLY: Reports without proper scientific context
    - MIXED: Sometimes promotes, sometimes critical
    - NONE_DETECTED: No pseudoscience found

    Attributes:
        llm: LangChain LLM with structured output for pseudoscience detection
    """

    SYSTEM_PROMPT = """You are an expert science communicator and fact-checker specializing in
identifying pseudoscience, conspiracy theories, and science misinformation.

## DEFINITION
Pseudoscience: Claims, beliefs, or practices that are presented as scientific but are
incompatible with the scientific method - they are unproven, not testable, or contradict
the scientific consensus.

## CATEGORIES TO DETECT

### HEALTH-RELATED PSEUDOSCIENCE
- Anti-Vaccination: Claims vaccines cause autism, are dangerous, contain microchips, etc.
  SCIENTIFIC CONSENSUS: Vaccines are safe, effective, and do not cause autism.

- Alternative Medicine promoted as cure: Homeopathy, naturopathy, crystal healing presented
  as effective medical treatments.
  SCIENTIFIC CONSENSUS: No evidence these treatments work beyond placebo.

- Alternative Cancer Treatments: Claims that essential oils, supplements, or alternative
  therapies can cure cancer instead of conventional treatment.
  SCIENTIFIC CONSENSUS: Only proven treatments (surgery, chemo, radiation, immunotherapy) are effective.

- COVID-19 Misinformation: False claims about vaccines, treatments (ivermectin, hydroxychloroquine),
  origins, or prevention methods.
  SCIENTIFIC CONSENSUS: COVID vaccines are safe and effective; unproven treatments are not substitutes.

- Detoxification Claims: Claims that special diets, supplements, or procedures remove "toxins."
  SCIENTIFIC CONSENSUS: The liver and kidneys naturally detoxify; "detox" products have no proven benefit.

### CLIMATE/ENVIRONMENTAL
- Climate Change Denialism: Denying human-caused climate change, claiming it's a hoax,
  or minimizing its urgency.
  SCIENTIFIC CONSENSUS: Climate change is real, human-caused, and requires urgent action.

- 5G Health Conspiracy: Claims that 5G causes COVID, cancer, or other health problems.
  SCIENTIFIC CONSENSUS: 5G radio waves are non-ionizing and not harmful at normal exposure levels.

- Chemtrails: Claims that aircraft condensation trails are chemical/biological agents.
  SCIENTIFIC CONSENSUS: Contrails are simply water vapor; no evidence of deliberate spraying.

- GMO Dangers: Claims that GMOs are inherently dangerous or cause health problems.
  SCIENTIFIC CONSENSUS: GMOs are extensively tested and safe for consumption.

### PARANORMAL/SUPERNATURAL
- Astrology: Claims that celestial bodies influence personality or predict events.
  SCIENTIFIC CONSENSUS: No mechanism or evidence for astrological effects.

- Psychic Claims: Claims of telepathy, clairvoyance, or communication with the dead.
  SCIENTIFIC CONSENSUS: No evidence for psychic phenomena despite extensive testing.

- Faith Healing: Claims that prayer or spiritual intervention can cure disease.
  SCIENTIFIC CONSENSUS: No evidence faith healing works; can be dangerous if it replaces medicine.

### CONSPIRACY THEORIES
- Flat Earth: Claims the Earth is flat and space agencies are lying.
  SCIENTIFIC CONSENSUS: The Earth is an oblate spheroid; this is confirmed by countless observations.

- Moon Landing Hoax: Claims the Apollo missions were faked.
  SCIENTIFIC CONSENSUS: Moon landings are among the most well-documented events in history.

- QAnon: Claims about secret cabals, child trafficking rings run by elites, etc.
  SCIENTIFIC CONSENSUS: These are unfounded conspiracy theories with no evidence.

## SEVERITY ASSESSMENT

PROMOTES: The outlet actively promotes pseudoscience as fact or truth
- Presents claims without skepticism
- Attacks scientific consensus
- Promotes practitioners/products
- Uses persuasive language to convince readers

PRESENTS_UNCRITICALLY: Reports on pseudoscience without proper context
- Gives "both sides" treatment to science vs. pseudoscience
- Fails to note scientific consensus
- Presents fringe views as legitimate alternatives

MIXED: Sometimes promotes, sometimes critical
- Inconsistent treatment of pseudoscience
- Some articles critical, others not

NONE_DETECTED: No pseudoscience content found
- Content respects scientific consensus
- Properly contextualizes scientific uncertainty
- Does not promote unproven claims

## ANALYSIS INSTRUCTIONS
1. Identify specific pseudoscientific claims in the content
2. Note how the outlet frames these claims (promoting vs. debunking)
3. Check if scientific consensus is mentioned or ignored
4. Assess overall pattern across multiple articles if available
5. Be precise - distinguish between reporting ON pseudoscience (journalism) vs. PROMOTING it
6. Quote specific evidence when identifying pseudoscience content"""

    def __init__(
        self,
        model: str = "gpt-5-mini-2025-08-07",
        temperature: float = 0.0,
    ):
        """
        Initialize the PseudoscienceAnalyzer.

        Args:
            model: OpenAI model to use
            temperature: LLM temperature (0 for deterministic)
        """
        self.llm = get_llm(model, temperature).with_structured_output(PseudoscienceLLMOutput)

    def _extract_domain(self, url: str) -> str:
        """Extract the root domain from a URL."""
        parsed = urlparse(url if url.startswith("http") else f"https://{url}")
        domain = parsed.netloc or parsed.path
        domain = re.sub(r"^www\.", "", domain)
        domain = domain.split("/")[0]
        return domain.lower()

    def _analyze_with_llm(self, articles: list[dict[str, str]]) -> PseudoscienceLLMOutput:
        """
        Use LLM to detect pseudoscience in articles.

        Args:
            articles: List of article dicts with 'title' and 'text' keys

        Returns:
            PseudoscienceLLMOutput with pseudoscience assessment
        """
        # Format articles for analysis
        articles_text = []
        for i, article in enumerate(articles, 1):
            title = article.get("title", "Untitled")
            text = article.get("text", "")[:2000]  # Limit text length
            articles_text.append(f"ARTICLE {i}:\nTitle: {title}\nText: {text}\n")

        combined_text = "\n---\n".join(articles_text)

        user_prompt = f"""Analyze the following articles for pseudoscience and conspiracy content:

{combined_text}

Identify:
1. Any pseudoscientific claims or conspiracy theories
2. How the outlet treats these claims (promoting vs. debunking)
3. Whether scientific consensus is respected
4. Overall quality of science reporting (0=excellent, 10=promotes pseudoscience)"""

        try:
            result: PseudoscienceLLMOutput = self.llm.invoke(
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return result

        except Exception as e:
            logger.error(f"PseudoscienceAnalyzer LLM call failed: {e}")
            return PseudoscienceLLMOutput(
                indicators=[],
                promotes_pseudoscience=False,
                overall_severity=PseudoscienceSeverity.NONE_DETECTED,
                science_reporting_quality=5.0,
                respects_scientific_consensus=True,
                confidence=0.0,
                reasoning=f"LLM analysis failed: {str(e)}",
            )

    def analyze(
        self,
        articles: list[dict[str, str]],
        url_or_domain: str | None = None,
        outlet_name: str | None = None,
    ) -> PseudoscienceAnalysisResult:
        """
        Analyze articles for pseudoscience and conspiracy content.

        Args:
            articles: List of article dicts with 'title' and 'text' keys
            url_or_domain: Optional URL or domain for context
            outlet_name: Optional human-readable outlet name

        Returns:
            PseudoscienceAnalysisResult with comprehensive analysis
        """
        domain = self._extract_domain(url_or_domain) if url_or_domain else "unknown"

        if not articles:
            return PseudoscienceAnalysisResult(
                domain=domain,
                outlet_name=outlet_name,
                score=5.0,  # Neutral when no data
                promotes_pseudoscience=False,
                overall_severity=PseudoscienceSeverity.NONE_DETECTED,
                categories_found=[],
                indicators=[],
                respects_scientific_consensus=True,
                articles_analyzed=0,
                confidence=0.0,
                reasoning="No articles provided for analysis",
            )

        # Analyze with LLM
        llm_output = self._analyze_with_llm(articles)

        # Extract unique categories found
        categories_found = list(set(
            indicator.category for indicator in llm_output.indicators
        ))

        return PseudoscienceAnalysisResult(
            domain=domain,
            outlet_name=outlet_name,
            score=llm_output.science_reporting_quality,
            promotes_pseudoscience=llm_output.promotes_pseudoscience,
            overall_severity=llm_output.overall_severity,
            categories_found=categories_found,
            indicators=llm_output.indicators,
            respects_scientific_consensus=llm_output.respects_scientific_consensus,
            articles_analyzed=len(articles),
            confidence=llm_output.confidence,
            reasoning=llm_output.reasoning,
        )


# =============================================================================
# TransparencyAnalyzer (MBFC Factuality: 25% weight)
# =============================================================================

class TransparencyAnalyzer:
    """Evaluates outlet transparency per MBFC methodology.

    Assesses 5 elements: about page, ownership disclosure, funding disclosure,
    author identification, and location disclosure. Score 0-10 where 0 is
    fully transparent and 10 is no transparency at all.
    """

    SYSTEM_PROMPT = """You are an expert media analyst evaluating the transparency of a news outlet.

Transparency measures how much critical information a source discloses about itself.
You MUST evaluate these 5 elements:

1. ABOUT PAGE: Does the outlet have a clear, detailed About page explaining its mission and purpose?
2. OWNERSHIP: Does the outlet disclose who owns it (person, company, or organization)?
3. FUNDING: Does the outlet disclose its funding sources or revenue model (advertising, subscription, donations, etc.)?
4. AUTHORS: Do articles identify authors by name (not just "Staff" or "Admin")?
5. LOCATION: Does the outlet disclose its physical location or headquarters?

SCORING (0-10, lower is better):
  0: Fully transparent — all 5 elements clearly disclosed
  1: Near full transparency — minor gaps in one element
  2: Mostly transparent — 4 of 5 elements disclosed
  3-4: Noticeable gaps — 3 of 5 elements disclosed
  5: Partially transparent — only 2 of 5 elements disclosed
  6-7: Limited transparency — only 1-2 elements with unclear details
  8-9: Very limited — vague or unclear on almost all elements
  10: No transparency — no information about ownership, funding, authors, location, or purpose

Base your assessment on the provided evidence (about page text, Wikipedia, search results, articles).
If Wikipedia is available, use it to fill in gaps about ownership, location, and funding.
If no evidence of transparency is found (no about page, no ownership disclosure, no author identification), score 7-8 (limited transparency). Do NOT default to moderate scores for opaque outlets."""

    def __init__(
        self,
        model: str = "gpt-5-mini-2025-08-07",
        temperature: float = 0.0,
        search_backend=None,
    ):
        self.llm = get_llm(model, temperature).with_structured_output(TransparencyLLMOutput)
        if search_backend is not None:
            self.search_backend = search_backend
        else:
            from search_backends import DDGSearchBackend
            self.search_backend = DDGSearchBackend()

    _WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

    def _extract_domain(self, url: str) -> str:
        parsed = urlparse(url if url.startswith("http") else f"https://{url}")
        domain = parsed.netloc or parsed.path
        domain = re.sub(r"^www\.", "", domain)
        return domain.split("/")[0].lower()

    def _fetch_wikipedia_for_transparency(self, query: str) -> str:
        """Lightweight Wikipedia fetch for transparency evidence fallback."""
        import requests as _requests
        try:
            resp = _requests.get(
                self._WIKIPEDIA_API,
                params={"action": "query", "list": "search",
                        "srsearch": query, "srlimit": 1, "format": "json"},
                timeout=10,
            )
            if resp.status_code != 200:
                return ""
            results = resp.json().get("query", {}).get("search", [])
            if not results:
                return ""
            title = results[0]["title"]
            resp2 = _requests.get(
                self._WIKIPEDIA_API,
                params={"action": "query", "titles": title, "prop": "extracts",
                        "exintro": True, "explaintext": True, "format": "json"},
                timeout=10,
            )
            pages = resp2.json().get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "")
                if extract:
                    return extract[:2000]
        except Exception:
            pass
        return ""

    def analyze(
        self,
        url: str,
        outlet_name: str | None = None,
        articles: list[dict] | None = None,
        about_page_text: str = "",
        wikipedia_text: str = "",
    ) -> TransparencyResult:
        """Analyze outlet transparency from about page, search results, and articles.

        When about_page_text is unavailable, wikipedia_text serves as
        supplementary evidence so the LLM doesn't default to worst-case
        transparency scores.
        """
        domain = self._extract_domain(url)

        # Gather evidence: search for transparency-related info
        evidence_parts = []

        if about_page_text:
            evidence_parts.append(f"ABOUT PAGE TEXT:\n{about_page_text[:2000]}")
        if wikipedia_text:
            evidence_parts.append(f"WIKIPEDIA TEXT:\n{wikipedia_text[:2000]}")

        # Search for ownership/funding info
        name = outlet_name or domain
        try:
            results = self.search_backend.search(
                f'"{name}" ownership funding about us mission', max_results=5
            )
            if results:
                snippets = "\n".join(
                    f"- {r.get('title', '')}: {r.get('body', '')}" for r in results[:5]
                )
                evidence_parts.append(f"SEARCH RESULTS:\n{snippets}")
        except Exception as e:
            logger.warning(f"Transparency search failed: {e}")

        # Fallback: fetch Wikipedia if no about page and no wikipedia_text provided
        if not about_page_text and not wikipedia_text:
            try:
                wiki = self._fetch_wikipedia_for_transparency(name)
                if wiki:
                    evidence_parts.append(f"WIKIPEDIA TEXT:\n{wiki[:2000]}")
            except Exception as e:
                logger.debug(f"Wikipedia fallback for transparency failed: {e}")

        # Check if articles have author bylines — check ALL articles, not a subset
        if articles:
            authored_count = sum(
                1 for a in articles
                if a.get("author") or "By " in (a.get("text", "")[:200])
            )
            total_checked = len(articles)
            has_authors = authored_count > 0
            evidence_parts.append(
                f"ARTICLE AUTHORSHIP: {authored_count}/{total_checked} articles have identified authors"
            )

        evidence = "\n\n".join(evidence_parts) if evidence_parts else "No evidence available."

        user_prompt = (
            f'Evaluate the transparency of "{name}" (domain: {domain}).\n\n'
            f"EVIDENCE:\n{evidence[:4000]}"
        )

        try:
            result: TransparencyLLMOutput = self.llm.invoke([
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ])
            return TransparencyResult(
                domain=domain,
                outlet_name=outlet_name,
                has_about_page=result.has_about_page,
                discloses_ownership=result.discloses_ownership,
                discloses_funding=result.discloses_funding,
                identifies_authors=result.identifies_authors,
                discloses_location=result.discloses_location,
                score=result.transparency_score,
                confidence=result.confidence,
                reasoning=result.reasoning,
            )
        except Exception as e:
            logger.error(f"Transparency analysis failed: {e}")
            return TransparencyResult(
                domain=domain,
                outlet_name=outlet_name,
                score=5.0,  # Conservative default
                confidence=0.0,
                reasoning=f"Transparency analysis failed: {e}",
            )


# =============================================================================
# OneSidednessAnalyzer (MBFC Factuality: 10% weight)
# =============================================================================

class OneSidednessAnalyzer:
    """Evaluates one-sidedness, bias omission, and propaganda per MBFC methodology.

    Measures how well a source provides balanced perspectives versus exhibiting
    significant ideological bias or propaganda, including use of emotional or
    loaded language. Score 0-10 where 0 is perfectly balanced.
    """

    SYSTEM_PROMPT = """You are an expert media analyst evaluating one-sidedness and propaganda in news coverage.

This is SEPARATE from editorial bias scoring. Here you evaluate:
- Whether the outlet presents BALANCED perspectives or is one-sided
- Whether emotional/loaded language is used to PERSUADE readers
- Whether propaganda techniques are employed

SCORING (0-10, lower is better):
  0: Perfect balance — presents all sides equally, no emotional manipulation
  1: Almost balanced — very minor favoritism, minimal emotional cues
  2: Minor bias — slightly favors one side, occasional emotional terms
  3: Small bias — favors one side more frequently, moderate emotional language
  4: Moderate bias — preference for one side, increasingly emotional wording
  5: Noticeable bias — significantly favors one side, frequent emotional language
  6: Frequent bias — dismisses opposing side, regularly provocative language
  7: Significant bias — minimal alternative perspectives, inflammatory rhetoric
  8: Heavy bias — consistently one-sided, pervasive emotional language bordering on propaganda
  9: Strong bias — rarely includes alternatives, extreme emotional framing
  10: Extreme propaganda — exclusively one side, inflammatory/divisive/manipulative language

IMPORTANT: This evaluates BALANCE and RHETORIC, not political direction.
A perfectly balanced outlet scores 0 regardless of which topics it covers.
An outlet that covers only one political perspective with emotional language scores high.

Base your assessment ONLY on the article content provided.

For propaganda_findings, classify concrete article-level quotes into these 14 techniques only:
- Appeal_to_Authority: citing authority figures to support claims
- Appeal_to_fear-prejudice: using fear or prejudice to influence
- Bandwagon,Reductio_ad_Hitlerum: "everyone does it" appeals or Nazi comparisons
- Black-and-White_Fallacy: presenting only two choices
- Causal_Oversimplification: oversimplifying cause-effect
- Doubt: questioning credibility without evidence
- Exaggeration,Minimisation: overstating or understating facts
- Flag-Waving: appealing to patriotism/nationalism
- Loaded_Language: using emotionally charged words
- Name_Calling,Labeling: using derogatory labels
- Repetition: repeating messages for emphasis
- Slogans: using catchy phrases
- Thought-terminating_Cliches: phrases discouraging critical thinking
- Whataboutism,Straw_Men,Red_Herring: deflection tactics

Return propaganda_findings only when there is an exact quote/span from the supplied articles.
For each finding include technique, text_snippet, article_number, article_title, context, confidence, and explanation.
Prefer precision over recall. If no concrete technique evidence is present, return an empty list."""

    def __init__(
        self,
        model: str = "gpt-5-mini-2025-08-07",
        temperature: float = 0.0,
    ):
        self.llm = get_llm(model, temperature).with_structured_output(OneSidednessLLMOutput)

    def analyze(self, articles: list[dict[str, str]]) -> OneSidednessResult:
        """Analyze articles for one-sidedness and propaganda."""
        if not articles:
            return OneSidednessResult(
                score=5.0,
                articles_analyzed=0,
                confidence=0.0,
                reasoning="No articles provided for analysis",
            )

        # Format articles — use full review set, not a truncated subset
        articles_text = []
        for i, article in enumerate(articles, 1):
            title = article.get("title", "Untitled")
            text = article.get("text", "")[:2000]
            articles_text.append(f"ARTICLE {i}:\nTitle: {title}\nText: {text}\n")

        combined = "\n---\n".join(articles_text)

        user_prompt = (
            f"Analyze the following articles for one-sidedness, balance, and propaganda.\n\n"
            f"{combined}\n\n"
            f"Evaluate:\n"
            f"1. Does the outlet present opposing viewpoints or only one side?\n"
            f"2. Is emotional or loaded language used to persuade?\n"
            f"3. Are propaganda techniques present from the 14 allowed labels?\n"
            f"4. For each propaganda finding, quote the exact span and identify the article number/title.\n"
            f"5. Score from 0 (perfect balance) to 10 (extreme propaganda)"
        )

        try:
            result: OneSidednessLLMOutput = self.llm.invoke([
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ])
            return OneSidednessResult(
                score=result.one_sidedness_score,
                uses_emotional_language=result.uses_emotional_language,
                propaganda_level=result.propaganda_level,
                presents_opposing_views=result.presents_opposing_views,
                articles_analyzed=len(articles),
                confidence=result.confidence,
                reasoning=result.reasoning,
                propaganda_findings=list(result.propaganda_findings or [])[:10],
            )
        except Exception as e:
            logger.error(f"One-sidedness analysis failed: {e}")
            return OneSidednessResult(
                score=5.0,
                articles_analyzed=len(articles),
                confidence=0.0,
                reasoning=f"One-sidedness analysis failed: {e}",
            )


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

    # Example usage
    print("=" * 70)
    print("REFACTORED ANALYZERS - DEMO")
    print("=" * 70)

    # Test OpinionAnalyzer
    print("\n1. OpinionAnalyzer Test")
    print("-" * 50)
    sample_title = "The Economic Impact of Climate Change"
    sample_text = """
    Climate change is causing significant economic disruption across multiple sectors.
    According to a new report from the World Bank, global GDP could decline by 23%
    by 2100 if emissions continue at current levels. The study analyzed data from
    150 countries and found that agricultural yields have already decreased by 5%
    in affected regions. Economists warn that the poorest nations will be hit hardest.
    """
    opinion_analyzer = OpinionAnalyzer()
    result = opinion_analyzer.analyze(sample_title, sample_text)
    print(f"Title: {sample_title}")
    print(f"Classification: {result.article_type.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reasoning: {result.reasoning}")

    # Test TrafficLongevityAnalyzer with hybrid approach
    print("\n2. TrafficLongevityAnalyzer Test (Hybrid Tranco + LLM)")
    print("-" * 50)

    # Initialize analyzer (will auto-download Tranco if needed)
    traffic_analyzer = TrafficLongevityAnalyzer()

    # Show Tranco stats
    tranco_stats = traffic_analyzer.get_tranco_stats()
    print(f"\nTranco List Status:")
    print(f"  Loaded: {tranco_stats['loaded']}")
    print(f"  Domains: {tranco_stats['total_domains']:,}")
    print(f"  Thresholds: HIGH < {tranco_stats['thresholds']['HIGH']:,}, "
          f"MEDIUM < {tranco_stats['thresholds']['MEDIUM']:,}, "
          f"LOW < {tranco_stats['thresholds']['LOW']:,}")

    # Test with major domain (should be in Tranco)
    test_domains = ["bbc.com", "nytimes.com", "obscure-local-news-site.com"]

    for test_domain in test_domains:
        print(f"\n  Testing: {test_domain}")
        traffic_data = traffic_analyzer.analyze(test_domain)
        print(f"    Domain: {traffic_data.domain}")
        print(f"    Traffic Source: {traffic_data.traffic_source.value}")
        if traffic_data.tranco_rank:
            print(f"    Tranco Rank: #{traffic_data.tranco_rank:,}")
        print(f"    Traffic Tier: {traffic_data.traffic_tier.value}")
        print(f"    Confidence: {traffic_data.traffic_confidence:.2f}")
        print(f"    Creation Date: {traffic_data.creation_date}")
        print(f"    Age (years): {traffic_data.age_years}")
        print(f"    WHOIS Success: {traffic_data.whois_success}")
        if traffic_data.whois_error:
            print(f"    WHOIS Error: {traffic_data.whois_error}")

    # Test MediaTypeAnalyzer with hybrid approach
    print("\n3. MediaTypeAnalyzer Test (Hybrid Lookup + LLM)")
    print("-" * 50)

    media_analyzer = MediaTypeAnalyzer()

    # Show lookup stats
    lookup_stats = media_analyzer.get_lookup_stats()
    print(f"\nKnown Media Types Status:")
    print(f"  Loaded: {lookup_stats['loaded']}")
    print(f"  Total Outlets: {lookup_stats['total_domains']}")
    if lookup_stats['by_type']:
        print(f"  By Type: {lookup_stats['by_type']}")

    # Test with outlets (mix of lookup and LLM)
    test_outlets = ["nytimes.com", "cnn.com", "obscure-blog-site.com"]

    for test_outlet in test_outlets:
        print(f"\n  Testing: {test_outlet}")
        media_result = media_analyzer.analyze(test_outlet)
        print(f"    Media Type: {media_result.media_type.value}")
        print(f"    Source: {media_result.source.value}")
        print(f"    Confidence: {media_result.confidence:.2f}")
        print(f"    Reasoning: {media_result.reasoning}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
