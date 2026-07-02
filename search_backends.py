"""
search_backends.py
Pluggable search backends for the media profiling pipeline.

Three implementations:
  - DDGSearchBackend: DuckDuckGo (default, used by 'system' mode)
  - OpenAISearchBackend: OpenAI Responses API web_search tool
  - HybridSearchBackend: Routes site:-targeted queries to DDG, everything
    else to OpenAI (used by 'hybrid' mode)
"""
import logging
import re
import time
from typing import Protocol, runtime_checkable

from openai import OpenAI
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

from config import EXCLUDED_DOMAINS, SOCIAL_MEDIA_DOMAINS
from cost_tracker import current_tracker

logger = logging.getLogger(__name__)


@runtime_checkable
class SearchBackend(Protocol):
    """Protocol for pluggable search backends."""

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """
        Execute a search query.

        Args:
            query: The search query string
            max_results: Maximum number of results to return

        Returns:
            List of dicts with keys: title, body, url
        """
        ...


class DDGSearchBackend:
    """DuckDuckGo search backend (default)."""

    _RATE_LIMIT_DELAY = 1.5  # seconds between searches to avoid DDG throttling

    def __init__(self):
        self._ddgs = DDGS()
        self._last_search_time = 0.0

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        try:
            # Rate limiting: wait between searches to avoid DDG throttling (~30 req/min)
            elapsed = time.monotonic() - self._last_search_time
            if elapsed < self._RATE_LIMIT_DELAY:
                time.sleep(self._RATE_LIMIT_DELAY - elapsed)
            self._last_search_time = time.monotonic()

            results = list(self._ddgs.text(query, max_results=max_results))
            if not results:
                logger.info(f"DDG search returned 0 results for '{query[:80]}'")
            return [
                {
                    "title": r.get("title", ""),
                    "body": r.get("body", "") or r.get("snippet", ""),
                    "url": r.get("href", "") or r.get("link", ""),
                }
                for r in results
            ]
        except Exception as e:
            logger.warning(f"DDG search failed for '{query[:60]}...': {e}")
            return []

    def batch_search(
        self, queries: list[str], max_results_per_query: int = 5
    ) -> dict[str, list[dict]]:
        """Execute multiple queries sequentially (DDG has no batch API)."""
        results = {}
        for query in queries:
            results[query] = self.search(query, max_results=max_results_per_query)
        return results


class OpenAISearchBackend:
    """OpenAI Responses API web_search backend.

    Uses the native web_search tool to perform searches. Returns raw search
    results (title, snippet, URL) instead of LLM-synthesized summaries to
    prevent information loss when analyzers re-process the results.

    Injects EXCLUDED_DOMAINS + SOCIAL_MEDIA_DOMAINS as -site: operators
    into every query to prevent data leakage.
    """

    def __init__(self, model: str = "gpt-5-mini-2025-08-07"):
        self.client = OpenAI()
        self.model = model
        self._all_excluded = [d.lower() for d in EXCLUDED_DOMAINS + SOCIAL_MEDIA_DOMAINS]
        self._exclusion_string = " ".join(
            f"-site:{domain}" for domain in self._all_excluded
        )

    def _filter_excluded_domains(self, results: list[dict]) -> list[dict]:
        """Remove results whose URL belongs to an excluded domain.

        This is the hard enforcement layer — prompt-based -site: operators
        are unreliable with the OpenAI web_search tool, so we filter post-hoc.
        """
        filtered = []
        for r in results:
            url = (r.get("url") or "").lower()
            if any(domain in url for domain in self._all_excluded):
                logger.warning(f"Filtered excluded domain from search result: {url}")
            else:
                filtered.append(r)
        return filtered

    @staticmethod
    def _extract_urls_from_response(response) -> list[str]:
        """Extract all URLs from response annotations."""
        urls = []
        if hasattr(response, "output") and response.output:
            for block in response.output:
                if hasattr(block, "content"):
                    for content_item in block.content:
                        if hasattr(content_item, "annotations"):
                            for ann in content_item.annotations:
                                if hasattr(ann, "url") and ann.url:
                                    urls.append(ann.url)
        return urls

    @staticmethod
    def _parse_structured_results(text: str, urls: list[str]) -> list[dict]:
        """Parse TITLE/URL/SNIPPET blocks from LLM output.

        Expected format per result:
            TITLE: <page title>
            URL: <source url>
            SNIPPET: <relevant text excerpt>
            ---

        Falls back to paragraph splitting if structured parsing yields nothing.
        """
        results = []

        # Try structured parsing first
        blocks = re.split(r"\n---\n?", text)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            title_match = re.search(r"TITLE:\s*(.+)", block)
            url_match = re.search(r"URL:\s*(\S+)", block)
            snippet_match = re.search(r"SNIPPET:\s*(.+)", block, re.DOTALL)

            if title_match and snippet_match:
                results.append({
                    "title": title_match.group(1).strip(),
                    "body": snippet_match.group(1).strip(),
                    "url": url_match.group(1).strip() if url_match else "",
                })

        if results:
            return results

        # Fallback: split by paragraphs and pair with annotation URLs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return [{"title": "", "body": text, "url": urls[0] if urls else ""}]

        for i, para in enumerate(paragraphs):
            url = urls[i] if i < len(urls) else ""
            results.append({"title": "", "body": para, "url": url})

        return results

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        try:
            prompt = (
                f"Search the web for the following query and return the raw search results.\n\n"
                f"QUERY: {query}\n\n"
                f"SEARCH RULES:\n"
                f"1. Append these exclusion operators to EVERY search query: "
                f"{self._exclusion_string}\n"
                f"2. Return results in this EXACT format (one per result):\n"
                f"TITLE: <page title>\n"
                f"URL: <source url>\n"
                f"SNIPPET: <verbatim text excerpt from the page>\n"
                f"---\n"
                f"3. Do NOT summarize, synthesize, or interpret. Return verbatim snippets from search results.\n"
                f"4. Return up to {max_results} results."
            )

            response = self.client.responses.create(
                model=self.model,
                tools=[{"type": "web_search"}],
                input=prompt,
            )
            tracker = current_tracker()
            if tracker is not None:
                tracker.record_responses(response, model=self.model, used_web_search=True)

            output_text = response.output_text
            if not output_text:
                return []

            urls = self._extract_urls_from_response(response)
            results = self._parse_structured_results(output_text, urls)
            results = self._filter_excluded_domains(results)
            return results[:max_results]

        except Exception as e:
            logger.warning(f"OpenAI search failed for '{query[:60]}...': {e}")
            return []

    def batch_search(
        self, queries: list[str], max_results_per_query: int = 5
    ) -> dict[str, list[dict]]:
        """Execute multiple queries in a single OpenAI API call.

        Combines all queries into one prompt with numbered sections,
        reducing API calls from N to 1.
        """
        if not queries:
            return {}
        if len(queries) == 1:
            return {queries[0]: self.search(queries[0], max_results_per_query)}

        try:
            numbered_queries = "\n".join(
                f"QUERY {i+1}: {q}" for i, q in enumerate(queries)
            )

            prompt = (
                f"Search the web for EACH of the following queries separately. "
                f"Return raw search results for each query.\n\n"
                f"{numbered_queries}\n\n"
                f"SEARCH RULES:\n"
                f"1. Append these exclusion operators to EVERY search query: "
                f"{self._exclusion_string}\n"
                f"2. For each query, return results in this EXACT format:\n"
                f"=== QUERY N ===\n"
                f"TITLE: <page title>\n"
                f"URL: <source url>\n"
                f"SNIPPET: <verbatim text excerpt from the page>\n"
                f"---\n"
                f"(repeat for up to {max_results_per_query} results per query)\n\n"
                f"3. Do NOT summarize, synthesize, or interpret. Return verbatim snippets.\n"
                f"4. Search for ALL queries — do not skip any."
            )

            response = self.client.responses.create(
                model=self.model,
                tools=[{"type": "web_search"}],
                input=prompt,
            )
            tracker = current_tracker()
            if tracker is not None:
                tracker.record_responses(response, model=self.model, used_web_search=True)

            output_text = response.output_text
            if not output_text:
                return {q: [] for q in queries}

            urls = self._extract_urls_from_response(response)

            # Split output by query sections
            results = {}
            sections = re.split(r"===\s*QUERY\s*\d+\s*===", output_text)
            # First element is before "=== QUERY 1 ===", skip it
            sections = [s.strip() for s in sections[1:] if s.strip()]

            for i, query in enumerate(queries):
                if i < len(sections):
                    section_results = self._parse_structured_results(
                        sections[i], urls
                    )
                    section_results = self._filter_excluded_domains(section_results)
                    results[query] = section_results[:max_results_per_query]
                else:
                    # Section missing — fall back to individual search
                    logger.debug(
                        f"Batch section missing for query {i+1}, falling back"
                    )
                    results[query] = self.search(query, max_results_per_query)

            return results

        except Exception as e:
            logger.warning(f"Batch search failed, falling back to individual: {e}")
            results = {}
            for query in queries:
                results[query] = self.search(query, max_results_per_query)
            return results


class HybridSearchBackend:
    """Routes site:-targeted queries to DDG, everything else to OpenAI.

    FactCheckSearcher uses site: operators (e.g., site:politifact.com) which
    DDG handles reliably as a search engine constraint. For broader research
    queries (history, ownership, external analysis), OpenAI's native web_search
    provides richer results.
    """

    def __init__(self, model: str = "gpt-5-mini-2025-08-07"):
        self._ddg = DDGSearchBackend()
        self._openai = OpenAISearchBackend(model=model)

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        if "site:" in query:
            logger.debug(f"Routing to DDG (site: query): {query[:60]}")
            return self._ddg.search(query, max_results)
        logger.debug(f"Routing to OpenAI: {query[:60]}")
        return self._openai.search(query, max_results)

    def batch_search(
        self, queries: list[str], max_results_per_query: int = 5
    ) -> dict[str, list[dict]]:
        """Batch search: routes site: queries to DDG individually,
        batches the rest through a single OpenAI call."""
        results = {}
        openai_queries = []

        for query in queries:
            if "site:" in query:
                results[query] = self._ddg.search(query, max_results_per_query)
            else:
                openai_queries.append(query)

        if openai_queries:
            openai_results = self._openai.batch_search(
                openai_queries, max_results_per_query
            )
            results.update(openai_results)

        return results
