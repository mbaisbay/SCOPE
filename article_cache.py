"""
article_cache.py
Caches scraped articles per media outlet domain for benchmark fairness.
Ensures all models and modes use the same scraped articles.

Storage format:
  article_cache/
    <domain>/
      articles.json   <-- List of serialized Article objects
      metadata.json   <-- Scrape timestamp, count, source_url
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from urllib.parse import urlparse

from scraper import Article, MediaScraper

logger = logging.getLogger(__name__)

ARTICLE_CACHE_DIR = Path("article_cache")


class ArticleCache:
    """
    Scrape-once, reuse-everywhere article cache for benchmark fairness.

    Keyed by outlet domain. On first request for a domain, scrapes articles
    and saves them. Subsequent requests return the cached articles.
    """

    def __init__(self, cache_dir: Path = ARTICLE_CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self._memory_cache: dict[str, list[Article]] = {}

    @staticmethod
    def _domain_from_url(url: str) -> str:
        parsed = urlparse(url if url.startswith("http") else f"https://{url}")
        domain = parsed.netloc or parsed.path
        domain = domain.replace("www.", "").strip("/")
        return domain

    def _domain_dir(self, domain: str) -> Path:
        return self.cache_dir / domain.replace("/", "_")

    def get_articles(self, source_url: str, max_articles: int = 20) -> list[Article]:
        """
        Get articles for a source URL. Returns cached if available,
        otherwise scrapes and caches.
        """
        domain = self._domain_from_url(source_url)

        # 1. Check in-memory cache
        if domain in self._memory_cache:
            logger.info(f"[ArticleCache] HIT (memory): {domain}, {len(self._memory_cache[domain])} articles")
            return self._memory_cache[domain][:max_articles]

        # 2. Check disk cache
        domain_dir = self._domain_dir(domain)
        articles_path = domain_dir / "articles.json"
        if articles_path.exists():
            try:
                articles = self._load_from_disk(articles_path)
                self._memory_cache[domain] = articles
                logger.info(f"[ArticleCache] HIT (disk): {domain}, {len(articles)} articles")
                return articles[:max_articles]
            except Exception as e:
                logger.warning(f"[ArticleCache] Disk cache corrupt for {domain}: {e}")

        # 3. Scrape and cache
        logger.info(f"[ArticleCache] MISS: {domain}, scraping up to {max_articles} articles...")
        scraper = MediaScraper(source_url, max_articles=max_articles)
        articles = scraper.scrape_feed()

        self._save_to_disk(domain, source_url, articles)
        self._memory_cache[domain] = articles
        logger.info(f"[ArticleCache] Cached {len(articles)} articles for {domain}")

        return articles[:max_articles]

    def _save_to_disk(self, domain: str, source_url: str, articles: list[Article]):
        domain_dir = self._domain_dir(domain)
        domain_dir.mkdir(parents=True, exist_ok=True)

        articles_data = [
            {
                "url": a.url,
                "title": a.title,
                "text": a.text,
                "author": a.author,
                "date": a.date,
                "category": a.category,
                "has_sources": a.has_sources,
                "source_links": a.source_links,
                "is_opinion": a.is_opinion,
            }
            for a in articles
        ]
        with open(domain_dir / "articles.json", "w", encoding="utf-8") as f:
            json.dump(articles_data, f, indent=2, ensure_ascii=False)

        metadata = {
            "domain": domain,
            "source_url": source_url,
            "scrape_date": datetime.now().isoformat(),
            "article_count": len(articles),
        }
        with open(domain_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _load_from_disk(self, articles_path: Path) -> list[Article]:
        with open(articles_path, "r", encoding="utf-8") as f:
            articles_data = json.load(f)
        return [
            Article(
                url=a["url"],
                title=a["title"],
                text=a["text"],
                author=a.get("author"),
                date=a.get("date"),
                category=a.get("category"),
                has_sources=a.get("has_sources", False),
                source_links=a.get("source_links", []),
                is_opinion=a.get("is_opinion", False),
            )
            for a in articles_data
        ]

    def clear(self):
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._memory_cache.clear()
        logger.info("[ArticleCache] Cache cleared")
