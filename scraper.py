"""
Web Scraper Module - "Brute Force" Edition
Designed to aggressively crawl homepage links when sitemaps fail.
"""

import re
import time
import logging
import random
from typing import List, Optional, Set, Dict
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
import requests
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# --- Data Models ---

@dataclass
class Article:
    url: str
    title: str
    text: str
    author: Optional[str] = None
    date: Optional[str] = None
    category: Optional[str] = None
    has_sources: bool = False
    source_links: List[str] = field(default_factory=list)
    is_opinion: bool = False

@dataclass
class SiteMetadata:
    domain: str
    has_about_page: bool = False
    about_text: str = ""
    ownership_disclosed: bool = False
    ownership_info: str = ""
    funding_disclosed: bool = False
    funding_info: str = ""
    location_disclosed: bool = False
    location_info: str = ""
    contact_info: str = ""
    has_author_pages: bool = False

# --- The Scraper Class ---

class MediaScraper:
    def __init__(self, base_url: str, max_articles: int = 30):
        self.base_url = base_url.rstrip('/')
        self.domain = urlparse(self.base_url).netloc.replace('www.', '')
        self.max_articles = max_articles
        self.visited_urls: Set[str] = set()
        self.session = requests.Session()
        
        # Robust Headers to look like a real browser (Chrome on Windows)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/',
        }

    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Downloads and parses a page safely."""
        try:
            time.sleep(random.uniform(0.5, 1.5)) # Delay to avoid 429 Rate Limits
            resp = self.session.get(url, headers=self.headers, timeout=15)
            resp.raise_for_status()
            
            # Fix encoding issues
            if resp.encoding == 'ISO-8859-1':
                resp.encoding = resp.apparent_encoding
            
            # Suppress XML warning for RSS feeds/Sitemaps
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
                soup = BeautifulSoup(resp.text, 'html.parser')
                
            return soup
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

    def _discover_rss_urls(self, soup) -> Set[str]:
        """Discover article URLs from RSS/Atom feeds."""
        urls = set()

        # 1. Check for <link rel="alternate" type="application/rss+xml"> in homepage HTML
        feed_links = soup.find_all('link', attrs={'type': re.compile(r'application/(rss|atom)\+xml')})
        feed_urls = [urljoin(self.base_url, link.get('href', '')) for link in feed_links if link.get('href')]

        # 2. Try common feed paths
        for path in ['/feed', '/feed.xml', '/rss', '/rss.xml', '/atom.xml']:
            feed_urls.append(f"{self.base_url}{path}")

        for feed_url in feed_urls:
            try:
                resp = self.session.get(feed_url, headers=self.headers, timeout=10)
                if resp.status_code != 200:
                    continue
                content = resp.text
                if '<rss' not in content and '<feed' not in content and '<item' not in content:
                    continue
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
                    feed_soup = BeautifulSoup(content, 'html.parser')
                # RSS <item><link> or Atom <entry><link>
                for item in feed_soup.find_all(['item', 'entry']):
                    link_tag = item.find('link')
                    if link_tag:
                        href = link_tag.get('href') or link_tag.get_text(strip=True)
                        if href and href.startswith('http'):
                            urls.add(href)
                if urls:
                    logger.info(f"RSS/Atom feed discovery found {len(urls)} URLs from {feed_url}")
                    break  # Use first successful feed
            except Exception as e:
                logger.debug(f"Feed fetch failed for {feed_url}: {e}")

        return urls

    def _discover_substack_urls(self, soup) -> Set[str]:
        """Discover article URLs from Substack archive pages."""
        urls = set()
        is_substack = (
            'substack.com' in self.domain or
            bool(soup.find('meta', attrs={'content': re.compile(r'substack', re.I)})) or
            bool(soup.find('link', attrs={'href': re.compile(r'substack')}))
        )
        if not is_substack:
            return urls

        logger.info(f"Detected Substack site: {self.base_url}")
        archive_url = f"{self.base_url}/archive"
        archive_soup = self.fetch_page(archive_url)
        if archive_soup:
            for a in archive_soup.find_all('a', href=True):
                href = a['href']
                full_url = urljoin(self.base_url, href)
                if self.domain in full_url and '/p/' in full_url:
                    urls.add(full_url)
            logger.info(f"Substack archive discovery found {len(urls)} URLs")

        return urls

    def _discover_sitemap_urls(self) -> Set[str]:
        """Discover article URLs from sitemap.xml as fallback."""
        urls = set()
        for path in ['/sitemap.xml', '/sitemap-posts.xml', '/post-sitemap.xml']:
            try:
                resp = self.session.get(f"{self.base_url}{path}", headers=self.headers, timeout=10)
                if resp.status_code != 200:
                    continue
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
                    sitemap_soup = BeautifulSoup(resp.text, 'html.parser')
                for loc in sitemap_soup.find_all('loc'):
                    url = loc.get_text(strip=True)
                    if url and self.domain in url and len(url) > len(self.base_url) + 10:
                        urls.add(url)
                if urls:
                    logger.info(f"Sitemap discovery found {len(urls)} URLs from {path}")
                    break
            except Exception as e:
                logger.debug(f"Sitemap fetch failed for {path}: {e}")
        return urls

    def scrape_feed(self) -> List[Article]:
        """
        The Main Method called by profiler.py.
        Strategy: Try RSS/feed first, then homepage links, with sitemap fallback.
        """
        t_start = time.time()
        logger.info(f"Scraping homepage: {self.base_url}")
        soup = self.fetch_page(self.base_url)
        if not soup:
            logger.error(f"Could not load homepage for {self.base_url}. Site might be blocking requests.")
            return []

        # 0. Try RSS/Atom feeds and Substack archive first
        candidates = self._discover_rss_urls(soup)
        candidates |= self._discover_substack_urls(soup)

        # 1. Collect all potential links from homepage
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(self.base_url, href)

            # Filter logic
            if self.domain not in full_url: continue # Internal only
            if len(full_url) < len(self.base_url) + 10: continue # Too short
            # Skip obvious non-article pages
            if any(x in full_url for x in ['/tag/', '/search/', '/category/', '/login', '.pdf', '.jpg', '/video/', '/live/']): continue

            candidates.add(full_url)

        # 1b. Sitemap fallback if few candidates found
        if len(candidates) < 5:
            logger.info(f"Only {len(candidates)} candidates found, trying sitemap fallback...")
            candidates |= self._discover_sitemap_urls()

        logger.info(f"Found {len(candidates)} links total.")

        # 2. Prioritize hard news over soft news for better bias analysis
        scored_candidates = []
        for url in candidates:
            score = 0
            u = url.lower()

            # Boost hard news sections
            if any(x in u for x in ['/news', '/politics', '/world', '/business', '/economy',
                                     '/uk-news', '/us-news', '/us-politics', '/global']):
                score += 10

            # Boost hard news keywords in URL slug
            if any(x in u for x in ['government', 'election', 'war', 'senate', 'congress',
                                      'parliament', 'law', 'court', 'policy', 'minister',
                                      'president', 'military', 'conflict', 'protest']):
                score += 5

            # Demote soft news sections
            if any(x in u for x in ['/sport', '/sports', '/culture', '/arts', '/travel',
                                     '/food', '/style', '/entertainment', '/life', '/lifestyle',
                                     '/celebrity', '/recipe', '/wellness', '/fitness',
                                     '/music', '/movies', '/tv-shows', '/gaming']):
                score -= 10

            scored_candidates.append((score, url))

        # Sort by score descending (hard news first)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        target_links = [x[1] for x in scored_candidates[:self.max_articles * 2]]

        top_score = scored_candidates[0][0] if scored_candidates else 0
        logger.info(f"Prioritized {len(target_links)} links (top score: {top_score}). Scraping {self.max_articles}...")

        # 3. Scrape them in parallel
        articles = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(self._parse_article, url): url for url in target_links}

            for future in as_completed(future_to_url):
                if len(articles) >= self.max_articles: break

                res = future.result()
                if res and len(res.text) > 200: # Ensure valid article text (lowered for newsletters)
                    articles.append(res)
                    logger.debug(f"Scraped article: {res.title[:80]} ({res.url})")

        elapsed = time.time() - t_start
        logger.info(f"Scraping complete for {self.base_url}: {len(articles)} articles in {elapsed:.1f}s")
        return articles

    def _parse_article(self, url: str) -> Optional[Article]:
        """Parses a single article URL."""
        if url in self.visited_urls: return None
        self.visited_urls.add(url)

        soup = self.fetch_page(url)
        if not soup:
            logger.debug(f"Failed to parse article (no soup): {url}")
            return None

        # Extract Title
        title = soup.title.get_text(strip=True) if soup.title else ""
        h1 = soup.find('h1')
        if h1: title = h1.get_text(strip=True)

        # Extract Text (Heuristic: Find the container with the most paragraphs)
        best_div = None
        max_p = 0

        # Search common content containers
        candidates = soup.find_all(['div', 'article', 'section', 'main'])
        for div in candidates:
            p_count = len(div.find_all('p', recursive=False))
            if p_count > max_p:
                max_p = p_count
                best_div = div

        if best_div and max_p > 3:
            paragraphs = best_div.find_all('p')
        else:
            paragraphs = soup.find_all('p') # Fallback

        text = "\n\n".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30])

        if len(text) < 200:
            logger.debug(f"Article too short ({len(text)} chars), skipping: {url}")
            return None

        # Check for Sources (External links)
        sources = []
        for a in soup.find_all('a', href=True):
            if 'http' in a['href'] and self.domain not in a['href']:
                sources.append(a['href'])

        # Detect if article is opinion/editorial vs straight news
        is_opinion = self._detect_opinion_article(url, title, soup)

        # Try to extract author
        author = self._extract_author(soup)

        # Try to extract category
        category = self._extract_category(url, soup)

        return Article(
            url=url,
            title=title,
            text=text,
            author=author,
            date="Unknown",
            category=category,
            has_sources=len(sources) > 0,
            source_links=sources,
            is_opinion=is_opinion
        )

    def _detect_opinion_article(self, url: str, title: str, soup: BeautifulSoup) -> bool:
        """
        Detects if an article is opinion/editorial vs straight news.
        Important for MBFC methodology which separates news reporting from editorial bias.
        """
        # URL indicators
        opinion_url_patterns = [
            '/opinion/', '/opinions/', '/editorial/', '/editorials/',
            '/op-ed/', '/oped/', '/commentary/', '/perspective/',
            '/analysis/', '/column/', '/columns/', '/blog/',
            '/views/', '/viewpoint/', '/contributor/'
        ]
        url_lower = url.lower()
        if any(pattern in url_lower for pattern in opinion_url_patterns):
            return True

        # Title indicators
        title_lower = title.lower()
        opinion_title_patterns = [
            'opinion:', 'editorial:', 'commentary:', 'analysis:',
            'column:', 'op-ed:', 'perspective:', 'letter to',
            'my view', 'i think', 'why we should', 'why i'
        ]
        if any(pattern in title_lower for pattern in opinion_title_patterns):
            return True

        # Meta tag indicators
        meta_section = soup.find('meta', {'property': 'article:section'})
        if meta_section:
            section = meta_section.get('content', '').lower()
            if any(x in section for x in ['opinion', 'editorial', 'commentary', 'analysis']):
                return True

        # Schema.org indicators
        schema_type = soup.find('script', {'type': 'application/ld+json'})
        if schema_type:
            try:
                import json
                data = json.loads(schema_type.string)
                if isinstance(data, dict):
                    article_type = data.get('@type', '').lower()
                    if 'opinion' in article_type or 'analysis' in article_type:
                        return True
            except:
                pass

        # CSS class indicators
        article_elem = soup.find('article')
        if article_elem:
            classes = ' '.join(article_elem.get('class', []))
            if any(x in classes.lower() for x in ['opinion', 'editorial', 'commentary']):
                return True

        return False

    def _extract_author(self, soup: BeautifulSoup) -> str:
        """Extracts author name from article."""
        # Common author selectors
        author_selectors = [
            ('meta', {'name': 'author'}),
            ('meta', {'property': 'article:author'}),
            ('a', {'rel': 'author'}),
            ('span', {'class': re.compile(r'author', re.I)}),
            ('p', {'class': re.compile(r'author', re.I)}),
            ('div', {'class': re.compile(r'byline', re.I)}),
        ]

        for tag, attrs in author_selectors:
            elem = soup.find(tag, attrs)
            if elem:
                if tag == 'meta':
                    return elem.get('content', 'Unknown')
                else:
                    return elem.get_text(strip=True)[:100]  # Cap length

        return "Unknown"

    def _extract_category(self, url: str, soup: BeautifulSoup) -> Optional[str]:
        """Extracts article category/section."""
        # Try meta tag
        meta_section = soup.find('meta', {'property': 'article:section'})
        if meta_section:
            return meta_section.get('content')

        # Try URL path
        path_parts = url.split('/')
        if len(path_parts) > 3:
            potential_category = path_parts[3]
            if len(potential_category) > 2 and potential_category.isalpha():
                return potential_category.title()

        return None

    def get_metadata(self) -> SiteMetadata:
        """
        Scans homepage for 'About', 'Contact', 'Terms' to estimate transparency.
        """
        meta = SiteMetadata(domain=self.domain)
        
        soup = self.fetch_page(self.base_url)
        if not soup:
            return meta

        # Convert all link text to lowercase for searching
        links_text = " ".join([a.get_text().lower() for a in soup.find_all('a', href=True)])
        footer_text = " ".join([f.get_text().lower() for f in soup.find_all('footer')])

        # 1. Check for About Page
        if any(x in links_text for x in ['about us', 'about the bbc', 'who we are', 'our story']):
            meta.has_about_page = True
            
        # 2. Check for Contact/Location
        if any(x in links_text for x in ['contact', 'contact us', 'help', 'locations']):
            meta.location_disclosed = True
            meta.contact_info = "Found contact link"

        # 3. Check for Authors/Masthead
        if any(x in links_text for x in ['meet the team', 'editorial staff', 'authors', 'journalists']):
            meta.has_author_pages = True

        # 4. Check for Funding/Ownership (Keywords in footer often indicate this)
        if any(x in footer_text for x in ['copyright', 'all rights reserved', 'published by', 'funded by']):
            meta.ownership_disclosed = True # Basic assumption for standard footers
            meta.funding_disclosed = True

        return meta