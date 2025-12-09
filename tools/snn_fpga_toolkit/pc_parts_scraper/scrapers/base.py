"""
Base scraper engine with async support, rate limiting, and proxy rotation
"""

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import urljoin, urlparse

import httpx
from fake_useragent import UserAgent
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from bs4 import BeautifulSoup
from asyncio_throttle import Throttler

from utils.models import PCPart, ScrapingRun
from utils.helpers import (
    clean_price, normalize_condition, categorize_part,
    calculate_rarity_score, fuzzy_match_keywords, load_config
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """
    Base class for all PC parts scrapers.
    Provides common functionality for HTTP requests, parsing, and data handling.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.ua = UserAgent()
        self.session: Optional[httpx.AsyncClient] = None
        self.throttler: Optional[Throttler] = None

        # Scraper metadata (override in subclasses)
        self.name = "base"
        self.base_url = ""
        self.source_category = "unknown"

        # Rate limiting
        domain = urlparse(self.base_url).netloc if self.base_url else "default"
        rate_limit = self.config.get('rate_limits', {}).get(
            domain,
            self.config.get('rate_limits', {}).get('default', 30)
        )
        self.requests_per_minute = rate_limit

        # Results
        self.items: List[PCPart] = []
        self.errors: List[str] = []

        # Proxy configuration
        self.proxies = []
        if self.config.get('proxy', {}).get('enabled'):
            self.proxies = self.config['proxy'].get('pool', [])

    async def __aenter__(self):
        await self.init_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_session()

    async def init_session(self):
        """Initialize async HTTP session with rotation"""
        # Set up throttler for rate limiting
        self.throttler = Throttler(
            rate_limit=self.requests_per_minute,
            period=60.0
        )

        # Configure client
        timeout = httpx.Timeout(
            self.config.get('general', {}).get('request_timeout', 30)
        )

        self.session = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            http2=True
        )

    async def close_session(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.aclose()

    def get_headers(self) -> Dict[str, str]:
        """Generate random headers to avoid detection"""
        return {
            'User-Agent': self.ua.random if self.config.get('general', {}).get('user_agent_rotation', True) else self.ua.chrome,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }

    def get_proxy(self) -> Optional[str]:
        """Get a random proxy from the pool"""
        if self.proxies and self.config.get('proxy', {}).get('rotation', True):
            return random.choice(self.proxies)
        return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError))
    )
    async def fetch(self, url: str, **kwargs) -> httpx.Response:
        """Fetch a URL with rate limiting, retries, and proxy rotation"""
        async with self.throttler:
            headers = {**self.get_headers(), **kwargs.pop('headers', {})}

            proxy = self.get_proxy()
            if proxy:
                # Create new client with proxy for this request
                async with httpx.AsyncClient(proxies=proxy, timeout=self.session.timeout) as client:
                    response = await client.get(url, headers=headers, **kwargs)
            else:
                response = await self.session.get(url, headers=headers, **kwargs)

            response.raise_for_status()
            return response

    async def fetch_json(self, url: str, **kwargs) -> Dict[str, Any]:
        """Fetch and parse JSON response"""
        response = await self.fetch(url, **kwargs)
        return response.json()

    async def fetch_html(self, url: str, **kwargs) -> BeautifulSoup:
        """Fetch and parse HTML response"""
        response = await self.fetch(url, **kwargs)
        return BeautifulSoup(response.text, 'lxml')

    def parse_item(self, raw_item: Any, soup: BeautifulSoup = None) -> Optional[PCPart]:
        """
        Parse a raw item into a PCPart object.
        Override in subclasses for specific parsing logic.
        """
        try:
            # Extract basic info
            title = self.extract_title(raw_item, soup)
            if not title:
                return None

            description = self.extract_description(raw_item, soup)
            price = self.extract_price(raw_item, soup)
            url = self.extract_url(raw_item, soup)

            if not url:
                return None

            # Categorize and score
            category = categorize_part(title, description or "")
            rare_keywords = self.config.get('rare_part_keywords', [])
            matched_keywords = fuzzy_match_keywords(
                f"{title} {description or ''}",
                rare_keywords
            )

            rarity_score = calculate_rarity_score(
                title,
                description or "",
                rare_keywords,
                category,
                price
            )

            # Create PCPart object
            part = PCPart(
                title=title,
                description=description,
                category=category,
                price=price,
                source_name=self.name,
                source_url=url,
                source_id=self.extract_item_id(raw_item, soup),
                source_category=self.source_category,
                location=self.extract_location(raw_item, soup),
                shipping_available=self.extract_shipping(raw_item, soup),
                condition=normalize_condition(self.extract_condition(raw_item, soup) or ""),
                condition_notes=self.extract_condition_notes(raw_item, soup),
                image_urls=self.extract_images(raw_item, soup),
                thumbnail_url=self.extract_thumbnail(raw_item, soup),
                seller_name=self.extract_seller(raw_item, soup),
                seller_rating=self.extract_seller_rating(raw_item, soup),
                is_auction=self.extract_is_auction(raw_item, soup),
                bid_count=self.extract_bid_count(raw_item, soup),
                auction_end_time=self.extract_auction_end(raw_item, soup),
                matched_keywords=matched_keywords,
                rarity_score=rarity_score,
                is_rare=rarity_score >= 50,
                first_seen=datetime.utcnow(),
                last_seen=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            return part

        except Exception as e:
            logger.error(f"Error parsing item: {e}")
            self.errors.append(str(e))
            return None

    # Abstract methods - must be implemented by subclasses
    @abstractmethod
    async def scrape(self, keywords: List[str] = None, **kwargs) -> List[PCPart]:
        """Main scraping method - implement in subclasses"""
        pass

    @abstractmethod
    def extract_title(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        pass

    @abstractmethod
    def extract_url(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        pass

    # Optional extraction methods - override as needed
    def extract_description(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        return None

    def extract_price(self, item: Any, soup: BeautifulSoup = None) -> Optional[float]:
        return None

    def extract_item_id(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        return None

    def extract_location(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        return None

    def extract_shipping(self, item: Any, soup: BeautifulSoup = None) -> bool:
        return True

    def extract_condition(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        return None

    def extract_condition_notes(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        return None

    def extract_images(self, item: Any, soup: BeautifulSoup = None) -> List[str]:
        return []

    def extract_thumbnail(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        return None

    def extract_seller(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        return None

    def extract_seller_rating(self, item: Any, soup: BeautifulSoup = None) -> Optional[float]:
        return None

    def extract_is_auction(self, item: Any, soup: BeautifulSoup = None) -> bool:
        return False

    def extract_bid_count(self, item: Any, soup: BeautifulSoup = None) -> Optional[int]:
        return None

    def extract_auction_end(self, item: Any, soup: BeautifulSoup = None) -> Optional[datetime]:
        return None

    def make_absolute_url(self, url: str) -> str:
        """Convert relative URL to absolute"""
        if url.startswith('http'):
            return url
        return urljoin(self.base_url, url)


class MultiSourceScraper:
    """
    Orchestrates scraping from multiple sources concurrently
    """

    def __init__(self, scrapers: List[BaseScraper], config: Dict[str, Any] = None):
        self.scrapers = scrapers
        self.config = config or load_config()
        self.all_items: List[PCPart] = []
        self.runs: List[ScrapingRun] = []

    async def scrape_all(
        self,
        keywords: List[str] = None,
        max_concurrent: int = 5
    ) -> List[PCPart]:
        """Scrape all sources concurrently with controlled concurrency"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def scrape_with_semaphore(scraper: BaseScraper):
            async with semaphore:
                return await self._run_scraper(scraper, keywords)

        # Run all scrapers concurrently
        tasks = [scrape_with_semaphore(s) for s in self.scrapers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Scraper failed: {result}")
            elif result:
                self.all_items.extend(result)

        return self.all_items

    async def _run_scraper(
        self,
        scraper: BaseScraper,
        keywords: List[str] = None
    ) -> List[PCPart]:
        """Run a single scraper and track the run"""
        run = ScrapingRun(
            source_name=scraper.name,
            start_time=datetime.utcnow(),
            status='running'
        )
        self.runs.append(run)

        try:
            async with scraper:
                items = await scraper.scrape(keywords)

                run.end_time = datetime.utcnow()
                run.items_found = len(items)
                run.errors = len(scraper.errors)
                run.error_messages = scraper.errors[:10]  # Keep first 10 errors
                run.status = 'completed'

                logger.info(
                    f"{scraper.name}: Found {len(items)} items, {len(scraper.errors)} errors"
                )

                return items

        except Exception as e:
            run.end_time = datetime.utcnow()
            run.status = 'failed'
            run.error_messages = [str(e)]
            logger.error(f"{scraper.name} failed: {e}")
            raise
