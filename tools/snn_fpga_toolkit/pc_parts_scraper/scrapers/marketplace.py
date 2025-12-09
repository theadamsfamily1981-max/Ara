"""
Used Marketplace Scrapers
Finds PC parts on Mercari, OfferUp, and other used marketplaces
"""

import re
from typing import List, Any, Optional
from urllib.parse import urlencode

from bs4 import BeautifulSoup

from scrapers.base import BaseScraper
from utils.models import PCPart
from utils.helpers import clean_price


class MercariScraper(BaseScraper):
    """
    Scraper for Mercari - Popular used goods marketplace
    Great for finding vintage PC parts at low prices
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.name = "mercari"
        self.base_url = "https://www.mercari.com"
        self.source_category = "marketplace"

    async def scrape(self, keywords: List[str] = None, **kwargs) -> List[PCPart]:
        """Scrape Mercari for PC parts"""
        if not keywords:
            keywords = self.config.get('rare_part_keywords', [])[:15]

        all_items = []

        for keyword in keywords:
            try:
                items = await self._search(keyword, **kwargs)
                all_items.extend(items)
            except Exception as e:
                self.errors.append(f"Mercari search error for '{keyword}': {e}")

        # Deduplicate
        seen = set()
        unique = [item for item in all_items if item.source_url not in seen and not seen.add(item.source_url)]

        self.items = unique
        return self.items

    async def _search(self, keyword: str, max_pages: int = 3, **kwargs) -> List[PCPart]:
        """Search Mercari for a keyword"""
        items = []

        for page in range(1, max_pages + 1):
            params = {
                'keyword': keyword,
                'categoryIds': '12',  # Electronics
                'itemStatuses': '1',  # On sale
                'sortBy': '1',  # Newest first
                'page': page,
            }

            if kwargs.get('max_price'):
                params['maxPrice'] = int(kwargs['max_price'])
            if kwargs.get('min_price'):
                params['minPrice'] = int(kwargs['min_price'])

            url = f"{self.base_url}/search?{urlencode(params)}"

            try:
                soup = await self.fetch_html(url)
                page_items = self._parse_results(soup)

                if not page_items:
                    break

                items.extend(page_items)

            except Exception as e:
                self.errors.append(f"Page {page} error: {e}")
                break

        return items

    def _parse_results(self, soup: BeautifulSoup) -> List[PCPart]:
        """Parse Mercari search results"""
        items = []

        listings = soup.select('[data-testid="SearchResults"] > div, .item-box')

        for listing in listings:
            try:
                # Skip sold items
                if listing.select_one('[data-testid="SoldBadge"], .sold'):
                    continue

                part = self.parse_item(listing, soup)
                if part:
                    items.append(part)

            except Exception as e:
                self.errors.append(f"Parse error: {e}")

        return items

    def extract_title(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        title_elem = item.select_one('[data-testid="ItemName"], .item-name, h3')
        if title_elem:
            return title_elem.get_text(strip=True)
        return None

    def extract_url(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        link = item.select_one('a[href*="/item/"], a[href*="/us/item/"]')
        if link and link.get('href'):
            href = link['href']
            if not href.startswith('http'):
                return f"{self.base_url}{href}"
            return href
        return None

    def extract_price(self, item: Any, soup: BeautifulSoup = None) -> Optional[float]:
        price_elem = item.select_one('[data-testid="ItemPrice"], .item-price, p.price')
        if price_elem:
            return clean_price(price_elem.get_text(strip=True))
        return None

    def extract_item_id(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        url = self.extract_url(item, soup)
        if url:
            match = re.search(r'/item/m(\d+)', url)
            if match:
                return match.group(1)
        return None

    def extract_thumbnail(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        img = item.select_one('img[src*="mercari"], img')
        if img:
            return img.get('src') or img.get('data-src')
        return None

    def extract_condition(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        cond_elem = item.select_one('[data-testid="ItemCondition"], .item-condition')
        if cond_elem:
            return cond_elem.get_text(strip=True)
        return 'used'

    def extract_seller(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        seller_elem = item.select_one('[data-testid="SellerName"], .seller-name')
        if seller_elem:
            return seller_elem.get_text(strip=True)
        return None

    def extract_shipping(self, item: Any, soup: BeautifulSoup = None) -> bool:
        shipping_elem = item.select_one('[data-testid="ShippingBadge"], .shipping')
        if shipping_elem:
            text = shipping_elem.get_text().lower()
            return 'free' in text or 'ship' in text
        return True


class OfferUpScraper(BaseScraper):
    """
    Scraper for OfferUp - Local marketplace with shipping option
    Good for finding local deals on PC parts
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.name = "offerup"
        self.base_url = "https://offerup.com"
        self.source_category = "local"

    async def scrape(self, keywords: List[str] = None, **kwargs) -> List[PCPart]:
        """Scrape OfferUp for PC parts"""
        if not keywords:
            keywords = [
                'graphics card', 'gpu', 'cpu', 'motherboard',
                'ram ddr', 'computer parts', 'gaming pc'
            ]

        all_items = []

        for keyword in keywords:
            try:
                items = await self._search(keyword, **kwargs)
                all_items.extend(items)
            except Exception as e:
                self.errors.append(f"OfferUp search error: {e}")

        # Deduplicate
        seen = set()
        unique = [item for item in all_items if item.source_url not in seen and not seen.add(item.source_url)]

        self.items = unique
        return self.items

    async def _search(self, keyword: str, max_pages: int = 2, **kwargs) -> List[PCPart]:
        """Search OfferUp for a keyword"""
        items = []

        for page in range(1, max_pages + 1):
            params = {
                'q': keyword,
                'SORT': 'POSTED_DATE_NEWEST',
                'PRICE_MAX': kwargs.get('max_price', 500),
                'page': page,
            }

            url = f"{self.base_url}/search?{urlencode(params)}"

            try:
                soup = await self.fetch_html(url)
                page_items = self._parse_results(soup)

                if not page_items:
                    break

                items.extend(page_items)

            except Exception as e:
                self.errors.append(f"Page {page} error: {e}")
                break

        return items

    def _parse_results(self, soup: BeautifulSoup) -> List[PCPart]:
        """Parse OfferUp search results"""
        items = []

        listings = soup.select('[data-testid="feed-item"], .listing-card, .item')

        for listing in listings:
            try:
                part = self.parse_item(listing, soup)
                if part:
                    items.append(part)
            except Exception as e:
                self.errors.append(f"Parse error: {e}")

        return items

    def extract_title(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        title_elem = item.select_one('[data-testid="item-title"], .item-title, span')
        if title_elem:
            return title_elem.get_text(strip=True)
        return None

    def extract_url(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        link = item.select_one('a[href*="/item/"]')
        if link and link.get('href'):
            href = link['href']
            if not href.startswith('http'):
                return f"{self.base_url}{href}"
            return href
        return None

    def extract_price(self, item: Any, soup: BeautifulSoup = None) -> Optional[float]:
        price_elem = item.select_one('[data-testid="item-price"], .item-price')
        if price_elem:
            text = price_elem.get_text(strip=True)
            if 'free' in text.lower():
                return 0.0
            return clean_price(text)
        return None

    def extract_item_id(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        url = self.extract_url(item, soup)
        if url:
            match = re.search(r'/item/detail/(\d+)', url)
            if match:
                return match.group(1)
        return None

    def extract_location(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        loc_elem = item.select_one('[data-testid="item-location"], .location')
        if loc_elem:
            return loc_elem.get_text(strip=True)
        return None

    def extract_thumbnail(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        img = item.select_one('img')
        if img:
            return img.get('src') or img.get('data-src')
        return None

    def extract_shipping(self, item: Any, soup: BeautifulSoup = None) -> bool:
        shipping_elem = item.select_one('[data-testid="shipping-badge"]')
        return shipping_elem is not None


class CraigslistScraper(BaseScraper):
    """
    Scraper for Craigslist - Multiple cities
    """

    def __init__(self, config=None, cities: List[str] = None):
        super().__init__(config)
        self.name = "craigslist"
        self.base_url = "https://craigslist.org"
        self.source_category = "local"
        self.cities = cities or [
            'sfbay', 'losangeles', 'seattle', 'portland',
            'denver', 'austin', 'chicago', 'newyork', 'boston'
        ]

    async def scrape(self, keywords: List[str] = None, **kwargs) -> List[PCPart]:
        """Scrape Craigslist across multiple cities"""
        if not keywords:
            keywords = [
                'vintage computer', 'old pc parts', 'graphics card',
                'server equipment', 'computer lot'
            ]

        all_items = []

        for city in self.cities:
            for keyword in keywords[:5]:  # Limit to avoid overwhelming
                try:
                    items = await self._search_city(city, keyword, **kwargs)
                    all_items.extend(items)
                except Exception as e:
                    self.errors.append(f"Craigslist {city} error: {e}")

        # Deduplicate
        seen = set()
        unique = [item for item in all_items if item.source_url not in seen and not seen.add(item.source_url)]

        self.items = unique
        return self.items

    async def _search_city(self, city: str, keyword: str, **kwargs) -> List[PCPart]:
        """Search a specific Craigslist city"""
        items = []

        params = {
            'query': keyword,
            'sort': 'date',
            'hasPic': '1',
        }

        if kwargs.get('max_price'):
            params['max_price'] = kwargs['max_price']
        if kwargs.get('min_price'):
            params['min_price'] = kwargs['min_price']

        url = f"https://{city}.craigslist.org/search/sss?{urlencode(params)}"

        try:
            soup = await self.fetch_html(url)
            listings = soup.select('.result-row, li.result-row')

            for listing in listings:
                part = self.parse_item(listing, soup)
                if part:
                    # Add city to location
                    part.location = f"{city}, {part.location or 'Unknown'}"
                    items.append(part)

        except Exception as e:
            self.errors.append(f"{city} search error: {e}")

        return items

    def extract_title(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        title_elem = item.select_one('.result-title, a.hdrlnk')
        if title_elem:
            return title_elem.get_text(strip=True)
        return None

    def extract_url(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        link = item.select_one('a.result-title, a.hdrlnk')
        if link and link.get('href'):
            return link['href']
        return None

    def extract_price(self, item: Any, soup: BeautifulSoup = None) -> Optional[float]:
        price_elem = item.select_one('.result-price')
        if price_elem:
            return clean_price(price_elem.get_text(strip=True))
        return None

    def extract_item_id(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        return item.get('data-pid')

    def extract_location(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        hood = item.select_one('.result-hood')
        if hood:
            return hood.get_text(strip=True).strip('()')
        return None

    def extract_thumbnail(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        img = item.select_one('img')
        if img:
            return img.get('src')

        # Craigslist uses data-ids
        data_ids = item.get('data-ids')
        if data_ids:
            img_id = data_ids.split(',')[0].split(':')[-1]
            return f"https://images.craigslist.org/{img_id}_300x300.jpg"

        return None


class FreeGeekScraper(BaseScraper):
    """
    Scraper for FreeGeek - Electronics recycler/refurbisher
    Great for finding working vintage parts at low prices
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.name = "freegeek"
        self.base_url = "https://www.freegeek.org"
        self.source_category = "recycler"

    async def scrape(self, keywords: List[str] = None, **kwargs) -> List[PCPart]:
        """Scrape FreeGeek store"""
        all_items = []

        # Try different store pages
        store_urls = [
            '/shop',
            '/store',
            '/products',
        ]

        for url_path in store_urls:
            try:
                url = f"{self.base_url}{url_path}"
                soup = await self.fetch_html(url)
                items = self._parse_products(soup)
                all_items.extend(items)
            except Exception as e:
                self.errors.append(f"FreeGeek error: {e}")

        # Filter by keywords if provided
        if keywords:
            filtered = []
            for item in all_items:
                text = f"{item.title} {item.description or ''}".lower()
                if any(kw.lower() in text for kw in keywords):
                    filtered.append(item)
            all_items = filtered

        self.items = all_items
        return self.items

    def _parse_products(self, soup: BeautifulSoup) -> List[PCPart]:
        """Parse FreeGeek product listings"""
        items = []

        listings = soup.select('.product, .item, .product-card')

        for listing in listings:
            try:
                part = self.parse_item(listing, soup)
                if part:
                    items.append(part)
            except Exception as e:
                self.errors.append(f"Parse error: {e}")

        return items

    def extract_title(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        title_elem = item.select_one('.product-title, .item-name, h3, h4')
        if title_elem:
            return title_elem.get_text(strip=True)
        return None

    def extract_url(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        link = item.select_one('a')
        if link and link.get('href'):
            return self.make_absolute_url(link['href'])
        return None

    def extract_price(self, item: Any, soup: BeautifulSoup = None) -> Optional[float]:
        price_elem = item.select_one('.price, .product-price')
        if price_elem:
            return clean_price(price_elem.get_text(strip=True))
        return None

    def extract_thumbnail(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        img = item.select_one('img')
        if img:
            return img.get('src')
        return None

    def extract_condition(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        # FreeGeek items are typically refurbished
        return 'refurbished'
