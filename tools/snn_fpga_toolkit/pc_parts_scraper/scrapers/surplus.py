"""
Government and Commercial Surplus Scrapers
Finds forgotten PC parts in government auctions and surplus liquidation
"""

import re
from typing import List, Any, Optional
from datetime import datetime
from urllib.parse import urlencode

from bs4 import BeautifulSoup

from scrapers.base import BaseScraper
from utils.models import PCPart
from utils.helpers import clean_price, parse_auction_time


class GovPlanetScraper(BaseScraper):
    """
    Scraper for GovPlanet - Government surplus auctions
    Often has bulk IT equipment, servers, and networking gear
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.name = "govplanet"
        self.base_url = "https://www.govplanet.com"
        self.source_category = "government_surplus"

    async def scrape(self, keywords: List[str] = None, **kwargs) -> List[PCPart]:
        """Scrape GovPlanet for surplus IT equipment"""
        if not keywords:
            keywords = [
                'computer', 'server', 'workstation', 'laptop',
                'networking', 'storage', 'monitor', 'printer'
            ]

        all_items = []

        for keyword in keywords:
            try:
                items = await self._search(keyword, **kwargs)
                all_items.extend(items)
            except Exception as e:
                self.errors.append(f"GovPlanet search error for '{keyword}': {e}")

        # Deduplicate
        seen = set()
        unique = [item for item in all_items if item.source_url not in seen and not seen.add(item.source_url)]

        self.items = unique
        return self.items

    async def _search(self, keyword: str, max_pages: int = 3, **kwargs) -> List[PCPart]:
        """Search GovPlanet for a keyword"""
        items = []

        for page in range(1, max_pages + 1):
            params = {
                'keywords': keyword,
                'page': page,
                'category': 'Computers & Electronics',
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
        """Parse GovPlanet search results"""
        items = []

        # Find auction items
        listings = soup.select('.auction-item, .lot-card, [data-lot-id]')

        for listing in listings:
            try:
                part = self.parse_item(listing, soup)
                if part:
                    items.append(part)
            except Exception as e:
                self.errors.append(f"Parse error: {e}")

        return items

    def extract_title(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        title_elem = item.select_one('.lot-title, .item-title, h3, h4')
        if title_elem:
            return title_elem.get_text(strip=True)
        return None

    def extract_url(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        link = item.select_one('a[href*="/lot/"], a[href*="/item/"]')
        if link and link.get('href'):
            return self.make_absolute_url(link['href'])
        # Try the item itself
        if item.name == 'a' and item.get('href'):
            return self.make_absolute_url(item['href'])
        return None

    def extract_price(self, item: Any, soup: BeautifulSoup = None) -> Optional[float]:
        price_elem = item.select_one('.current-bid, .price, .bid-amount')
        if price_elem:
            return clean_price(price_elem.get_text(strip=True))
        return None

    def extract_item_id(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        # Try data attribute
        lot_id = item.get('data-lot-id')
        if lot_id:
            return lot_id

        # Try URL
        url = self.extract_url(item, soup)
        if url:
            match = re.search(r'/lot/(\d+)', url)
            if match:
                return match.group(1)
        return None

    def extract_location(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        loc_elem = item.select_one('.location, .item-location')
        if loc_elem:
            return loc_elem.get_text(strip=True)
        return None

    def extract_condition(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        cond_elem = item.select_one('.condition, .item-condition')
        if cond_elem:
            return cond_elem.get_text(strip=True)
        return 'used'  # Government surplus is typically used

    def extract_thumbnail(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        img = item.select_one('img')
        if img:
            return img.get('src') or img.get('data-src')
        return None

    def extract_is_auction(self, item: Any, soup: BeautifulSoup = None) -> bool:
        return True  # GovPlanet is auction-based

    def extract_auction_end(self, item: Any, soup: BeautifulSoup = None) -> Optional[datetime]:
        time_elem = item.select_one('.time-left, .countdown, .auction-end')
        if time_elem:
            return parse_auction_time(time_elem.get_text(strip=True))
        return None


class PublicSurplusScraper(BaseScraper):
    """
    Scraper for PublicSurplus.com - State/local government surplus
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.name = "publicsurplus"
        self.base_url = "https://www.publicsurplus.com"
        self.source_category = "government_surplus"

    async def scrape(self, keywords: List[str] = None, **kwargs) -> List[PCPart]:
        """Scrape PublicSurplus for IT equipment"""
        if not keywords:
            keywords = ['computer', 'server', 'laptop', 'electronics']

        all_items = []

        for keyword in keywords:
            try:
                items = await self._search(keyword)
                all_items.extend(items)
            except Exception as e:
                self.errors.append(f"PublicSurplus error: {e}")

        # Deduplicate
        seen = set()
        unique = [item for item in all_items if item.source_url not in seen and not seen.add(item.source_url)]

        self.items = unique
        return self.items

    async def _search(self, keyword: str, max_pages: int = 3) -> List[PCPart]:
        """Search PublicSurplus"""
        items = []

        for page in range(1, max_pages + 1):
            params = {
                'q': keyword,
                'catid': '4',  # Computers category
                'page': page,
            }

            url = f"{self.base_url}/sms/search?{urlencode(params)}"

            try:
                soup = await self.fetch_html(url)
                listings = soup.select('.search-result-item, .auction-item, tr.item-row')

                for listing in listings:
                    part = self.parse_item(listing, soup)
                    if part:
                        items.append(part)

                if len(listings) == 0:
                    break

            except Exception as e:
                self.errors.append(f"Page {page}: {e}")
                break

        return items

    def extract_title(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        title_elem = item.select_one('.title, .item-name, a')
        if title_elem:
            return title_elem.get_text(strip=True)
        return None

    def extract_url(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        link = item.select_one('a[href*="browse"], a[href*="view"]')
        if link and link.get('href'):
            return self.make_absolute_url(link['href'])
        return None

    def extract_price(self, item: Any, soup: BeautifulSoup = None) -> Optional[float]:
        price_elem = item.select_one('.bid, .price, .current-bid')
        if price_elem:
            return clean_price(price_elem.get_text(strip=True))
        return None

    def extract_location(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        loc_elem = item.select_one('.location, .city-state')
        if loc_elem:
            return loc_elem.get_text(strip=True)
        return None

    def extract_is_auction(self, item: Any, soup: BeautifulSoup = None) -> bool:
        return True


class PropertyRoomScraper(BaseScraper):
    """
    Scraper for PropertyRoom.com - Police/law enforcement surplus
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.name = "propertyroom"
        self.base_url = "https://www.propertyroom.com"
        self.source_category = "police_surplus"

    async def scrape(self, keywords: List[str] = None, **kwargs) -> List[PCPart]:
        """Scrape PropertyRoom for electronics"""
        if not keywords:
            keywords = ['computer', 'laptop', 'tablet', 'electronics', 'phone']

        all_items = []

        for keyword in keywords:
            try:
                url = f"{self.base_url}/search/{keyword.replace(' ', '+')}"
                soup = await self.fetch_html(url)

                listings = soup.select('.item-card, .product-item, .auction-listing')

                for listing in listings:
                    part = self.parse_item(listing, soup)
                    if part:
                        all_items.append(part)

            except Exception as e:
                self.errors.append(f"PropertyRoom error: {e}")

        # Deduplicate
        seen = set()
        unique = [item for item in all_items if item.source_url not in seen and not seen.add(item.source_url)]

        self.items = unique
        return self.items

    def extract_title(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        title_elem = item.select_one('.item-title, .product-name, h3, h4')
        if title_elem:
            return title_elem.get_text(strip=True)
        return None

    def extract_url(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        link = item.select_one('a')
        if link and link.get('href'):
            return self.make_absolute_url(link['href'])
        return None

    def extract_price(self, item: Any, soup: BeautifulSoup = None) -> Optional[float]:
        price_elem = item.select_one('.price, .bid, .current-price')
        if price_elem:
            return clean_price(price_elem.get_text(strip=True))
        return None

    def extract_thumbnail(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        img = item.select_one('img')
        if img:
            return img.get('src') or img.get('data-src')
        return None

    def extract_is_auction(self, item: Any, soup: BeautifulSoup = None) -> bool:
        return True


class LiquidationScraper(BaseScraper):
    """
    Scraper for Liquidation.com - Business liquidation and surplus
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.name = "liquidation"
        self.base_url = "https://www.liquidation.com"
        self.source_category = "liquidation"

    async def scrape(self, keywords: List[str] = None, **kwargs) -> List[PCPart]:
        """Scrape Liquidation.com for electronics"""
        if not keywords:
            keywords = ['computer', 'electronics', 'server', 'networking']

        all_items = []

        # Search in computers category
        for keyword in keywords:
            try:
                params = {
                    'q': keyword,
                    'category': 'computers',
                }

                url = f"{self.base_url}/auctioncal?{urlencode(params)}"
                soup = await self.fetch_html(url)

                listings = soup.select('.auction-card, .lot-item, .product-card')

                for listing in listings:
                    part = self.parse_item(listing, soup)
                    if part:
                        all_items.append(part)

            except Exception as e:
                self.errors.append(f"Liquidation.com error: {e}")

        # Deduplicate
        seen = set()
        unique = [item for item in all_items if item.source_url not in seen and not seen.add(item.source_url)]

        self.items = unique
        return self.items

    def extract_title(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        title_elem = item.select_one('.auction-title, .lot-title, h3')
        if title_elem:
            return title_elem.get_text(strip=True)
        return None

    def extract_url(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        link = item.select_one('a[href*="auction"]')
        if link and link.get('href'):
            return self.make_absolute_url(link['href'])
        return None

    def extract_price(self, item: Any, soup: BeautifulSoup = None) -> Optional[float]:
        price_elem = item.select_one('.current-bid, .price')
        if price_elem:
            return clean_price(price_elem.get_text(strip=True))
        return None

    def extract_location(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        loc_elem = item.select_one('.location, .warehouse')
        if loc_elem:
            return loc_elem.get_text(strip=True)
        return None

    def extract_is_auction(self, item: Any, soup: BeautifulSoup = None) -> bool:
        return True

    def extract_condition(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        cond_elem = item.select_one('.condition, .manifest-type')
        if cond_elem:
            return cond_elem.get_text(strip=True)
        return 'mixed'  # Liquidation lots are usually mixed condition
