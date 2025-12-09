"""
Discount and Overstock Scrapers
Finds deals on PC parts from clearance, open box, and overstock sources
"""

import re
from typing import List, Any, Optional
from urllib.parse import urlencode

from bs4 import BeautifulSoup

from scrapers.base import BaseScraper
from utils.models import PCPart
from utils.helpers import clean_price


class WootScraper(BaseScraper):
    """
    Scraper for Woot.com - Amazon's daily deal site
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.name = "woot"
        self.base_url = "https://computers.woot.com"
        self.source_category = "overstock"

    async def scrape(self, keywords: List[str] = None, **kwargs) -> List[PCPart]:
        """Scrape Woot for computer deals"""
        all_items = []

        # Scrape main computer deals page
        try:
            soup = await self.fetch_html(self.base_url)
            items = self._parse_deals(soup)
            all_items.extend(items)
        except Exception as e:
            self.errors.append(f"Woot main page error: {e}")

        # Scrape all deals if available
        try:
            all_deals_url = f"{self.base_url}/plus/alldeals"
            soup = await self.fetch_html(all_deals_url)
            items = self._parse_deals(soup)
            all_items.extend(items)
        except Exception as e:
            self.errors.append(f"Woot all deals error: {e}")

        # Filter by keywords if provided
        if keywords:
            filtered = []
            for item in all_items:
                text = f"{item.title} {item.description or ''}".lower()
                if any(kw.lower() in text for kw in keywords):
                    filtered.append(item)
            all_items = filtered

        # Deduplicate
        seen = set()
        unique = [item for item in all_items if item.source_url not in seen and not seen.add(item.source_url)]

        self.items = unique
        return self.items

    def _parse_deals(self, soup: BeautifulSoup) -> List[PCPart]:
        """Parse Woot deals page"""
        items = []

        # Find deal items
        listings = soup.select('.deal-item, .item, article.offer')

        for listing in listings:
            try:
                part = self.parse_item(listing, soup)
                if part:
                    items.append(part)
            except Exception as e:
                self.errors.append(f"Parse error: {e}")

        return items

    def extract_title(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        title_elem = item.select_one('.deal-title, h2, h3, .title')
        if title_elem:
            return title_elem.get_text(strip=True)
        return None

    def extract_url(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        link = item.select_one('a[href*="woot.com"]')
        if link and link.get('href'):
            return link['href']
        # Try data attribute
        if item.get('data-url'):
            return item['data-url']
        return None

    def extract_price(self, item: Any, soup: BeautifulSoup = None) -> Optional[float]:
        price_elem = item.select_one('.price, .sale-price, .current-price')
        if price_elem:
            return clean_price(price_elem.get_text(strip=True))
        return None

    def extract_description(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        desc_elem = item.select_one('.deal-description, .description, p')
        if desc_elem:
            return desc_elem.get_text(strip=True)
        return None

    def extract_thumbnail(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        img = item.select_one('img')
        if img:
            return img.get('src') or img.get('data-src')
        return None

    def extract_condition(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        # Woot often has refurb or open box
        text = item.get_text().lower()
        if 'refurb' in text:
            return 'refurbished'
        if 'open box' in text:
            return 'open_box'
        return 'new'


class NeweggOpenBoxScraper(BaseScraper):
    """
    Scraper for Newegg Open Box and Clearance items
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.name = "newegg_openbox"
        self.base_url = "https://www.newegg.com"
        self.source_category = "open_box"

    async def scrape(self, keywords: List[str] = None, **kwargs) -> List[PCPart]:
        """Scrape Newegg for open box and clearance items"""
        all_items = []

        # Categories to check
        categories = [
            'Computer-Hardware',
            'Computer-Systems',
            'Networking',
            'Computer-Accessories',
        ]

        for category in categories:
            try:
                # Open box items
                url = f"{self.base_url}/{category}/Open-Box/SubCategory/ID-0"
                soup = await self.fetch_html(url)
                items = self._parse_results(soup)
                all_items.extend(items)

            except Exception as e:
                self.errors.append(f"Newegg {category} error: {e}")

        # Also search for specific keywords
        if keywords:
            for keyword in keywords[:10]:  # Limit to avoid rate limits
                try:
                    params = {
                        'N': '4016',  # Open box filter
                        'd': keyword,
                    }
                    url = f"{self.base_url}/p/pl?{urlencode(params)}"
                    soup = await self.fetch_html(url)
                    items = self._parse_results(soup)
                    all_items.extend(items)
                except Exception as e:
                    self.errors.append(f"Newegg search error: {e}")

        # Deduplicate
        seen = set()
        unique = [item for item in all_items if item.source_url not in seen and not seen.add(item.source_url)]

        self.items = unique
        return self.items

    def _parse_results(self, soup: BeautifulSoup) -> List[PCPart]:
        """Parse Newegg search results"""
        items = []

        listings = soup.select('.item-cell, .item-container, [data-product-id]')

        for listing in listings:
            try:
                # Skip sponsored items
                if listing.select_one('.item-sponsored'):
                    continue

                part = self.parse_item(listing, soup)
                if part:
                    items.append(part)
            except Exception as e:
                self.errors.append(f"Parse error: {e}")

        return items

    def extract_title(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        title_elem = item.select_one('.item-title, a.item-title')
        if title_elem:
            return title_elem.get_text(strip=True)
        return None

    def extract_url(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        link = item.select_one('a.item-title, a[href*="/p/"]')
        if link and link.get('href'):
            href = link['href']
            if href.startswith('//'):
                return f"https:{href}"
            return self.make_absolute_url(href)
        return None

    def extract_price(self, item: Any, soup: BeautifulSoup = None) -> Optional[float]:
        # Try current price
        price_elem = item.select_one('.price-current, .price-was-data')
        if price_elem:
            # Get strong element for dollars
            dollars = price_elem.select_one('strong')
            cents = price_elem.select_one('sup')

            if dollars:
                price_str = dollars.get_text(strip=True)
                if cents:
                    price_str += cents.get_text(strip=True)
                return clean_price(price_str)

        return None

    def extract_item_id(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        item_id = item.get('data-product-id')
        if item_id:
            return item_id

        url = self.extract_url(item, soup)
        if url:
            match = re.search(r'/p/([A-Z0-9]+)', url)
            if match:
                return match.group(1)
        return None

    def extract_condition(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        text = item.get_text().lower()
        if 'open box' in text:
            return 'open_box'
        if 'refurb' in text:
            return 'refurbished'
        return 'open_box'  # Default for this scraper

    def extract_thumbnail(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        img = item.select_one('img.item-img, img')
        if img:
            return img.get('src') or img.get('data-src')
        return None

    def extract_seller_rating(self, item: Any, soup: BeautifulSoup = None) -> Optional[float]:
        rating_elem = item.select_one('.item-rating-num, .rating')
        if rating_elem:
            text = rating_elem.get_text(strip=True)
            match = re.search(r'([\d.]+)', text)
            if match:
                return float(match.group(1))
        return None


class MicrocenterClearanceScraper(BaseScraper):
    """
    Scraper for Micro Center clearance items
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.name = "microcenter"
        self.base_url = "https://www.microcenter.com"
        self.source_category = "clearance"

    async def scrape(self, keywords: List[str] = None, **kwargs) -> List[PCPart]:
        """Scrape Micro Center clearance section"""
        all_items = []

        # Clearance categories
        clearance_urls = [
            '/search/search_results.aspx?N=4294966998',  # All clearance
            '/search/search_results.aspx?N=4294966998+4294818890',  # CPUs
            '/search/search_results.aspx?N=4294966998+4294820692',  # GPUs
            '/search/search_results.aspx?N=4294966998+4294818900',  # Motherboards
        ]

        for url_path in clearance_urls:
            try:
                url = f"{self.base_url}{url_path}"
                soup = await self.fetch_html(url)
                items = self._parse_results(soup)
                all_items.extend(items)
            except Exception as e:
                self.errors.append(f"Microcenter error: {e}")

        # Filter by keywords if provided
        if keywords:
            filtered = []
            for item in all_items:
                text = f"{item.title} {item.description or ''}".lower()
                if any(kw.lower() in text for kw in keywords):
                    filtered.append(item)
            all_items = filtered

        # Deduplicate
        seen = set()
        unique = [item for item in all_items if item.source_url not in seen and not seen.add(item.source_url)]

        self.items = unique
        return self.items

    def _parse_results(self, soup: BeautifulSoup) -> List[PCPart]:
        """Parse Micro Center search results"""
        items = []

        listings = soup.select('.product_wrapper, .product, li[data-sku]')

        for listing in listings:
            try:
                part = self.parse_item(listing, soup)
                if part:
                    items.append(part)
            except Exception as e:
                self.errors.append(f"Parse error: {e}")

        return items

    def extract_title(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        title_elem = item.select_one('h2 a, .pDescription a, .product-title')
        if title_elem:
            return title_elem.get_text(strip=True)
        return None

    def extract_url(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        link = item.select_one('h2 a, .pDescription a')
        if link and link.get('href'):
            return self.make_absolute_url(link['href'])
        return None

    def extract_price(self, item: Any, soup: BeautifulSoup = None) -> Optional[float]:
        # Look for the main price
        price_elem = item.select_one('[itemprop="price"], .price, .savings-price')
        if price_elem:
            # Try content attribute first
            price = price_elem.get('content')
            if price:
                return float(price)
            return clean_price(price_elem.get_text(strip=True))
        return None

    def extract_item_id(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        sku = item.get('data-sku')
        if sku:
            return sku

        url = self.extract_url(item, soup)
        if url:
            match = re.search(r'/product/(\d+)', url)
            if match:
                return match.group(1)
        return None

    def extract_location(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        # Micro Center is store-specific
        avail_elem = item.select_one('.stock, .availability')
        if avail_elem:
            return avail_elem.get_text(strip=True)
        return None

    def extract_thumbnail(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        img = item.select_one('img.SearchResultProductImage, img')
        if img:
            return img.get('src')
        return None

    def extract_condition(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        text = item.get_text().lower()
        if 'open box' in text:
            return 'open_box'
        if 'clearance' in text:
            return 'new'
        return 'new'


class AmazonWarehouseScraper(BaseScraper):
    """
    Scraper for Amazon Warehouse Deals
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.name = "amazon_warehouse"
        self.base_url = "https://www.amazon.com"
        self.source_category = "warehouse"

    async def scrape(self, keywords: List[str] = None, **kwargs) -> List[PCPart]:
        """Scrape Amazon Warehouse for deals"""
        if not keywords:
            keywords = ['computer components', 'graphics card', 'motherboard', 'cpu']

        all_items = []

        for keyword in keywords:
            try:
                # Search warehouse deals
                params = {
                    'k': keyword,
                    'i': 'warehouse-deals',
                    'rh': 'n:16225009011',  # Computers & Accessories
                }

                url = f"{self.base_url}/s?{urlencode(params)}"

                soup = await self.fetch_html(url)
                items = self._parse_results(soup)
                all_items.extend(items)

            except Exception as e:
                self.errors.append(f"Amazon search error: {e}")

        # Deduplicate
        seen = set()
        unique = [item for item in all_items if item.source_url not in seen and not seen.add(item.source_url)]

        self.items = unique
        return self.items

    def _parse_results(self, soup: BeautifulSoup) -> List[PCPart]:
        """Parse Amazon search results"""
        items = []

        listings = soup.select('[data-asin]:not([data-asin=""]), .s-result-item')

        for listing in listings:
            try:
                # Skip non-product items
                if not listing.get('data-asin'):
                    continue

                part = self.parse_item(listing, soup)
                if part:
                    items.append(part)

            except Exception as e:
                self.errors.append(f"Parse error: {e}")

        return items

    def extract_title(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        title_elem = item.select_one('h2 a span, .a-text-normal')
        if title_elem:
            return title_elem.get_text(strip=True)
        return None

    def extract_url(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        link = item.select_one('h2 a, a.a-link-normal')
        if link and link.get('href'):
            href = link['href']
            # Clean Amazon tracking params
            if '/dp/' in href:
                asin = re.search(r'/dp/([A-Z0-9]+)', href)
                if asin:
                    return f"{self.base_url}/dp/{asin.group(1)}"
            return self.make_absolute_url(href)
        return None

    def extract_price(self, item: Any, soup: BeautifulSoup = None) -> Optional[float]:
        price_elem = item.select_one('.a-price .a-offscreen, .a-price-whole')
        if price_elem:
            return clean_price(price_elem.get_text(strip=True))
        return None

    def extract_item_id(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        return item.get('data-asin')

    def extract_seller_rating(self, item: Any, soup: BeautifulSoup = None) -> Optional[float]:
        rating_elem = item.select_one('.a-icon-star-small, [aria-label*="stars"]')
        if rating_elem:
            label = rating_elem.get('aria-label', '')
            match = re.search(r'([\d.]+)', label)
            if match:
                return float(match.group(1))
        return None

    def extract_thumbnail(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        img = item.select_one('.s-image, img')
        if img:
            return img.get('src')
        return None

    def extract_condition(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        # Amazon Warehouse items are typically used/open box
        text = item.get_text().lower()
        if 'like new' in text:
            return 'open_box'
        if 'very good' in text or 'good' in text:
            return 'used'
        if 'acceptable' in text:
            return 'used'
        return 'open_box'
