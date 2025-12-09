"""
eBay Scraper - Finds rare/vintage PC parts on eBay
"""

import re
from typing import List, Any, Optional
from datetime import datetime
from urllib.parse import urlencode, quote_plus

from bs4 import BeautifulSoup

from scrapers.base import BaseScraper
from utils.models import PCPart
from utils.helpers import clean_price, parse_auction_time


class EbayScraper(BaseScraper):
    """
    Scraper for eBay listings - supports both auction and buy-it-now
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.name = "ebay"
        self.base_url = "https://www.ebay.com"
        self.source_category = "auction"
        self.api_base = "https://www.ebay.com/sch/i.html"

    async def scrape(self, keywords: List[str] = None, **kwargs) -> List[PCPart]:
        """Scrape eBay for PC parts"""
        if not keywords:
            # Use rare keywords from config
            keywords = self.config.get('rare_part_keywords', [])[:20]

        all_items = []

        # Search for each keyword
        for keyword in keywords:
            try:
                items = await self._search_keyword(keyword, **kwargs)
                all_items.extend(items)
            except Exception as e:
                self.errors.append(f"Error searching '{keyword}': {e}")

        # Deduplicate by URL
        seen_urls = set()
        unique_items = []
        for item in all_items:
            if item.source_url not in seen_urls:
                seen_urls.add(item.source_url)
                unique_items.append(item)

        self.items = unique_items
        return self.items

    async def _search_keyword(
        self,
        keyword: str,
        category: str = "175673",  # Computer Components
        max_pages: int = 3,
        sort: str = "newly_listed",
        **kwargs
    ) -> List[PCPart]:
        """Search eBay for a specific keyword"""
        items = []

        # Build search URL
        params = {
            '_nkw': keyword,
            '_sacat': category,
            '_sop': self._get_sort_code(sort),
            'LH_TitleDesc': '1',  # Search title and description
            '_ipg': '100',  # Items per page
        }

        # Add filters
        if kwargs.get('max_price'):
            params['_udhi'] = kwargs['max_price']
        if kwargs.get('min_price'):
            params['_udlo'] = kwargs['min_price']
        if kwargs.get('condition'):
            params['LH_ItemCondition'] = kwargs['condition']
        if kwargs.get('auction_only'):
            params['LH_Auction'] = '1'
        if kwargs.get('buy_now_only'):
            params['LH_BIN'] = '1'

        for page in range(1, max_pages + 1):
            params['_pgn'] = page
            url = f"{self.api_base}?{urlencode(params)}"

            try:
                soup = await self.fetch_html(url)
                page_items = self._parse_search_results(soup)

                if not page_items:
                    break  # No more results

                items.extend(page_items)

            except Exception as e:
                self.errors.append(f"Error on page {page}: {e}")
                break

        return items

    def _get_sort_code(self, sort: str) -> str:
        """Convert sort name to eBay sort code"""
        sort_codes = {
            'newly_listed': '10',
            'ending_soonest': '1',
            'price_low': '15',
            'price_high': '16',
            'best_match': '12',
        }
        return sort_codes.get(sort, '10')

    def _parse_search_results(self, soup: BeautifulSoup) -> List[PCPart]:
        """Parse eBay search results page"""
        items = []

        # Find all listing items
        listings = soup.select('li.s-item')

        for listing in listings:
            try:
                # Skip ads/promoted
                if listing.select_one('.s-item__ad-badge'):
                    continue

                part = self.parse_item(listing, soup)
                if part:
                    items.append(part)

            except Exception as e:
                self.errors.append(f"Error parsing listing: {e}")

        return items

    def extract_title(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        title_elem = item.select_one('.s-item__title')
        if title_elem:
            # Remove "New Listing" prefix
            title = title_elem.get_text(strip=True)
            title = re.sub(r'^New Listing\s*', '', title)
            return title
        return None

    def extract_url(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        link = item.select_one('.s-item__link')
        if link and link.get('href'):
            url = link['href']
            # Remove tracking parameters
            url = re.sub(r'\?.*$', '', url)
            return url
        return None

    def extract_price(self, item: Any, soup: BeautifulSoup = None) -> Optional[float]:
        price_elem = item.select_one('.s-item__price')
        if price_elem:
            price_text = price_elem.get_text(strip=True)
            # Handle price ranges
            if 'to' in price_text.lower():
                price_text = price_text.split('to')[0]
            return clean_price(price_text)
        return None

    def extract_item_id(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        url = self.extract_url(item, soup)
        if url:
            match = re.search(r'/itm/(\d+)', url)
            if match:
                return match.group(1)
        return None

    def extract_location(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        loc_elem = item.select_one('.s-item__location')
        if loc_elem:
            loc = loc_elem.get_text(strip=True)
            return re.sub(r'^from\s+', '', loc, flags=re.IGNORECASE)
        return None

    def extract_shipping(self, item: Any, soup: BeautifulSoup = None) -> bool:
        shipping_elem = item.select_one('.s-item__shipping')
        if shipping_elem:
            text = shipping_elem.get_text(strip=True).lower()
            return 'local pickup' not in text
        return True

    def extract_condition(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        cond_elem = item.select_one('.SECONDARY_INFO')
        if cond_elem:
            return cond_elem.get_text(strip=True)
        return None

    def extract_images(self, item: Any, soup: BeautifulSoup = None) -> List[str]:
        images = []
        img_elem = item.select_one('.s-item__image-img')
        if img_elem and img_elem.get('src'):
            images.append(img_elem['src'])
        return images

    def extract_thumbnail(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        img_elem = item.select_one('.s-item__image-img')
        if img_elem and img_elem.get('src'):
            return img_elem['src']
        return None

    def extract_seller(self, item: Any, soup: BeautifulSoup = None) -> Optional[str]:
        seller_elem = item.select_one('.s-item__seller-info-text')
        if seller_elem:
            return seller_elem.get_text(strip=True)
        return None

    def extract_is_auction(self, item: Any, soup: BeautifulSoup = None) -> bool:
        # Check for bid count or auction-specific elements
        bid_elem = item.select_one('.s-item__bids')
        return bid_elem is not None

    def extract_bid_count(self, item: Any, soup: BeautifulSoup = None) -> Optional[int]:
        bid_elem = item.select_one('.s-item__bids')
        if bid_elem:
            bid_text = bid_elem.get_text(strip=True)
            match = re.search(r'(\d+)', bid_text)
            if match:
                return int(match.group(1))
        return None

    def extract_auction_end(self, item: Any, soup: BeautifulSoup = None) -> Optional[datetime]:
        time_elem = item.select_one('.s-item__time-left')
        if time_elem:
            return parse_auction_time(time_elem.get_text(strip=True))
        return None


class EbayVintageScraper(EbayScraper):
    """
    Specialized eBay scraper focused on vintage/collectible PC parts
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.name = "ebay_vintage"

    async def scrape(self, keywords: List[str] = None, **kwargs) -> List[PCPart]:
        """Scrape eBay specifically for vintage items"""
        # Vintage-specific categories
        vintage_categories = {
            '175673': 'Computer Components',
            '162': 'Vintage Computing',
            '170083': 'Computer Memory',
            '27386': 'Graphics/Video Cards',
        }

        # Vintage-focused keywords
        if not keywords:
            keywords = [
                '3dfx voodoo',
                'voodoo graphics',
                'vintage gpu',
                'isa sound card',
                'sound blaster awe',
                'gravis ultrasound',
                'pentium pro',
                'socket 7',
                'agp graphics',
                'vintage motherboard',
                'rdram rambus',
                'edo simm',
                'scsi controller',
                'vintage server',
                'engineering sample cpu',
            ]

        all_items = []

        for keyword in keywords:
            for cat_id in vintage_categories.keys():
                try:
                    items = await self._search_keyword(
                        keyword,
                        category=cat_id,
                        max_pages=2,
                        sort='newly_listed'
                    )
                    all_items.extend(items)
                except Exception as e:
                    self.errors.append(f"Error: {keyword} in {cat_id}: {e}")

        # Deduplicate
        seen = set()
        unique = []
        for item in all_items:
            if item.source_url not in seen:
                seen.add(item.source_url)
                unique.append(item)

        self.items = unique
        return self.items
