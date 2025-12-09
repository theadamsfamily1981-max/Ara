"""
PC Parts Scrapers Package
"""

from scrapers.base import BaseScraper, MultiSourceScraper
from scrapers.ebay import EbayScraper, EbayVintageScraper
from scrapers.surplus import (
    GovPlanetScraper,
    PublicSurplusScraper,
    PropertyRoomScraper,
    LiquidationScraper
)
from scrapers.discount import (
    WootScraper,
    NeweggOpenBoxScraper,
    MicrocenterClearanceScraper,
    AmazonWarehouseScraper
)
from scrapers.marketplace import (
    MercariScraper,
    OfferUpScraper,
    CraigslistScraper,
    FreeGeekScraper
)

__all__ = [
    # Base
    'BaseScraper',
    'MultiSourceScraper',

    # eBay
    'EbayScraper',
    'EbayVintageScraper',

    # Surplus
    'GovPlanetScraper',
    'PublicSurplusScraper',
    'PropertyRoomScraper',
    'LiquidationScraper',

    # Discount
    'WootScraper',
    'NeweggOpenBoxScraper',
    'MicrocenterClearanceScraper',
    'AmazonWarehouseScraper',

    # Marketplace
    'MercariScraper',
    'OfferUpScraper',
    'CraigslistScraper',
    'FreeGeekScraper',
]


def get_all_scrapers(config=None):
    """
    Get instances of all available scrapers
    """
    return [
        # Primary sources
        EbayScraper(config),
        EbayVintageScraper(config),

        # Government/Surplus
        GovPlanetScraper(config),
        PublicSurplusScraper(config),
        PropertyRoomScraper(config),
        LiquidationScraper(config),

        # Discount
        WootScraper(config),
        NeweggOpenBoxScraper(config),
        MicrocenterClearanceScraper(config),
        AmazonWarehouseScraper(config),

        # Marketplaces
        MercariScraper(config),
        OfferUpScraper(config),
        CraigslistScraper(config),
        FreeGeekScraper(config),
    ]


def get_scraper_by_name(name: str, config=None):
    """
    Get a specific scraper by name
    """
    scrapers = {
        'ebay': EbayScraper,
        'ebay_vintage': EbayVintageScraper,
        'govplanet': GovPlanetScraper,
        'publicsurplus': PublicSurplusScraper,
        'propertyroom': PropertyRoomScraper,
        'liquidation': LiquidationScraper,
        'woot': WootScraper,
        'newegg': NeweggOpenBoxScraper,
        'microcenter': MicrocenterClearanceScraper,
        'amazon': AmazonWarehouseScraper,
        'mercari': MercariScraper,
        'offerup': OfferUpScraper,
        'craigslist': CraigslistScraper,
        'freegeek': FreeGeekScraper,
    }

    scraper_class = scrapers.get(name.lower())
    if scraper_class:
        return scraper_class(config)
    return None
