"""
Database models for PC Parts Scraper
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text,
    ForeignKey, Index, create_engine, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime

Base = declarative_base()


class PCPart(Base):
    """Main table for storing found PC parts"""
    __tablename__ = 'pc_parts'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Basic info
    title = Column(String(500), nullable=False)
    description = Column(Text)
    category = Column(String(100))  # cpu, gpu, motherboard, etc.

    # Pricing
    price = Column(Float)
    original_price = Column(Float)
    currency = Column(String(10), default='USD')

    # Source info
    source_name = Column(String(100), nullable=False)  # ebay, govplanet, etc.
    source_url = Column(String(1000), nullable=False)
    source_id = Column(String(200))  # Original item ID from source
    source_category = Column(String(100))  # auction, surplus, liquidation

    # Location
    location = Column(String(200))
    shipping_available = Column(Boolean, default=True)
    local_pickup = Column(Boolean, default=False)

    # Condition
    condition = Column(String(100))  # new, used, refurbished, for_parts
    condition_notes = Column(Text)

    # Images
    image_urls = Column(JSON)  # List of image URLs
    thumbnail_url = Column(String(1000))

    # Seller info
    seller_name = Column(String(200))
    seller_rating = Column(Float)
    seller_reviews = Column(Integer)

    # Auction specific
    is_auction = Column(Boolean, default=False)
    bid_count = Column(Integer)
    auction_end_time = Column(DateTime)

    # Matching & Classification
    matched_keywords = Column(JSON)  # Keywords that matched this item
    rarity_score = Column(Float)  # 0-100 score for how rare/valuable
    is_rare = Column(Boolean, default=False)

    # Timestamps
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow)

    # Status
    is_active = Column(Boolean, default=True)
    is_sold = Column(Boolean, default=False)

    # Price history relationship
    price_history = relationship("PriceHistory", back_populates="part", cascade="all, delete-orphan")

    # Indexes for fast querying
    __table_args__ = (
        Index('idx_category', 'category'),
        Index('idx_source', 'source_name'),
        Index('idx_price', 'price'),
        Index('idx_rarity', 'rarity_score'),
        Index('idx_is_rare', 'is_rare'),
        Index('idx_first_seen', 'first_seen'),
        Index('idx_source_url', 'source_url'),
    )

    def __repr__(self):
        return f"<PCPart {self.id}: {self.title[:50]}... ${self.price}>"

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'category': self.category,
            'price': self.price,
            'original_price': self.original_price,
            'currency': self.currency,
            'source_name': self.source_name,
            'source_url': self.source_url,
            'source_category': self.source_category,
            'location': self.location,
            'condition': self.condition,
            'thumbnail_url': self.thumbnail_url,
            'seller_name': self.seller_name,
            'is_auction': self.is_auction,
            'auction_end_time': str(self.auction_end_time) if self.auction_end_time else None,
            'rarity_score': self.rarity_score,
            'is_rare': self.is_rare,
            'first_seen': str(self.first_seen),
            'is_active': self.is_active,
        }


class PriceHistory(Base):
    """Track price changes over time"""
    __tablename__ = 'price_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    part_id = Column(Integer, ForeignKey('pc_parts.id'), nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    part = relationship("PCPart", back_populates="price_history")

    __table_args__ = (
        Index('idx_part_time', 'part_id', 'timestamp'),
    )


class SearchQuery(Base):
    """Store saved searches for monitoring"""
    __tablename__ = 'search_queries'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    keywords = Column(JSON)  # List of keywords
    categories = Column(JSON)  # List of categories
    max_price = Column(Float)
    min_rarity_score = Column(Float)
    sources = Column(JSON)  # List of sources to search
    alert_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_run = Column(DateTime)

    def __repr__(self):
        return f"<SearchQuery {self.id}: {self.name}>"


class ScrapingRun(Base):
    """Log scraping runs for debugging and stats"""
    __tablename__ = 'scraping_runs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_name = Column(String(100), nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    items_found = Column(Integer, default=0)
    items_new = Column(Integer, default=0)
    items_updated = Column(Integer, default=0)
    errors = Column(Integer, default=0)
    error_messages = Column(JSON)
    status = Column(String(50))  # running, completed, failed

    __table_args__ = (
        Index('idx_run_source', 'source_name'),
        Index('idx_run_start', 'start_time'),
    )


class Alert(Base):
    """Store sent alerts to avoid duplicates"""
    __tablename__ = 'alerts'

    id = Column(Integer, primary_key=True, autoincrement=True)
    part_id = Column(Integer, ForeignKey('pc_parts.id'))
    alert_type = Column(String(100))  # rare_item, price_drop, keyword_match
    channel = Column(String(50))  # discord, slack, email
    sent_at = Column(DateTime, default=datetime.utcnow)
    message = Column(Text)

    __table_args__ = (
        Index('idx_alert_part', 'part_id'),
        Index('idx_alert_type', 'alert_type'),
    )


def init_database(db_path: str = "data/pc_parts.db"):
    """Initialize the database and create tables"""
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return engine, Session
