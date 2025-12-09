"""
Database Manager for PC Parts Scraper
Handles storage, queries, and data management
"""

import os
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy import create_engine, func, and_, or_, desc
from sqlalchemy.orm import sessionmaker, Session

from utils.models import Base, PCPart, PriceHistory, SearchQuery, ScrapingRun, Alert


class DatabaseManager:
    """
    Manages all database operations for the PC Parts Scraper
    """

    def __init__(self, db_path: str = "data/pc_parts.db"):
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        """Get a new database session"""
        return self.Session()

    # ===== Part Operations =====

    def save_part(self, part: PCPart) -> PCPart:
        """Save or update a part in the database"""
        with self.get_session() as session:
            # Check if part already exists
            existing = session.query(PCPart).filter(
                PCPart.source_url == part.source_url
            ).first()

            if existing:
                # Update existing part
                self._update_part(existing, part)
                session.commit()
                session.refresh(existing)
                return existing
            else:
                # Add new part
                session.add(part)
                session.commit()
                session.refresh(part)
                return part

    def _update_part(self, existing: PCPart, new: PCPart):
        """Update existing part with new data"""
        # Track price change
        if existing.price != new.price and new.price is not None:
            price_record = PriceHistory(
                part_id=existing.id,
                price=new.price
            )
            existing.price_history.append(price_record)

        # Update fields
        existing.price = new.price or existing.price
        existing.title = new.title or existing.title
        existing.description = new.description or existing.description
        existing.condition = new.condition or existing.condition
        existing.is_active = True
        existing.last_seen = datetime.utcnow()
        existing.last_updated = datetime.utcnow()
        existing.bid_count = new.bid_count
        existing.auction_end_time = new.auction_end_time

        # Update rarity if higher
        if new.rarity_score > existing.rarity_score:
            existing.rarity_score = new.rarity_score
            existing.is_rare = new.is_rare
            existing.matched_keywords = new.matched_keywords

    def save_parts_bulk(self, parts: List[PCPart]) -> Dict[str, int]:
        """Save multiple parts efficiently"""
        stats = {'new': 0, 'updated': 0, 'errors': 0}

        with self.get_session() as session:
            for part in parts:
                try:
                    existing = session.query(PCPart).filter(
                        PCPart.source_url == part.source_url
                    ).first()

                    if existing:
                        self._update_part(existing, part)
                        stats['updated'] += 1
                    else:
                        session.add(part)
                        stats['new'] += 1

                except Exception as e:
                    stats['errors'] += 1

            session.commit()

        return stats

    def get_part_by_id(self, part_id: int) -> Optional[PCPart]:
        """Get a specific part by ID"""
        with self.get_session() as session:
            return session.query(PCPart).filter(PCPart.id == part_id).first()

    def search_parts(
        self,
        keywords: List[str] = None,
        categories: List[str] = None,
        sources: List[str] = None,
        min_price: float = None,
        max_price: float = None,
        min_rarity: float = None,
        condition: str = None,
        is_rare: bool = None,
        is_auction: bool = None,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = 'rarity_score',
        sort_order: str = 'desc'
    ) -> List[PCPart]:
        """Search for parts with various filters"""
        with self.get_session() as session:
            query = session.query(PCPart).filter(PCPart.is_active == True)

            # Apply filters
            if keywords:
                keyword_filters = []
                for kw in keywords:
                    kw_filter = or_(
                        PCPart.title.ilike(f'%{kw}%'),
                        PCPart.description.ilike(f'%{kw}%')
                    )
                    keyword_filters.append(kw_filter)
                query = query.filter(or_(*keyword_filters))

            if categories:
                query = query.filter(PCPart.category.in_(categories))

            if sources:
                query = query.filter(PCPart.source_name.in_(sources))

            if min_price is not None:
                query = query.filter(PCPart.price >= min_price)

            if max_price is not None:
                query = query.filter(PCPart.price <= max_price)

            if min_rarity is not None:
                query = query.filter(PCPart.rarity_score >= min_rarity)

            if condition:
                query = query.filter(PCPart.condition == condition)

            if is_rare is not None:
                query = query.filter(PCPart.is_rare == is_rare)

            if is_auction is not None:
                query = query.filter(PCPart.is_auction == is_auction)

            # Apply sorting
            sort_column = getattr(PCPart, sort_by, PCPart.rarity_score)
            if sort_order == 'desc':
                query = query.order_by(desc(sort_column))
            else:
                query = query.order_by(sort_column)

            # Apply pagination
            query = query.offset(offset).limit(limit)

            return query.all()

    def get_rare_parts(self, min_score: float = 50, limit: int = 50) -> List[PCPart]:
        """Get rare parts sorted by rarity score"""
        return self.search_parts(
            min_rarity=min_score,
            limit=limit,
            sort_by='rarity_score',
            sort_order='desc'
        )

    def get_recent_parts(self, hours: int = 24, limit: int = 100) -> List[PCPart]:
        """Get recently discovered parts"""
        with self.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            return session.query(PCPart).filter(
                PCPart.first_seen >= cutoff,
                PCPart.is_active == True
            ).order_by(desc(PCPart.first_seen)).limit(limit).all()

    def get_price_drops(self, min_drop_percent: float = 10) -> List[Dict[str, Any]]:
        """Find parts with significant price drops"""
        with self.get_session() as session:
            results = []

            # Get parts with price history
            parts = session.query(PCPart).filter(
                PCPart.is_active == True
            ).all()

            for part in parts:
                if len(part.price_history) >= 2:
                    # Get previous and current price
                    history = sorted(part.price_history, key=lambda x: x.timestamp)
                    prev_price = history[-2].price
                    curr_price = part.price

                    if prev_price > 0 and curr_price:
                        drop_percent = ((prev_price - curr_price) / prev_price) * 100

                        if drop_percent >= min_drop_percent:
                            results.append({
                                'part': part,
                                'previous_price': prev_price,
                                'current_price': curr_price,
                                'drop_percent': drop_percent
                            })

            return sorted(results, key=lambda x: x['drop_percent'], reverse=True)

    def get_ending_auctions(self, hours: int = 24) -> List[PCPart]:
        """Get auctions ending soon"""
        with self.get_session() as session:
            cutoff = datetime.utcnow() + timedelta(hours=hours)
            return session.query(PCPart).filter(
                PCPart.is_auction == True,
                PCPart.auction_end_time <= cutoff,
                PCPart.auction_end_time >= datetime.utcnow(),
                PCPart.is_active == True
            ).order_by(PCPart.auction_end_time).all()

    def mark_inactive(self, source_url: str):
        """Mark a part as no longer available"""
        with self.get_session() as session:
            part = session.query(PCPart).filter(
                PCPart.source_url == source_url
            ).first()
            if part:
                part.is_active = False
                part.is_sold = True
                session.commit()

    # ===== Statistics =====

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.get_session() as session:
            total_parts = session.query(func.count(PCPart.id)).scalar()
            active_parts = session.query(func.count(PCPart.id)).filter(
                PCPart.is_active == True
            ).scalar()
            rare_parts = session.query(func.count(PCPart.id)).filter(
                PCPart.is_rare == True
            ).scalar()

            # By source
            by_source = dict(
                session.query(
                    PCPart.source_name,
                    func.count(PCPart.id)
                ).group_by(PCPart.source_name).all()
            )

            # By category
            by_category = dict(
                session.query(
                    PCPart.category,
                    func.count(PCPart.id)
                ).group_by(PCPart.category).all()
            )

            # Price stats
            price_stats = session.query(
                func.avg(PCPart.price),
                func.min(PCPart.price),
                func.max(PCPart.price)
            ).filter(PCPart.price > 0).first()

            return {
                'total_parts': total_parts,
                'active_parts': active_parts,
                'rare_parts': rare_parts,
                'by_source': by_source,
                'by_category': by_category,
                'avg_price': price_stats[0] if price_stats else 0,
                'min_price': price_stats[1] if price_stats else 0,
                'max_price': price_stats[2] if price_stats else 0,
            }

    # ===== Saved Searches =====

    def save_search(self, search: SearchQuery) -> SearchQuery:
        """Save a search query for monitoring"""
        with self.get_session() as session:
            session.add(search)
            session.commit()
            session.refresh(search)
            return search

    def get_saved_searches(self) -> List[SearchQuery]:
        """Get all saved searches"""
        with self.get_session() as session:
            return session.query(SearchQuery).all()

    def delete_search(self, search_id: int):
        """Delete a saved search"""
        with self.get_session() as session:
            session.query(SearchQuery).filter(
                SearchQuery.id == search_id
            ).delete()
            session.commit()

    # ===== Scraping Runs =====

    def save_run(self, run: ScrapingRun) -> ScrapingRun:
        """Save a scraping run record"""
        with self.get_session() as session:
            session.add(run)
            session.commit()
            session.refresh(run)
            return run

    def get_recent_runs(self, limit: int = 20) -> List[ScrapingRun]:
        """Get recent scraping runs"""
        with self.get_session() as session:
            return session.query(ScrapingRun).order_by(
                desc(ScrapingRun.start_time)
            ).limit(limit).all()

    # ===== Alerts =====

    def save_alert(self, alert: Alert) -> Alert:
        """Save an alert record"""
        with self.get_session() as session:
            session.add(alert)
            session.commit()
            return alert

    def has_alert_been_sent(self, part_id: int, alert_type: str) -> bool:
        """Check if an alert has already been sent for a part"""
        with self.get_session() as session:
            count = session.query(func.count(Alert.id)).filter(
                Alert.part_id == part_id,
                Alert.alert_type == alert_type
            ).scalar()
            return count > 0

    # ===== Cleanup =====

    def cleanup_old_data(self, days: int = 90):
        """Remove old inactive parts and records"""
        with self.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(days=days)

            # Remove old inactive parts
            session.query(PCPart).filter(
                PCPart.is_active == False,
                PCPart.last_updated < cutoff
            ).delete()

            # Remove old scraping runs
            session.query(ScrapingRun).filter(
                ScrapingRun.start_time < cutoff
            ).delete()

            session.commit()
