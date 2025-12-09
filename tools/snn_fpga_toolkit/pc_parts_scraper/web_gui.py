"""
Web GUI for PC Parts Scraper using Streamlit
"""

import streamlit as st
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

from utils.database import DatabaseManager
from utils.helpers import load_config, format_price
from scrapers import get_all_scrapers, get_scraper_by_name, MultiSourceScraper
from alerts.notifier import AlertNotifier

# Page config
st.set_page_config(
    page_title="PC Parts Scraper",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .rare-badge {
        background-color: #ffd700;
        color: #000;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .ultra-rare-badge {
        background-color: #ff4500;
        color: #fff;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
@st.cache_resource
def get_db():
    return DatabaseManager()

db = get_db()
config = load_config()

# Sidebar navigation
st.sidebar.title("🔍 PC Parts Scraper")
page = st.sidebar.radio("Navigation", [
    "Dashboard",
    "Run Scraper",
    "Search Parts",
    "Rare Finds",
    "Recent Items",
    "Ending Auctions",
    "Settings"
])

# ===================================
# DASHBOARD PAGE
# ===================================
if page == "Dashboard":
    st.markdown('<div class="main-header">📊 Dashboard</div>', unsafe_allow_html=True)

    # Get stats
    stats = db.get_stats()

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Parts", f"{stats['total_parts']:,}")
    with col2:
        st.metric("Active Parts", f"{stats['active_parts']:,}")
    with col3:
        st.metric("Rare Items", f"{stats['rare_parts']:,}")
    with col4:
        avg_price = stats['avg_price'] or 0
        st.metric("Avg Price", format_price(avg_price))

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Parts by Source")
        if stats['by_source']:
            df_source = pd.DataFrame(list(stats['by_source'].items()),
                                    columns=['Source', 'Count'])
            fig = px.bar(df_source, x='Source', y='Count',
                        color='Count', color_continuous_scale='Blues')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet. Run a scrape to populate!")

    with col2:
        st.subheader("Parts by Category")
        if stats['by_category']:
            df_cat = pd.DataFrame(list(stats['by_category'].items()),
                                 columns=['Category', 'Count'])
            fig = px.pie(df_cat, values='Count', names='Category', hole=0.4)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet. Run a scrape to populate!")

    # Recent activity
    st.subheader("Recent Activity")
    recent_runs = db.get_recent_runs(limit=10)

    if recent_runs:
        runs_data = []
        for run in recent_runs:
            duration = "N/A"
            if run.end_time:
                duration = str(run.end_time - run.start_time).split('.')[0]

            runs_data.append({
                'Source': run.source_name,
                'Start Time': run.start_time.strftime('%Y-%m-%d %H:%M'),
                'Duration': duration,
                'Items Found': run.items_found,
                'New': run.items_new,
                'Errors': run.errors,
                'Status': run.status
            })

        df_runs = pd.DataFrame(runs_data)
        st.dataframe(df_runs, use_container_width=True)
    else:
        st.info("No scraping history yet.")

# ===================================
# RUN SCRAPER PAGE
# ===================================
elif page == "Run Scraper":
    st.markdown('<div class="main-header">🚀 Run Scraper</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Select Sources")

        # Quick presets
        preset = st.radio("Quick Select:",
                         ["All Sources", "SOMs/FPGAs Only", "GPUs Only", "Custom"],
                         horizontal=True)

        if preset == "Custom":
            # Source selection
            sources_to_scrape = st.multiselect(
                "Choose sources:",
                ["ebay", "ebay_vintage", "govplanet", "publicsurplus",
                 "liquidation", "woot", "newegg", "mercari", "offerup"],
                default=["ebay", "govplanet"]
            )
        else:
            sources_to_scrape = None  # Will use all or filtered

    with col2:
        st.subheader("Options")

        max_price = st.number_input("Max Price ($)", min_value=0, value=500, step=50)
        send_alerts = st.checkbox("Send Alerts", value=True)
        max_items = st.number_input("Max Items per Source", min_value=10,
                                    max_value=1000, value=100, step=10)

    # Keywords
    st.subheader("Keywords (Optional)")
    keywords_input = st.text_area(
        "Enter keywords (one per line):",
        placeholder="zu3eg\ntrenz te0803\nalveo u250\nvcu1525",
        height=100
    )

    keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]

    # Run button
    if st.button("🚀 Start Scraping", type="primary", use_container_width=True):

        # Determine which scrapers to use
        if preset == "SOMs/FPGAs Only":
            keywords = keywords or [
                "zu3eg", "zu7ev", "trenz", "alinx", "ultra96", "kria k26",
                "alveo", "virtex", "stratix", "fpga"
            ]
        elif preset == "GPUs Only":
            keywords = keywords or [
                "tesla", "quadro", "titan", "instinct", "a100", "h100"
            ]

        # Get scrapers
        if sources_to_scrape:
            scrapers = [get_scraper_by_name(s, config) for s in sources_to_scrape
                       if get_scraper_by_name(s, config)]
        else:
            scrapers = get_all_scrapers(config)

        if not scrapers:
            st.error("No valid scrapers selected!")
        else:
            st.info(f"Starting {len(scrapers)} scrapers...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Run scraping
            async def run_scrape():
                multi = MultiSourceScraper(scrapers, config)
                items = await multi.scrape_all(
                    keywords=keywords,
                    max_concurrent=3
                )
                return items, multi.runs

            # Execute
            with st.spinner("Scraping in progress..."):
                try:
                    items, runs = asyncio.run(run_scrape())

                    # Save to database
                    status_text.text("Saving to database...")
                    stats_save = db.save_parts_bulk(items)

                    # Send alerts
                    if send_alerts and items:
                        status_text.text("Checking for alerts...")
                        notifier = AlertNotifier(config, db)
                        asyncio.run(notifier.check_and_alert(items))

                    progress_bar.progress(100)

                    # Show results
                    st.success("✅ Scraping complete!")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Items Found", len(items))
                    with col2:
                        st.metric("New Items", stats_save['new'])
                    with col3:
                        st.metric("Updated Items", stats_save['updated'])

                    # Show rare items
                    rare_items = [i for i in items if i.is_rare]
                    if rare_items:
                        st.subheader(f"🌟 Found {len(rare_items)} Rare Items!")

                        rare_data = []
                        for item in sorted(rare_items, key=lambda x: x.rarity_score, reverse=True)[:20]:
                            rare_data.append({
                                'Title': item.title[:60] + '...' if len(item.title) > 60 else item.title,
                                'Price': format_price(item.price) if item.price else 'N/A',
                                'Rarity': f"{item.rarity_score:.0f}",
                                'Source': item.source_name,
                                'Category': item.category or 'N/A'
                            })

                        df_rare = pd.DataFrame(rare_data)
                        st.dataframe(df_rare, use_container_width=True)

                except Exception as e:
                    st.error(f"Error during scraping: {e}")
                    st.exception(e)

# ===================================
# SEARCH PAGE
# ===================================
elif page == "Search Parts":
    st.markdown('<div class="main-header">🔍 Search Parts</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        search_keywords = st.text_input("Keywords:", placeholder="e.g., zu3eg, alveo")

    with col2:
        category_filter = st.selectbox("Category:",
            ["All"] + config['pc_parts_categories'])

    with col3:
        source_filter = st.selectbox("Source:",
            ["All", "ebay", "govplanet", "mercari", "liquidation", "woot"])

    col1, col2, col3 = st.columns(3)

    with col1:
        max_price_filter = st.number_input("Max Price ($):", min_value=0, value=1000)

    with col2:
        min_rarity_filter = st.slider("Min Rarity Score:", 0, 100, 0)

    with col3:
        show_rare_only = st.checkbox("Rare Items Only")

    # Search button
    if st.button("Search", type="primary"):

        keywords_list = [k.strip() for k in search_keywords.split(',') if k.strip()] if search_keywords else None
        categories_list = [category_filter] if category_filter != "All" else None
        sources_list = [source_filter] if source_filter != "All" else None

        results = db.search_parts(
            keywords=keywords_list,
            categories=categories_list,
            sources=sources_list,
            max_price=max_price_filter,
            min_rarity=min_rarity_filter,
            is_rare=show_rare_only if show_rare_only else None,
            limit=100
        )

        st.subheader(f"Results: {len(results)} items")

        if results:
            results_data = []
            for item in results:
                rarity_badge = ""
                if item.rarity_score >= 80:
                    rarity_badge = "🔥 ULTRA"
                elif item.rarity_score >= 60:
                    rarity_badge = "⭐ RARE"

                results_data.append({
                    'ID': item.id,
                    'Title': item.title[:70] + '...' if len(item.title) > 70 else item.title,
                    'Price': format_price(item.price) if item.price else 'N/A',
                    'Rarity': f"{item.rarity_score:.0f} {rarity_badge}",
                    'Category': item.category or 'N/A',
                    'Source': item.source_name,
                    'Condition': item.condition or 'N/A',
                    'URL': item.source_url
                })

            df_results = pd.DataFrame(results_data)

            # Display with click-to-expand
            st.dataframe(
                df_results.drop('URL', axis=1),
                use_container_width=True,
                hide_index=True
            )

            # Item details
            st.subheader("View Item Details")
            item_id = st.number_input("Enter Item ID:", min_value=1, step=1)

            if st.button("View Details"):
                item = db.get_part_by_id(item_id)
                if item:
                    st.markdown(f"### {item.title}")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"**Price:** {format_price(item.price) if item.price else 'N/A'}")
                        st.markdown(f"**Category:** {item.category or 'N/A'}")
                        st.markdown(f"**Condition:** {item.condition or 'N/A'}")
                        st.markdown(f"**Source:** {item.source_name}")
                        st.markdown(f"**Rarity Score:** {item.rarity_score:.1f}/100")

                    with col2:
                        st.markdown(f"**Location:** {item.location or 'N/A'}")
                        st.markdown(f"**Shipping:** {'Yes' if item.shipping_available else 'No'}")
                        st.markdown(f"**Auction:** {'Yes' if item.is_auction else 'No'}")
                        st.markdown(f"**First Seen:** {item.first_seen.strftime('%Y-%m-%d')}")

                    if item.description:
                        st.markdown("**Description:**")
                        st.text(item.description[:500] + '...' if len(item.description) > 500 else item.description)

                    st.markdown(f"**[Open in Browser]({item.source_url})**")
                else:
                    st.error(f"Item {item_id} not found")
        else:
            st.info("No results found. Try adjusting your filters.")

# ===================================
# RARE FINDS PAGE
# ===================================
elif page == "Rare Finds":
    st.markdown('<div class="main-header">⭐ Rare Finds</div>', unsafe_allow_html=True)

    min_score = st.slider("Minimum Rarity Score:", 50, 100, 70)

    rare_items = db.get_rare_parts(min_score=min_score, limit=50)

    st.subheader(f"Found {len(rare_items)} rare items (score ≥ {min_score})")

    if rare_items:
        for item in rare_items:
            with st.expander(f"{'🔥' if item.rarity_score >= 80 else '⭐'} {item.title[:80]}"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Price", format_price(item.price) if item.price else 'N/A')
                with col2:
                    st.metric("Rarity Score", f"{item.rarity_score:.0f}/100")
                with col3:
                    st.metric("Category", item.category or 'N/A')

                st.markdown(f"**Source:** {item.source_name}")
                st.markdown(f"**Condition:** {item.condition or 'Unknown'}")

                if item.matched_keywords:
                    st.markdown(f"**Matched Keywords:** {', '.join(item.matched_keywords[:10])}")

                st.markdown(f"**[View Item]({item.source_url})**")
    else:
        st.info("No rare items found yet. Run a scrape to discover gems!")

# ===================================
# RECENT ITEMS PAGE
# ===================================
elif page == "Recent Items":
    st.markdown('<div class="main-header">🆕 Recent Items</div>', unsafe_allow_html=True)

    hours = st.slider("Show items from last X hours:", 1, 168, 24)

    recent = db.get_recent_parts(hours=hours, limit=100)

    st.subheader(f"Found {len(recent)} items in the last {hours} hours")

    if recent:
        recent_data = []
        for item in recent:
            recent_data.append({
                'ID': item.id,
                'Title': item.title[:60] + '...' if len(item.title) > 60 else item.title,
                'Price': format_price(item.price) if item.price else 'N/A',
                'Source': item.source_name,
                'Rarity': f"{item.rarity_score:.0f}",
                'Found': item.first_seen.strftime('%H:%M')
            })

        df_recent = pd.DataFrame(recent_data)
        st.dataframe(df_recent, use_container_width=True)
    else:
        st.info(f"No items found in the last {hours} hours.")

# ===================================
# ENDING AUCTIONS PAGE
# ===================================
elif page == "Ending Auctions":
    st.markdown('<div class="main-header">⏰ Ending Auctions</div>', unsafe_allow_html=True)

    hours = st.slider("Show auctions ending in next X hours:", 1, 72, 24)

    auctions = db.get_ending_auctions(hours=hours)

    st.subheader(f"Found {len(auctions)} auctions ending in the next {hours} hours")

    if auctions:
        for item in auctions:
            time_left = item.auction_end_time - datetime.utcnow()
            hours_left = int(time_left.total_seconds() / 3600)

            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

                with col1:
                    st.markdown(f"**{item.title[:60]}...**")
                with col2:
                    st.metric("Price", format_price(item.price) if item.price else 'N/A')
                with col3:
                    st.metric("Bids", item.bid_count or 0)
                with col4:
                    st.metric("Ends In", f"{hours_left}h")

                st.markdown(f"[View Auction]({item.source_url})")
                st.divider()
    else:
        st.info(f"No auctions ending in the next {hours} hours.")

# ===================================
# SETTINGS PAGE
# ===================================
elif page == "Settings":
    st.markdown('<div class="main-header">⚙️ Settings</div>', unsafe_allow_html=True)

    st.subheader("Alert Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Discord Webhook**")
        discord_enabled = st.checkbox("Enable Discord Alerts")
        if discord_enabled:
            discord_webhook = st.text_input("Webhook URL:",
                value=config['alerts']['channels']['discord'].get('webhook_url', ''))

    with col2:
        st.markdown("**Slack Webhook**")
        slack_enabled = st.checkbox("Enable Slack Alerts")
        if slack_enabled:
            slack_webhook = st.text_input("Webhook URL:",
                value=config['alerts']['channels']['slack'].get('webhook_url', ''))

    st.subheader("Database Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Clean Up Old Data (90+ days)", use_container_width=True):
            with st.spinner("Cleaning up..."):
                db.cleanup_old_data(days=90)
                st.success("Cleanup complete!")

    with col2:
        stats = db.get_stats()
        st.metric("Database Size", f"{stats['total_parts']:,} items")

    st.subheader("About")
    st.markdown("""
    ### PC Parts Scraper

    Advanced web scraper for finding rare and forgotten PC parts across:
    - 14+ data sources
    - Government surplus, liquidation, used marketplaces
    - Specializing in FPGA SOMs, datacenter GPUs, and hidden gold

    **Version:** 1.0.0
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Stats")
stats = db.get_stats()
st.sidebar.metric("Total Parts", f"{stats['total_parts']:,}")
st.sidebar.metric("Rare Items", f"{stats['rare_parts']:,}")
