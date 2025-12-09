"""
CLI Interface for PC Parts Scraper
"""

import asyncio
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown

from scrapers import get_all_scrapers, get_scraper_by_name, MultiSourceScraper
from utils.database import DatabaseManager
from utils.helpers import load_config, format_price, truncate_text
from alerts.notifier import AlertNotifier

console = Console()


@click.group()
@click.option('--config', '-c', default='config/settings.yaml', help='Config file path')
@click.pass_context
def cli(ctx, config):
    """
    PC Parts Scraper - Find rare and forgotten PC parts across the web

    A powerful tool to scrape junk stores, surplus auctions, discount retailers,
    and used marketplaces for vintage and hard-to-find PC components.
    """
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config)
    ctx.obj['db'] = DatabaseManager(ctx.obj['config']['database']['path'])


@cli.command()
@click.option('--sources', '-s', multiple=True, help='Specific sources to scrape')
@click.option('--keywords', '-k', multiple=True, help='Keywords to search for')
@click.option('--max-price', type=float, help='Maximum price filter')
@click.option('--all', 'scrape_all', is_flag=True, help='Scrape all sources')
@click.option('--alert/--no-alert', default=True, help='Send alerts for rare finds')
@click.pass_context
def scrape(ctx, sources, keywords, max_price, scrape_all, alert):
    """Run the scraper to find PC parts"""
    config = ctx.obj['config']
    db = ctx.obj['db']

    console.print(Panel.fit(
        "[bold blue]PC Parts Scraper[/bold blue]\n"
        "Searching for rare and forgotten PC parts...",
        border_style="blue"
    ))

    # Get scrapers
    if scrape_all or not sources:
        scrapers = get_all_scrapers(config)
        console.print(f"Using all {len(scrapers)} scrapers")
    else:
        scrapers = []
        for source in sources:
            scraper = get_scraper_by_name(source, config)
            if scraper:
                scrapers.append(scraper)
            else:
                console.print(f"[yellow]Unknown source: {source}[/yellow]")

    if not scrapers:
        console.print("[red]No valid scrapers found![/red]")
        return

    # Get keywords
    kw_list = list(keywords) if keywords else config.get('rare_part_keywords', [])[:20]

    # Run scrapers
    async def run_scraping():
        multi_scraper = MultiSourceScraper(scrapers, config)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Scraping...", total=None)

            items = await multi_scraper.scrape_all(
                keywords=kw_list,
                max_concurrent=config.get('general', {}).get('concurrent_requests', 5)
            )

            progress.update(task, completed=True)

        return items, multi_scraper.runs

    items, runs = asyncio.run(run_scraping())

    # Save to database
    console.print("\n[bold]Saving results to database...[/bold]")
    stats = db.save_parts_bulk(items)

    # Show results
    console.print(f"\n[green]Scraping complete![/green]")
    console.print(f"  New items: {stats['new']}")
    console.print(f"  Updated items: {stats['updated']}")
    console.print(f"  Errors: {stats['errors']}")

    # Show rare finds
    rare_items = [item for item in items if item.is_rare]
    if rare_items:
        console.print(f"\n[bold yellow]Found {len(rare_items)} rare items![/bold yellow]")

        # Show top rare items
        table = Table(title="Top Rare Finds")
        table.add_column("Title", style="cyan", max_width=40)
        table.add_column("Price", style="green")
        table.add_column("Source", style="blue")
        table.add_column("Rarity", style="yellow")

        for item in sorted(rare_items, key=lambda x: x.rarity_score, reverse=True)[:10]:
            table.add_row(
                truncate_text(item.title, 40),
                format_price(item.price) if item.price else "N/A",
                item.source_name,
                f"{item.rarity_score:.0f}"
            )

        console.print(table)

    # Send alerts
    if alert:
        async def send_alerts():
            notifier = AlertNotifier(config, db)
            await notifier.check_and_alert(items)

        asyncio.run(send_alerts())


@cli.command()
@click.option('--keywords', '-k', multiple=True, help='Search keywords')
@click.option('--category', '-cat', help='Filter by category')
@click.option('--source', '-s', help='Filter by source')
@click.option('--max-price', type=float, help='Maximum price')
@click.option('--min-rarity', type=float, default=0, help='Minimum rarity score')
@click.option('--rare-only', is_flag=True, help='Show only rare items')
@click.option('--limit', '-l', default=50, help='Number of results')
@click.pass_context
def search(ctx, keywords, category, source, max_price, min_rarity, rare_only, limit):
    """Search the database for parts"""
    db = ctx.obj['db']

    # Build filters
    categories = [category] if category else None
    sources = [source] if source else None

    results = db.search_parts(
        keywords=list(keywords) if keywords else None,
        categories=categories,
        sources=sources,
        max_price=max_price,
        min_rarity=min_rarity,
        is_rare=True if rare_only else None,
        limit=limit
    )

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    # Display results
    table = Table(title=f"Search Results ({len(results)} items)")
    table.add_column("ID", style="dim")
    table.add_column("Title", style="cyan", max_width=35)
    table.add_column("Price", style="green")
    table.add_column("Category", style="blue")
    table.add_column("Source", style="magenta")
    table.add_column("Rarity", style="yellow")
    table.add_column("Condition")

    for item in results:
        rarity_color = "green" if item.rarity_score >= 70 else "yellow" if item.rarity_score >= 40 else "white"
        table.add_row(
            str(item.id),
            truncate_text(item.title, 35),
            format_price(item.price) if item.price else "N/A",
            item.category or "?",
            item.source_name,
            f"[{rarity_color}]{item.rarity_score:.0f}[/{rarity_color}]",
            item.condition or "?"
        )

    console.print(table)


@cli.command()
@click.argument('item_id', type=int)
@click.pass_context
def show(ctx, item_id):
    """Show detailed information about a specific item"""
    db = ctx.obj['db']

    item = db.get_part_by_id(item_id)

    if not item:
        console.print(f"[red]Item {item_id} not found[/red]")
        return

    # Build detail view
    info = f"""
# {item.title}

## Details

- **Price:** {format_price(item.price) if item.price else 'Not listed'}
- **Category:** {item.category or 'Unknown'}
- **Condition:** {item.condition or 'Unknown'}
- **Source:** {item.source_name}
- **Rarity Score:** {item.rarity_score:.1f}/100

## Location & Shipping

- **Location:** {item.location or 'Unknown'}
- **Shipping Available:** {'Yes' if item.shipping_available else 'No'}
- **Local Pickup:** {'Yes' if item.local_pickup else 'No'}

## Auction Info

- **Is Auction:** {'Yes' if item.is_auction else 'No'}
- **Bid Count:** {item.bid_count or 'N/A'}
- **Ends:** {item.auction_end_time or 'N/A'}

## Matched Keywords

{', '.join(item.matched_keywords) if item.matched_keywords else 'None'}

## Description

{item.description or 'No description available'}

## Link

{item.source_url}
"""

    console.print(Markdown(info))


@cli.command()
@click.option('--hours', '-h', default=24, help='Look back hours')
@click.option('--limit', '-l', default=20, help='Number of items')
@click.pass_context
def recent(ctx, hours, limit):
    """Show recently discovered items"""
    db = ctx.obj['db']

    items = db.get_recent_parts(hours=hours, limit=limit)

    if not items:
        console.print(f"[yellow]No items found in the last {hours} hours[/yellow]")
        return

    table = Table(title=f"Recent Items (Last {hours}h)")
    table.add_column("ID", style="dim")
    table.add_column("Title", style="cyan", max_width=35)
    table.add_column("Price", style="green")
    table.add_column("Source", style="blue")
    table.add_column("Found", style="magenta")

    for item in items:
        table.add_row(
            str(item.id),
            truncate_text(item.title, 35),
            format_price(item.price) if item.price else "N/A",
            item.source_name,
            item.first_seen.strftime("%H:%M")
        )

    console.print(table)


@cli.command()
@click.pass_context
def rare(ctx):
    """Show the rarest items found"""
    db = ctx.obj['db']

    items = db.get_rare_parts(min_score=50, limit=30)

    if not items:
        console.print("[yellow]No rare items found yet. Run a scrape first![/yellow]")
        return

    table = Table(title="Rarest PC Parts Found")
    table.add_column("ID", style="dim")
    table.add_column("Title", style="cyan", max_width=40)
    table.add_column("Price", style="green")
    table.add_column("Source", style="blue")
    table.add_column("Rarity", style="yellow")

    for item in items:
        table.add_row(
            str(item.id),
            truncate_text(item.title, 40),
            format_price(item.price) if item.price else "N/A",
            item.source_name,
            f"{item.rarity_score:.0f}/100"
        )

    console.print(table)


@cli.command()
@click.pass_context
def auctions(ctx):
    """Show auctions ending soon"""
    db = ctx.obj['db']

    items = db.get_ending_auctions(hours=24)

    if not items:
        console.print("[yellow]No auctions ending in the next 24 hours[/yellow]")
        return

    table = Table(title="Auctions Ending Soon")
    table.add_column("Title", style="cyan", max_width=35)
    table.add_column("Price", style="green")
    table.add_column("Bids", style="blue")
    table.add_column("Ends", style="red")
    table.add_column("Source")

    for item in items:
        end_time = item.auction_end_time.strftime("%H:%M") if item.auction_end_time else "?"
        table.add_row(
            truncate_text(item.title, 35),
            format_price(item.price) if item.price else "N/A",
            str(item.bid_count or 0),
            end_time,
            item.source_name
        )

    console.print(table)


@cli.command()
@click.pass_context
def stats(ctx):
    """Show database statistics"""
    db = ctx.obj['db']

    stats = db.get_stats()

    console.print(Panel.fit(
        "[bold]Database Statistics[/bold]",
        border_style="blue"
    ))

    console.print(f"\n[bold]Overall:[/bold]")
    console.print(f"  Total parts: {stats['total_parts']}")
    console.print(f"  Active parts: {stats['active_parts']}")
    console.print(f"  Rare parts: {stats['rare_parts']}")

    console.print(f"\n[bold]Pricing:[/bold]")
    console.print(f"  Average: {format_price(stats['avg_price'])}")
    console.print(f"  Min: {format_price(stats['min_price'])}")
    console.print(f"  Max: {format_price(stats['max_price'])}")

    # By source
    console.print(f"\n[bold]By Source:[/bold]")
    for source, count in sorted(stats['by_source'].items(), key=lambda x: x[1], reverse=True):
        console.print(f"  {source}: {count}")

    # By category
    console.print(f"\n[bold]By Category:[/bold]")
    for cat, count in sorted(stats['by_category'].items(), key=lambda x: x[1], reverse=True):
        console.print(f"  {cat}: {count}")


@cli.command()
@click.pass_context
def sources(ctx):
    """List available scraper sources"""
    config = ctx.obj['config']

    console.print(Panel.fit(
        "[bold]Available Scraper Sources[/bold]",
        border_style="blue"
    ))

    # List all sources from config
    for category, data in config.get('sources', {}).items():
        if data.get('enabled', False):
            console.print(f"\n[bold blue]{category.upper()}[/bold blue]")
            for site in data.get('sites', []):
                status = "[green]✓[/green]" if not site.get('requires_auth') else "[yellow]⚠ auth[/yellow]"
                console.print(f"  {status} {site['name']}: {site['url']}")


@cli.command()
@click.option('--days', '-d', default=90, help='Remove data older than days')
@click.pass_context
def cleanup(ctx, days):
    """Clean up old data from database"""
    db = ctx.obj['db']

    console.print(f"Cleaning up data older than {days} days...")
    db.cleanup_old_data(days)
    console.print("[green]Cleanup complete![/green]")


def main():
    """Main entry point"""
    cli(obj={})


if __name__ == '__main__':
    main()
