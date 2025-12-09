# PC Parts Scraper

An advanced web scraper for finding rare, vintage, and forgotten PC parts across the entire internet ecosystem - from government surplus auctions to junk stores, liquidation sales, used marketplaces, and discount retailers.

## Features

- **14+ Data Sources** - eBay, government surplus (GovPlanet, PublicSurplus), liquidation sites, Mercari, OfferUp, Craigslist, Woot, Newegg Open Box, Micro Center clearance, Amazon Warehouse, and more
- **Intelligent Rarity Scoring** - Automatically identifies rare/vintage items with a 0-100 rarity score
- **Async Scraping** - Concurrent scraping with rate limiting and proxy rotation support
- **Smart Categorization** - Auto-categorizes parts (CPU, GPU, RAM, motherboard, etc.)
- **Price Tracking** - Tracks price changes over time with history
- **Alert System** - Discord, Slack, and email notifications for rare finds
- **SQLite Database** - Persistent storage with powerful search capabilities
- **Rich CLI** - Beautiful terminal interface with tables and progress indicators
- **🆕 Web GUI** - Modern browser-based interface with charts, filters, and real-time scraping

## Web GUI (Recommended for Beginners)

The easiest way to use the scraper is through the web interface:

```bash
# Linux/Mac
./run_gui.sh

# Windows
run_gui.bat

# Or directly:
streamlit run web_gui.py
```

The GUI opens at `http://localhost:8501` and provides:

- **📊 Dashboard** - Visual stats, charts, and recent activity
- **🚀 Run Scraper** - Click buttons to start scraping (no commands needed!)
- **🔍 Search Parts** - Interactive filters for keywords, price, category, rarity
- **⭐ Rare Finds** - Browse high-value items sorted by rarity
- **🆕 Recent Items** - See what was just found
- **⏰ Ending Auctions** - Track auctions ending soon
- **⚙️ Settings** - Configure alerts and manage database

**Perfect for:**
- First-time users who want a simple interface
- Monitoring scraping progress in real-time
- Quickly filtering and browsing results
- Setting up alerts without editing config files

## Supported Sources

### Government & Surplus
- GovPlanet (Government surplus auctions)
- PublicSurplus (State/local government)
- PropertyRoom (Police surplus)
- Liquidation.com (Business liquidation)

### Discount & Clearance
- Woot (Amazon's daily deals)
- Newegg Open Box
- Micro Center Clearance
- Amazon Warehouse Deals

### Used Marketplaces
- eBay (including vintage-specific searches)
- Mercari
- OfferUp
- Craigslist (multi-city)
- FreeGeek (recycler)

## Installation

### Easy Installation (Recommended)

Use the provided installation scripts:

```bash
# Linux/Mac
./install.sh

# Windows (Command Prompt)
install.bat

# Windows (PowerShell)
.\install.ps1
```

### Python 3.13 Compatibility Note

If you're using Python 3.13, some optional dependencies (pandas, fuzzywuzzy) may not install due to C extension compatibility. The scraper will work fine without them! If automatic installation fails, use this manual workaround:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac

# Install core packages (skip problematic ones)
pip install httpx beautifulsoup4 lxml parsel selenium sqlalchemy aiosqlite fake-useragent rich click pyyaml python-dotenv requests streamlit plotly asyncio-throttle

# The scraper will work perfectly without pandas and fuzzywuzzy!
```

### Manual Installation

```bash
# Clone and enter directory
cd pc_parts_scraper

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers (for JS-heavy sites)
playwright install
```

## Quick Start

```bash
# Run a full scrape across all sources
python main.py scrape --all

# Scrape specific sources
python main.py scrape -s ebay -s mercari

# Search with keywords
python main.py scrape -k "voodoo" -k "3dfx" -k "vintage gpu"

# View rare items found
python main.py rare

# Search the database
python main.py search -k "geforce" --max-price 100

# Show database statistics
python main.py stats
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `scrape` | Run scrapers to find PC parts |
| `search` | Search the database for parts |
| `show <id>` | Show detailed info about an item |
| `recent` | Show recently discovered items |
| `rare` | Show the rarest items found |
| `auctions` | Show auctions ending soon |
| `stats` | Display database statistics |
| `sources` | List available scraper sources |
| `cleanup` | Remove old data from database |

### Scrape Options

```bash
python main.py scrape [OPTIONS]

Options:
  -s, --sources TEXT     Specific sources to scrape (can use multiple)
  -k, --keywords TEXT    Keywords to search for (can use multiple)
  --max-price FLOAT      Maximum price filter
  --all                  Scrape all sources
  --alert/--no-alert     Send alerts for rare finds (default: enabled)
```

### Search Options

```bash
python main.py search [OPTIONS]

Options:
  -k, --keywords TEXT    Search keywords
  -cat, --category TEXT  Filter by category (cpu, gpu, motherboard, etc.)
  -s, --source TEXT      Filter by source
  --max-price FLOAT      Maximum price
  --min-rarity FLOAT     Minimum rarity score (0-100)
  --rare-only            Show only rare items
  -l, --limit INTEGER    Number of results (default: 50)
```

## Configuration

Edit `config/settings.yaml` to customize:

### Proxy Settings
```yaml
proxy:
  enabled: true
  rotation: true
  pool:
    - "http://proxy1:8080"
    - "socks5://proxy2:1080"
```

### Rate Limiting
```yaml
rate_limits:
  default: 30  # requests per minute
  ebay.com: 20
  craigslist.org: 10
```

### Alert Configuration
```yaml
alerts:
  enabled: true
  channels:
    discord:
      enabled: true
      webhook_url: "https://discord.com/api/webhooks/..."
    slack:
      enabled: true
      webhook_url: "https://hooks.slack.com/..."
    email:
      enabled: true
      smtp_server: "smtp.gmail.com"
      recipients:
        - "your@email.com"
```

### Rare Part Keywords

The scraper comes pre-configured with keywords for rare/vintage parts including:

- **Vintage GPUs**: 3dfx Voodoo, GeForce 256, Radeon 9800, Matrox, S3 Virge
- **Vintage CPUs**: Pentium Pro, Athlon XP, Socket 7/370/478/939
- **Vintage RAM**: SDRAM, RDRAM, EDO, SIMM
- **Legacy Interfaces**: AGP, ISA, VESA, SCSI
- **Sound Cards**: Sound Blaster, Gravis Ultrasound, Roland MT-32

## Rarity Scoring

Items are scored 0-100 based on:

- Keyword matches (vintage terms, specific part names)
- Old technology indicators (AGP, ISA, Socket 7, etc.)
- Rarity terms ("engineering sample", "nos", "sealed")
- Ultra-rare items (Voodoo 5, Xeon Phi, etc.)

Items scoring 50+ are flagged as "rare" and trigger alerts.

## Project Structure

```
pc_parts_scraper/
├── main.py                 # Entry point
├── cli.py                  # CLI interface
├── requirements.txt        # Dependencies
├── config/
│   └── settings.yaml       # Configuration
├── scrapers/
│   ├── base.py            # Base scraper class
│   ├── ebay.py            # eBay scrapers
│   ├── surplus.py         # Government surplus scrapers
│   ├── discount.py        # Discount/clearance scrapers
│   └── marketplace.py     # Used marketplace scrapers
├── utils/
│   ├── models.py          # Database models
│   ├── database.py        # Database manager
│   └── helpers.py         # Utility functions
├── alerts/
│   └── notifier.py        # Alert system
└── data/
    └── pc_parts.db        # SQLite database
```

## Advanced Usage

### Custom Scraper

```python
from scrapers.base import BaseScraper

class MyScraper(BaseScraper):
    def __init__(self, config=None):
        super().__init__(config)
        self.name = "my_scraper"
        self.base_url = "https://example.com"

    async def scrape(self, keywords=None, **kwargs):
        # Your scraping logic
        pass

    def extract_title(self, item, soup=None):
        return item.select_one('.title').text

    def extract_url(self, item, soup=None):
        return item.select_one('a')['href']
```

### Direct Database Access

```python
from utils.database import DatabaseManager

db = DatabaseManager()

# Search for rare GPUs
parts = db.search_parts(
    keywords=['voodoo', 'geforce'],
    categories=['gpu'],
    min_rarity=60,
    max_price=100
)

# Get price drops
drops = db.get_price_drops(min_drop_percent=20)

# Get ending auctions
auctions = db.get_ending_auctions(hours=6)
```

### Programmatic Scraping

```python
import asyncio
from scrapers import EbayScraper, MercariScraper, MultiSourceScraper
from utils.helpers import load_config

config = load_config()

async def main():
    scrapers = [
        EbayScraper(config),
        MercariScraper(config),
    ]

    multi = MultiSourceScraper(scrapers, config)
    items = await multi.scrape_all(
        keywords=['3dfx', 'voodoo'],
        max_concurrent=3
    )

    for item in items:
        if item.rarity_score >= 70:
            print(f"RARE: {item.title} - ${item.price}")

asyncio.run(main())
```

## Tips for Finding Rare Parts

1. **Set up alerts** - Configure Discord/Slack webhooks for instant notifications
2. **Run frequently** - Schedule scrapes every 15-30 minutes for rare keyword alerts
3. **Use specific keywords** - "voodoo 5 6000" finds more than just "voodoo"
4. **Check auctions ending soon** - Best deals often come at auction close
5. **Monitor government surplus** - Bulk IT liquidations often include gems
6. **Watch local listings** - Craigslist/OfferUp for in-person pickups

## Legal & Ethical

- Respects robots.txt by default
- Built-in rate limiting to avoid server overload
- Use responsibly and respect website terms of service
- For personal use in finding parts for vintage computing hobby

## License

MIT License - Use freely for personal projects.

## Contributing

Pull requests welcome! Areas for improvement:
- Additional scraper sources
- Better rarity detection algorithms
- Image analysis for part identification
- Price prediction models
