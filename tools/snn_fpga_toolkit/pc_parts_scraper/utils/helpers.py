"""
Utility functions for PC Parts Scraper
"""

import re
import hashlib
from typing import List, Optional, Dict, Any
from datetime import datetime
import yaml

# Optional fuzzy matching (performance optimization)
try:
    from fuzzywuzzy import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def clean_price(price_str: str) -> Optional[float]:
    """Extract numeric price from string"""
    if not price_str:
        return None

    # Remove currency symbols and whitespace
    cleaned = re.sub(r'[^\d.,]', '', str(price_str))

    # Handle different formats
    if ',' in cleaned and '.' in cleaned:
        # Could be 1,234.56 or 1.234,56
        if cleaned.rfind(',') > cleaned.rfind('.'):
            cleaned = cleaned.replace('.', '').replace(',', '.')
        else:
            cleaned = cleaned.replace(',', '')
    elif ',' in cleaned:
        # Could be 1,234 or 1,23
        if len(cleaned.split(',')[-1]) == 3:
            cleaned = cleaned.replace(',', '')
        else:
            cleaned = cleaned.replace(',', '.')

    try:
        return float(cleaned)
    except ValueError:
        return None


def normalize_condition(condition_str: str) -> str:
    """Normalize condition strings to standard values"""
    if not condition_str:
        return "unknown"

    condition_lower = condition_str.lower()

    # Map to standard conditions
    if any(word in condition_lower for word in ['new', 'sealed', 'bnib', 'nib']):
        return 'new'
    elif any(word in condition_lower for word in ['refurb', 'renewed', 'certified']):
        return 'refurbished'
    elif any(word in condition_lower for word in ['open box', 'open-box', 'openbox']):
        return 'open_box'
    elif any(word in condition_lower for word in ['used', 'pre-owned', 'preowned']):
        return 'used'
    elif any(word in condition_lower for word in ['parts', 'repair', 'broken', 'as-is', 'as is']):
        return 'for_parts'
    else:
        return 'unknown'


def categorize_part(title: str, description: str = "") -> str:
    """Categorize a PC part based on title and description"""
    text = f"{title} {description}".lower()

    # ===========================================
    # FPGA PATTERNS (Check first - high value)
    # ===========================================
    fpga_patterns = [
        r'\bfpga\b', r'\balveo\b', r'\bvirtex\b', r'\bkintex\b', r'\bartix\b',
        r'\bzynq\b', r'\bspartan\b', r'\bversal\b', r'\bstratix\b', r'\barria\b',
        r'\bcyclone\b', r'\bagilex\b', r'\bmax\s*10\b', r'\baltera\b',
        r'\bxilinx\b', r'\blattice\b', r'\becp5\b', r'\bice40\b',
        r'\bmicrosemi\b', r'\bpolarfire\b', r'\bsmartfusion\b', r'\bigloo\b',
        r'\bachronix\b', r'\bspeedster\b', r'\befinix\b', r'\bgowin\b',
        r'\bde10\b', r'\bde1-soc\b', r'\bde0\b', r'\bterasic\b', r'\bdigilent\b',
        r'\bbittware\b', r'\bhtg\b.*\bfpga\b', r'\bnumato\b'
    ]
    for pattern in fpga_patterns:
        if re.search(pattern, text):
            return 'fpga'

    # ===========================================
    # AI ACCELERATORS (Check before GPU)
    # ===========================================
    ai_accel_patterns = [
        r'\btpu\b', r'\bcoral\b.*\bedge\b', r'\bedge\s*tpu\b',
        r'\bhabana\b', r'\bgaudi\b', r'\bgoya\b',
        r'\bgraphcore\b', r'\bipu\b', r'\bcerebras\b', r'\bgroq\b',
        r'\bsambanova\b', r'\bmythic\b', r'\bhailo\b', r'\bblaize\b',
        r'\bjetson\b', r'\borin\b', r'\bxavier\b', r'\bagx\b'
    ]
    for pattern in ai_accel_patterns:
        if re.search(pattern, text):
            return 'ai_accelerator'

    # ===========================================
    # DATACENTER GPUs (Check before generic GPU)
    # ===========================================
    datacenter_gpu_patterns = [
        r'\bh100\b', r'\bh200\b', r'\ba100\b', r'\ba800\b', r'\ba30\b',
        r'\ba40\b', r'\ba10\b', r'\ba16\b', r'\bl40\b', r'\bl4\b',
        r'\btesla\s*v100\b', r'\btesla\s*p100\b', r'\btesla\s*p40\b',
        r'\btesla\s*p4\b', r'\btesla\s*k80\b', r'\btesla\s*k40\b',
        r'\btesla\s*k20\b', r'\btesla\s*m40\b', r'\btesla\s*m60\b',
        r'\bdgx\b', r'\bnvlink\b', r'\bsxm[245]\b',
        r'\binstinct\s*mi\d+', r'\bmi300\b', r'\bmi250\b', r'\bmi100\b',
        r'\bmi60\b', r'\bmi50\b', r'\bmi25\b'
    ]
    for pattern in datacenter_gpu_patterns:
        if re.search(pattern, text):
            return 'gpu_datacenter'

    # ===========================================
    # PROFESSIONAL GPUs (Quadro/FirePro/Radeon Pro)
    # ===========================================
    pro_gpu_patterns = [
        r'\bquadro\b', r'\brtx\s*a[456]\d{3}\b', r'\brtx\s*[56]000\s*ada\b',
        r'\bfirepro\b', r'\bradeon\s*pro\b', r'\bwx\s*\d{4}\b',
        r'\btitan\s*v\b', r'\btitan\s*rtx\b', r'\btitan\s*z\b',
        r'\btitan\s*xp\b', r'\btitan\s*x\b', r'\btitan\s*black\b'
    ]
    for pattern in pro_gpu_patterns:
        if re.search(pattern, text):
            return 'gpu_professional'

    # ===========================================
    # CPU patterns
    # ===========================================
    cpu_patterns = [
        r'\bcpu\b', r'\bprocessor\b', r'\bintel\b.*\b(core|xeon|pentium|celeron)\b',
        r'\bamd\b.*\b(ryzen|athlon|epyc|opteron|phenom)\b', r'\bsocket\s*\d+',
        r'\blga\s*\d+', r'\bam[45]\b', r'\bxeon\s*phi\b', r'\bitanium\b'
    ]
    for pattern in cpu_patterns:
        if re.search(pattern, text):
            return 'cpu'

    # ===========================================
    # Generic GPU patterns
    # ===========================================
    gpu_patterns = [
        r'\bgpu\b', r'\bgraphics\s*card\b', r'\bvideo\s*card\b',
        r'\bgeforce\b', r'\bradeon\b', r'\brtx\b', r'\bgtx\b',
        r'\bvoodoo\b', r'\b3dfx\b', r'\bmatrox\b', r'\bs3\s*virge\b',
        r'\briva\s*tnt\b', r'\brage\b', r'\bmach64\b'
    ]
    for pattern in gpu_patterns:
        if re.search(pattern, text):
            return 'gpu'

    # Motherboard patterns
    mobo_patterns = [
        r'\bmotherboard\b', r'\bmobo\b', r'\bmainboard\b',
        r'\batx\b', r'\bmicro.?atx\b', r'\bmini.?itx\b'
    ]
    for pattern in mobo_patterns:
        if re.search(pattern, text):
            return 'motherboard'

    # RAM patterns
    ram_patterns = [
        r'\bram\b', r'\bmemory\b', r'\bddr[1-5]\b', r'\bsdram\b',
        r'\brdram\b', r'\bdimm\b', r'\bsimm\b', r'\bsodimm\b'
    ]
    for pattern in ram_patterns:
        if re.search(pattern, text):
            return 'ram'

    # Storage patterns
    storage_patterns = [
        r'\bssd\b', r'\bhdd\b', r'\bhard\s*drive\b', r'\bsolid\s*state\b',
        r'\bnvme\b', r'\bsata\b', r'\bscsi\b', r'\bide\b', r'\bpata\b'
    ]
    for pattern in storage_patterns:
        if re.search(pattern, text):
            return 'storage'

    # PSU patterns
    psu_patterns = [
        r'\bpsu\b', r'\bpower\s*supply\b', r'\b\d+\s*w(att)?\b'
    ]
    for pattern in psu_patterns:
        if re.search(pattern, text):
            return 'psu'

    # Cooling patterns
    cooling_patterns = [
        r'\bcooler\b', r'\bfan\b', r'\bheatsink\b', r'\baio\b',
        r'\bwater\s*cool', r'\bliquid\s*cool'
    ]
    for pattern in cooling_patterns:
        if re.search(pattern, text):
            return 'cooling'

    # Case patterns
    case_patterns = [
        r'\bcase\b', r'\bchassis\b', r'\btower\b', r'\benclosure\b'
    ]
    for pattern in case_patterns:
        if re.search(pattern, text):
            return 'case'

    # Sound card patterns
    sound_patterns = [
        r'\bsound\s*card\b', r'\baudio\s*card\b', r'\bsound\s*blaster\b'
    ]
    for pattern in sound_patterns:
        if re.search(pattern, text):
            return 'sound_card'

    # Network patterns
    network_patterns = [
        r'\bnetwork\s*card\b', r'\bnic\b', r'\bethernet\b', r'\bwifi\b'
    ]
    for pattern in network_patterns:
        if re.search(pattern, text):
            return 'network'

    return 'other'


def calculate_rarity_score(
    title: str,
    description: str,
    keywords: List[str],
    category: str,
    price: Optional[float] = None
) -> float:
    """
    Calculate a rarity score (0-100) for a PC part.
    Higher scores indicate rarer/more valuable items.
    """
    score = 0.0
    text = f"{title} {description}".lower()

    # Keyword matching (up to 50 points)
    keyword_matches = []
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in text:
            keyword_matches.append(keyword)
            # More specific keywords get more points
            if len(keyword.split()) >= 2:
                score += 10
            else:
                score += 5

    score = min(score, 50)  # Cap keyword points at 50

    # Vintage indicators (up to 20 points)
    vintage_terms = [
        'vintage', 'retro', 'classic', 'legacy', 'rare', 'collectible',
        'nos', 'new old stock', 'sealed', 'bnib', 'engineering sample',
        'es chip', 'prototype', 'server pull', 'datacenter'
    ]
    for term in vintage_terms:
        if term in text:
            score += 5
    score = min(score, 70)  # Cap at 70 after vintage

    # Old technology indicators (up to 15 points)
    old_tech = [
        'agp', 'pci', 'isa', 'vesa', 'mca', 'eisa', 'ide', 'pata',
        'scsi', 'slot 1', 'slot a', 'socket 7', 'socket 370', 'socket 478'
    ]
    for tech in old_tech:
        if tech in text:
            score += 5
    score = min(score, 85)  # Cap at 85

    # Specific rare items (bonus points)
    ultra_rare = [
        '3dfx', 'voodoo 5', 'voodoo5', 'xeon phi', 'titan v', 'tesla v100',
        'radeon pro duo', 'r9 fury x', 'pentium pro', 'gravis ultrasound',
        'roland mt-32'
    ]
    for item in ultra_rare:
        if item in text:
            score += 15

    # ===========================================
    # HIGH-VALUE DATACENTER GPUs (Major bonus)
    # ===========================================
    datacenter_gpus = [
        'h100', 'h200', 'a100', 'a800', 'dgx', 'sxm4', 'sxm5',
        'mi300', 'mi250', 'mi100', 'instinct'
    ]
    for gpu in datacenter_gpus:
        if gpu in text:
            score += 25

    # ===========================================
    # HIGH-VALUE FPGAs (Major bonus)
    # ===========================================
    premium_fpgas = [
        'alveo', 'virtex ultrascale', 'versal', 'stratix 10', 'agilex',
        'arria 10', 'kintex ultrascale'
    ]
    for fpga in premium_fpgas:
        if fpga in text:
            score += 25

    mid_tier_fpgas = [
        'virtex-7', 'virtex-6', 'kintex-7', 'stratix v', 'stratix iv',
        'arria v', 'cyclone 10', 'polarfire'
    ]
    for fpga in mid_tier_fpgas:
        if fpga in text:
            score += 15

    entry_fpgas = [
        'spartan', 'artix', 'cyclone v', 'cyclone iv', 'max 10',
        'ecp5', 'ice40', 'de10', 'de1-soc'
    ]
    for fpga in entry_fpgas:
        if fpga in text:
            score += 10

    # ===========================================
    # MODULAR FPGA SALVAGE GOLD (SOMs - Best Value)
    # ===========================================
    # These are the real finds - high-end silicon on removable modules
    premium_soms = [
        'te0803', 'te0807', 'te0808', 'trenz',
        'acu3eg', 'acu2cg', 'alinx',
        'zu3eg', 'zu7ev', 'zu9eg', 'zu15eg', 'zu19eg',
        'kria k26', 'kria som',
        'enclustra mercury', 'mercury xu',
        'ultra96', 'microzed', 'picozed'
    ]
    for som in premium_soms:
        if som in text:
            score += 20  # High value - cheap silicon

    # Zynq UltraScale+ MPSoC chips (The Gold Standard)
    mpsoc_chips = [
        'zynq ultrascale', 'mpsoc', 'zu2cg', 'zu4ev', 'zu5ev'
    ]
    for chip in mpsoc_chips:
        if chip in text:
            score += 15

    # Budget Zynq-7000 SOMs (Still good value)
    budget_soms = [
        'z7020', 'z7045', 'zynq 7000', 'zedboard', 'arty z7',
        'pynq', 'basys', 'nexys', 'cmod'
    ]
    for som in budget_soms:
        if som in text:
            score += 10

    # Official eval boards (often heavily discounted)
    eval_boards = [
        'zcu102', 'zcu104', 'zcu106', 'zcu111',
        'vcu118', 'vcu128', 'kcu105', 'kcu116'
    ]
    for board in eval_boards:
        if board in text:
            score += 15

    # ===========================================
    # AI ACCELERATORS (Major bonus)
    # ===========================================
    ai_accelerators = [
        'tpu', 'gaudi', 'graphcore', 'ipu', 'cerebras', 'groq',
        'sambanova', 'jetson orin', 'jetson agx'
    ]
    for accel in ai_accelerators:
        if accel in text:
            score += 20

    # ===========================================
    # PROFESSIONAL GPUs (Moderate bonus)
    # ===========================================
    pro_gpus = [
        'quadro rtx 8000', 'quadro rtx 6000', 'quadro gv100', 'quadro gp100',
        'rtx a6000', 'rtx 6000 ada', 'firepro w9100', 'firepro s9170',
        'radeon pro w7900', 'radeon pro vii'
    ]
    for gpu in pro_gpus:
        if gpu in text:
            score += 15

    # ===========================================
    # HIDDEN GOLD: FPGA MINING CARDS
    # ===========================================
    fpga_miners = [
        'vcu1525', 'bcu-1525', 'cvp-13', 'sqrl', 'blackminer',
        'fpga miner'
    ]
    for miner in fpga_miners:
        if miner in text:
            score += 20  # Premium FPGAs inside

    # ===========================================
    # HIDDEN GOLD: SMARTNICS & NETWORK FPGAS
    # ===========================================
    smartnics = [
        'solarflare', 'napatech', 'bluefield', 'pensando', 'exablaze',
        'smartnic', 'fungible'
    ]
    for nic in smartnics:
        if nic in text:
            score += 15

    # ===========================================
    # HIDDEN GOLD: TEST & MEASUREMENT
    # ===========================================
    test_equipment = [
        'usrp', 'ettus', 'ni pxi', 'compactrio', 'ni crio',
        'adalm pluto', 'lime sdr'
    ]
    for equip in test_equipment:
        if equip in text:
            score += 15

    # ===========================================
    # HIDDEN GOLD: TELECOM BLADES
    # ===========================================
    telecom_blades = [
        'atca blade', 'advancedtca', 'microtca', 'amc module',
        'cpci board', 'compactpci', 'vme board', 'vpx board', 'openvpx'
    ]
    for blade in telecom_blades:
        if blade in text:
            score += 15

    # ===========================================
    # HIDDEN GOLD: MULTI-GPU SYSTEMS
    # ===========================================
    multi_gpu = [
        '8 gpu', '6 gpu', '4 gpu', 'mining rig', 'gpu rig',
        'crypto mining', 'ethereum mining'
    ]
    for mg in multi_gpu:
        if mg in text:
            score += 10

    # ===========================================
    # HIDDEN GOLD: INDUSTRIAL EQUIPMENT
    # ===========================================
    industrial = [
        'beckhoff', 'national instruments', 'cognex', 'keyence',
        'basler camera', 'machine vision'
    ]
    for ind in industrial:
        if ind in text:
            score += 10

    # Category bonus
    if category in ['fpga', 'gpu_datacenter', 'ai_accelerator']:
        score += 10
    elif category == 'gpu_professional':
        score += 5

    return min(score, 100)


def generate_item_hash(source_name: str, source_url: str) -> str:
    """Generate a unique hash for an item to detect duplicates"""
    unique_str = f"{source_name}:{source_url}"
    return hashlib.md5(unique_str.encode()).hexdigest()


def fuzzy_match_keywords(text: str, keywords: List[str], threshold: int = 80) -> List[str]:
    """Find keywords that fuzzy match in the text"""
    matched = []
    text_lower = text.lower()

    for keyword in keywords:
        # First try exact match
        if keyword.lower() in text_lower:
            matched.append(keyword)
            continue

        # Then try fuzzy match on word boundaries (if available)
        if FUZZY_AVAILABLE:
            words = text_lower.split()
            for i in range(len(words)):
                # Try matching against single words and word pairs
                for j in range(i + 1, min(i + 4, len(words) + 1)):
                    phrase = ' '.join(words[i:j])
                    ratio = fuzz.ratio(keyword.lower(), phrase)
                    if ratio >= threshold:
                        matched.append(keyword)
                        break
                else:
                    continue
                break

    return list(set(matched))


def parse_auction_time(time_str: str) -> Optional[datetime]:
    """Parse various auction end time formats"""
    if not time_str:
        return None

    # Common patterns
    patterns = [
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})',  # ISO format
        r'(\w+ \d+, \d{4} \d{1,2}:\d{2} [AP]M)',    # "Jan 1, 2024 3:00 PM"
        r'(\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2})',   # "1/1/2024 15:00"
    ]

    for pattern in patterns:
        match = re.search(pattern, time_str)
        if match:
            try:
                # Try parsing with different formats
                for fmt in [
                    '%Y-%m-%dT%H:%M:%S',
                    '%b %d, %Y %I:%M %p',
                    '%m/%d/%Y %H:%M'
                ]:
                    try:
                        return datetime.strptime(match.group(1), fmt)
                    except ValueError:
                        continue
            except Exception:
                continue

    return None


def format_price(price: float, currency: str = 'USD') -> str:
    """Format price for display"""
    if currency == 'USD':
        return f"${price:,.2f}"
    elif currency == 'EUR':
        return f"€{price:,.2f}"
    elif currency == 'GBP':
        return f"£{price:,.2f}"
    else:
        return f"{price:,.2f} {currency}"


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
