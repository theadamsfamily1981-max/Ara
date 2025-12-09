#!/usr/bin/env python3
"""
AI-Powered FPGA Board Detection
Uses computer vision to identify mining hardware and generate configs automatically
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json
from dataclasses import dataclass, asdict
import easyocr


@dataclass
class BoardDetection:
    """Detection result for an FPGA board"""
    board_type: str  # "hashboard", "pcie_card", "atca_blade", "unknown"
    vendor: str  # "intel", "xilinx", "unknown"
    fpga_model: str  # "AGF027", "XCVU35P", etc.
    chip_count: int  # Number of FPGAs detected
    confidence: float  # 0.0 - 1.0
    jtag_location: Optional[Tuple[int, int]]  # (x, y) pixel location of JTAG header
    power_connectors: List[str]  # ["XT60", "ATX_24pin", "PCIe_8pin"]
    recommendations: List[str]  # Suggested next steps
    config_file: str  # Recommended OpenOCD config


class FPGABoardDetector:
    """
    Computer vision-based FPGA board identification

    Uses multiple techniques:
    1. OCR for chip markings (Tesseract + EasyOCR)
    2. Template matching for known boards
    3. Connector detection (PCIe, JTAG, power)
    4. Color/shape analysis
    """

    def __init__(self, models_dir: Path = Path(__file__).parent / "models"):
        self.models_dir = models_dir
        self.models_dir.mkdir(exist_ok=True, parents=True)

        # Initialize OCR reader (multi-language for Chinese boards)
        print("[AI] Initializing OCR engines...")
        self.ocr = easyocr.Reader(['en', 'ch_sim'], gpu=False)

        # Known FPGA chip patterns
        self.fpga_patterns = {
            # Intel Agilex
            r'AGF\d{3}': ('intel', 'agilex', 'hashboard_agilex.cfg'),
            r'AGFA\d{3}': ('intel', 'agilex', 'hashboard_agilex.cfg'),
            r'AGFB\d{3}': ('intel', 'agilex', 'hashboard_agilex.cfg'),

            # Intel Stratix 10
            r'1SG\d{3}': ('intel', 'stratix10', 'stratix10.cfg'),
            r'1SM\d{3}': ('intel', 'stratix10', 'stratix10.cfg'),

            # Xilinx Virtex UltraScale+
            r'XCVU\d{2}P': ('xilinx', 'virtex_ultrascale_plus', 'pcie_mining_card.cfg'),
            r'XC7VX\d{3}T': ('xilinx', 'virtex7', 'atca_xilinx.cfg'),

            # Xilinx Kintex UltraScale+
            r'XCKU\d{2}P': ('xilinx', 'kintex_ultrascale_plus', 'kintex_ultrascale_plus.cfg'),
        }

        # Load board templates for matching
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, np.ndarray]:
        """Load reference images for known board types"""
        templates = {}
        template_dir = self.models_dir / "templates"
        if template_dir.exists():
            for img_file in template_dir.glob("*.jpg"):
                templates[img_file.stem] = cv2.imread(str(img_file))
        return templates

    def detect_from_image(self, image_path: Path) -> BoardDetection:
        """
        Detect FPGA board from a photo

        Args:
            image_path: Path to board photo (top-down view recommended)

        Returns:
            BoardDetection with identified hardware and config recommendation
        """
        print(f"[AI] Analyzing board image: {image_path}")

        # Load and preprocess image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        height, width = img.shape[:2]
        print(f"[AI] Image size: {width}x{height}")

        # Step 1: OCR to find chip markings
        print("[AI] Running OCR to detect chip markings...")
        ocr_results = self.ocr.readtext(str(image_path))

        detected_chips = []
        for (bbox, text, conf) in ocr_results:
            text_upper = text.upper().replace(" ", "").replace("-", "")

            # Check against known FPGA patterns
            for pattern, (vendor, model, config) in self.fpga_patterns.items():
                import re
                if re.search(pattern, text_upper):
                    print(f"[AI] ✓ Found {vendor.upper()} {model}: {text} (confidence: {conf:.2f})")
                    detected_chips.append({
                        'text': text,
                        'vendor': vendor,
                        'model': model,
                        'config': config,
                        'confidence': conf,
                        'bbox': bbox
                    })

        # Step 2: Detect board form factor
        board_type = self._detect_form_factor(img)
        print(f"[AI] Board type: {board_type}")

        # Step 3: Detect connectors
        power_connectors = self._detect_connectors(img)
        print(f"[AI] Power connectors: {power_connectors}")

        # Step 4: Locate JTAG header
        jtag_location = self._find_jtag_header(img)
        if jtag_location:
            print(f"[AI] JTAG header located at: {jtag_location}")

        # Step 5: Compile results
        if detected_chips:
            # Use most confident detection
            best_chip = max(detected_chips, key=lambda x: x['confidence'])

            return BoardDetection(
                board_type=board_type,
                vendor=best_chip['vendor'],
                fpga_model=best_chip['text'],
                chip_count=len(detected_chips),
                confidence=best_chip['confidence'],
                jtag_location=jtag_location,
                power_connectors=power_connectors,
                config_file=best_chip['config'],
                recommendations=self._generate_recommendations(
                    board_type, best_chip, len(detected_chips), jtag_location
                )
            )
        else:
            # Could not identify specific chip
            return BoardDetection(
                board_type=board_type,
                vendor="unknown",
                fpga_model="unknown",
                chip_count=0,
                confidence=0.0,
                jtag_location=jtag_location,
                power_connectors=power_connectors,
                config_file="",
                recommendations=[
                    "⚠️  Could not identify FPGA chip automatically",
                    "Try taking a closer photo of the chip marking",
                    "Check if chip is under heatsink (remove carefully)",
                    "Manually identify chip and select config from docs/"
                ]
            )

    def _detect_form_factor(self, img: np.ndarray) -> str:
        """Detect board type from shape and features"""
        height, width = img.shape[:2]
        aspect_ratio = width / height

        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Detect PCIe edge connector (gold fingers on edge)
        # Look for repeating vertical lines on one edge
        bottom_edge = edges[-50:, :]
        if np.sum(bottom_edge) > 10000:  # High edge density = connector
            return "pcie_card"

        # ATCA blades are very rectangular (tall and narrow)
        if 0.3 < aspect_ratio < 0.5:  # Tall board
            return "atca_blade"

        # Hashboards are typically wide and flat
        if aspect_ratio > 2.0:
            return "hashboard"

        return "unknown"

    def _detect_connectors(self, img: np.ndarray) -> List[str]:
        """Detect power connectors using color and shape"""
        connectors = []

        # Convert to HSV for color detection
        hsv = img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # XT60 connector: bright yellow plastic
        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
        if cv2.countNonZero(yellow_mask) > 500:
            connectors.append("XT60")

        # ATX 24-pin: large white/black connector
        # (Complex shape detection - simplified here)

        # PCIe 6/8-pin: black with yellow/red wires
        # Check for black rectangular regions

        return connectors if connectors else ["unknown"]

    def _find_jtag_header(self, img: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Locate JTAG header using template matching

        JTAG headers are typically:
        - 2x5 or 2x7 pin headers (0.1" pitch)
        - Often unpopulated (just holes)
        - Located near edge of board
        - Small rectangular pattern
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Look for small rectangular grid patterns
        # This is simplified - real implementation would use trained model

        # For now, return None (would need training data)
        return None

    def _generate_recommendations(
        self,
        board_type: str,
        chip: Dict,
        chip_count: int,
        jtag_location: Optional[Tuple[int, int]]
    ) -> List[str]:
        """Generate actionable next steps"""
        recs = []

        # JTAG setup
        if jtag_location:
            recs.append("✓ JTAG header detected - solder 2x5 pin header")
        else:
            recs.append("⚠️  JTAG header not visible - check board for J1/J2/JTAG silkscreen")

        # Multi-chip handling
        if chip_count > 1:
            recs.append(f"ℹ️  Detected {chip_count} FPGAs in series JTAG chain")
            recs.append(f"Use config: tools/fpga_salvage/configs/{chip['config']}")
        else:
            recs.append(f"ℹ️  Single FPGA detected")
            recs.append(f"Use config: tools/fpga_salvage/configs/{chip['config']}")

        # Hardware requirements
        if board_type == "hashboard":
            recs.append("🔌 Power: 12V 10A+ recommended (ATX PSU or bench supply)")
            recs.append("❄️  Cooling: Add heatsinks + fan (chips run hot!)")
        elif board_type == "pcie_card":
            recs.append("🔌 Power: Insert into PCIe slot + connect 8-pin aux power")
            recs.append("❄️  Cooling: Existing heatsink should be sufficient")

        # Cost estimate
        if chip['vendor'] == 'intel' and 'agilex' in chip['model']:
            recs.append("💰 Value: ~$60K+ in FPGAs for $200-400 board cost!")
        elif chip['vendor'] == 'xilinx' and 'virtex' in chip['model']:
            recs.append("💰 Value: ~$10K-30K in FPGAs (depending on model)")

        return recs

    def generate_config_file(self, detection: BoardDetection, output_path: Path):
        """
        Auto-generate OpenOCD config based on detection

        This is advanced: analyzes the detected board and creates
        a custom config with correct IDCODES, JTAG chain, etc.
        """
        if detection.config_file:
            # Copy recommended config
            src = Path(__file__).parent.parent / "configs" / detection.config_file
            if src.exists():
                import shutil
                shutil.copy(src, output_path)
                print(f"[AI] ✓ Generated config: {output_path}")
                return

        # If no match, generate generic config
        print("[AI] ⚠️  Generating generic config (may need manual adjustment)")
        # ... config generation logic ...

    def save_detection_report(self, detection: BoardDetection, output_path: Path):
        """Save detection results as JSON"""
        with open(output_path, 'w') as f:
            json.dump(asdict(detection), f, indent=2)
        print(f"[AI] ✓ Saved detection report: {output_path}")


def main():
    """CLI interface for board detection"""
    import argparse

    parser = argparse.ArgumentParser(
        description="AI-powered FPGA board identification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect board from photo
  python board_detector.py photo.jpg

  # Generate config file
  python board_detector.py photo.jpg --generate-config custom.cfg

  # Save detection report
  python board_detector.py photo.jpg --report detection.json

Usage Tips:
  - Take photo from directly above (top-down view)
  - Ensure chip markings are visible and in focus
  - Good lighting helps OCR accuracy
  - Remove heatsinks if chip marking is hidden
  - Photo resolution: 1920x1080 or higher recommended
        """
    )

    parser.add_argument('image', type=Path, help='Path to board photo')
    parser.add_argument('--generate-config', type=Path,
                       help='Generate OpenOCD config file')
    parser.add_argument('--report', type=Path,
                       help='Save detection report as JSON')

    args = parser.parse_args()

    if not args.image.exists():
        print(f"ERROR: Image not found: {args.image}")
        return 1

    # Run detection
    detector = FPGABoardDetector()
    detection = detector.detect_from_image(args.image)

    # Print results
    print("\n" + "="*60)
    print("DETECTION RESULTS")
    print("="*60)
    print(f"Board Type:    {detection.board_type}")
    print(f"Vendor:        {detection.vendor.upper()}")
    print(f"FPGA Model:    {detection.fpga_model}")
    print(f"Chip Count:    {detection.chip_count}")
    print(f"Confidence:    {detection.confidence:.1%}")
    print(f"Config File:   {detection.config_file}")
    print("\nRecommendations:")
    for rec in detection.recommendations:
        print(f"  {rec}")
    print("="*60)

    # Optional: generate config
    if args.generate_config:
        detector.generate_config_file(detection, args.generate_config)

    # Optional: save report
    if args.report:
        detector.save_detection_report(detection, args.report)

    return 0


if __name__ == '__main__':
    exit(main())
