#!/usr/bin/env python3
"""
Ara Consult CLI - command-line interface for quick analysis.

Usage:
    python -m ara_consult.cli emails data.csv --target=open_rate
    python -m ara_consult.cli pricing data.csv --target=conversion_rate
    python -m ara_consult.cli kpis data.csv --target=revenue

"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Ara Consult - Pattern analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze email subject lines
    python -m ara_consult.cli emails campaigns.csv --target=open_rate

    # Analyze pricing data
    python -m ara_consult.cli pricing sales.csv --target=conversion_rate

    # Analyze generic KPIs
    python -m ara_consult.cli kpis dashboard.csv --target=revenue

    # Output to file
    python -m ara_consult.cli emails campaigns.csv --target=open_rate --output=report.md
        """,
    )

    parser.add_argument(
        "workflow",
        choices=["emails", "pricing", "kpis"],
        help="Type of analysis to run",
    )
    parser.add_argument(
        "csv",
        help="Path to CSV file with data",
    )
    parser.add_argument(
        "--target",
        "-t",
        default="open_rate",
        help="Target column to analyze (default: open_rate)",
    )
    parser.add_argument(
        "--client",
        "-c",
        default="Client",
        help="Client name for report",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (markdown). If not specified, prints to stdout.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of markdown",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only output the report, no extra messages",
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.csv).exists():
        print(f"Error: File not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    # Run the appropriate workflow
    if not args.quiet:
        print(f"\n🔍 Ara is analyzing {args.csv}...\n", file=sys.stderr)

    try:
        if args.workflow == "emails":
            from .workflows.email_subjects import analyze_email_subjects
            result = analyze_email_subjects(
                args.csv,
                target_col=args.target,
                client_name=args.client,
            )
        elif args.workflow == "pricing":
            from .workflows.pricing import analyze_pricing
            result = analyze_pricing(
                args.csv,
                target_col=args.target,
                client_name=args.client,
            )
        else:  # kpis
            from .workflows.generic_kpis import analyze_kpis
            result = analyze_kpis(
                args.csv,
                target_col=args.target,
                client_name=args.client,
            )
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)

    # Output results
    if args.json:
        output = json.dumps(result, indent=2, default=str)
    else:
        output = result["report_markdown"]

    if args.output:
        Path(args.output).write_text(output)
        if not args.quiet:
            print(f"✅ Report saved to {args.output}", file=sys.stderr)
    else:
        print(output)

    # Print summary stats if not quiet and not JSON
    if not args.quiet and not args.json:
        print("\n" + "=" * 60, file=sys.stderr)
        print("📊 Quick Stats:", file=sys.stderr)
        print(f"   Patterns found: {len(result['patterns'].get('top_patterns', []))}", file=sys.stderr)
        print(f"   Experiments suggested: {len(result['experiments'])}", file=sys.stderr)
        print("=" * 60 + "\n", file=sys.stderr)


if __name__ == "__main__":
    main()
