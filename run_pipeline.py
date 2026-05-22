"""
AgriSense Commons — Standalone Pipeline Runner
================================================
Run this directly from the terminal to fetch sensor data,
run the full pipeline, and save the result as a JSON file.

Usage:
    python run_pipeline.py
    python run_pipeline.py --kit 1001 --sensor ftTemp
    python run_pipeline.py --kit 1001 --sensor ftTemp --lat 47.7981 --lon 13.0456
    python run_pipeline.py --kit 1001 --sensor ftSoilMoist --output my_results/

The JSON file is saved to the outputs/ folder by default.
"""

import argparse
import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.pipeline import run_pipeline, save_to_json


def main():
    parser = argparse.ArgumentParser(
        description="Run the AgriSense pipeline and save results to JSON."
    )
    parser.add_argument("--kit",    default="1001",    help="TeleAgriCulture Kit ID (default: 1001)")
    parser.add_argument("--sensor", default="ftTemp",  help="Sensor name (default: ftTemp)")
    parser.add_argument("--lat",    default=47.7981,   type=float, help="Board latitude")
    parser.add_argument("--lon",    default=13.0456,   type=float, help="Board longitude")
    parser.add_argument("--output", default="outputs", help="Output folder (default: outputs/)")
    args = parser.parse_args()

    owm_key = os.getenv("OWM_API_KEY", "")

    print(f"\nAgriSense Commons — Pipeline Runner")
    print(f"{'─' * 40}")
    print(f"Kit ID:  {args.kit}")
    print(f"Sensor:  {args.sensor}")
    print(f"Lat/Lon: {args.lat}, {args.lon}")
    print(f"OWM key: {'set' if owm_key else 'not set — skipping weather comparison'}")
    print(f"Output:  {args.output}/")
    print(f"{'─' * 40}")

    print("\nFetching sensor data...")
    result = run_pipeline(
        kit_id=args.kit,
        sensor_name=args.sensor,
        lat=args.lat,
        lon=args.lon,
        owm_api_key=owm_key or None,
    )

    if "error" in result:
        print(f"\nError: {result['error']}")
        sys.exit(1)

    filepath = result.get("json_file") or save_to_json(result, args.output)

    print(f"\nResults:")
    print(f"  Records fetched:   {result['record_count']}")
    print(f"  Date range:        {result['date_from'][:10]} → {result['date_to'][:10]}")
    print(f"  Current state:     {result['current_state']}")
    print(f"  Board mean:        {result['board_profile'].get('mean')} (sensor units)")
    if result.get('deviation', {}).get('available'):
        print(f"  vs outdoor:        {result['deviation']['note']}")
    if result.get('recent_anomalies'):
        print(f"  Anomalies found:   {len(result['recent_anomalies'])}")
    print(f"\nSummary: {result['summary']}")
    print(f"\nJSON saved to: {filepath}")
    print()


if __name__ == "__main__":
    main()
