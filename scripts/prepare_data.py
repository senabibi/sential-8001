"""
Prepare raw Kraken OHLCV data for SENTINEL-8004.

Copies the relevant BTC/USD and ETH/USD files from the raw data directory
into data/historical/, then validates them and prints a summary.

Usage:
    python scripts/prepare_data.py --raw-dir /home/nursena/Downloads/Kraken_OHLCVT/master_q4

Output files in data/historical/:
    XBTUSD_60.csv     — BTC/USD hourly  (primary: ~96k candles, 2013-2025)
    ETHUSD_60.csv     — ETH/USD hourly  (primary: ~87k candles, 2015-2025)
    XBTUSD_1440.csv   — BTC/USD daily   (for broad context)
    ETHUSD_1440.csv   — ETH/USD daily   (for broad context)
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_data")

DEST_DIR = Path(__file__).resolve().parents[1] / "data" / "historical"

# Files we want: (source filename, destination filename)
TARGETS = [
    ("XBTUSD_60.csv",   "XBTUSD_60.csv"),    # BTC/USD 1H — primary
    ("ETHUSD_60.csv",   "ETHUSD_60.csv"),    # ETH/USD 1H — primary
    ("XBTUSD_1440.csv", "XBTUSD_1440.csv"),  # BTC/USD daily
    ("ETHUSD_1440.csv", "ETHUSD_1440.csv"),  # ETH/USD daily
]

# Key historical events we expect to be present in the data
KEY_EVENTS = [
    (datetime(2022, 5, 1, tzinfo=timezone.utc),  datetime(2022, 5, 31, tzinfo=timezone.utc),  "LUNA collapse (May 2022)"),
    (datetime(2022, 11, 1, tzinfo=timezone.utc), datetime(2022, 11, 30, tzinfo=timezone.utc), "FTX collapse (Nov 2022)"),
    (datetime(2024, 4, 1, tzinfo=timezone.utc),  datetime(2024, 5, 31, tzinfo=timezone.utc),  "Halving + correction (Apr-May 2024)"),
    (datetime(2024, 8, 1, tzinfo=timezone.utc),  datetime(2024, 8, 31, tzinfo=timezone.utc),  "Yen carry trade flash crash (Aug 2024)"),
]


def copy_files(raw_dir: Path) -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    for src_name, dst_name in TARGETS:
        src = raw_dir / src_name
        dst = DEST_DIR / dst_name

        if not src.exists():
            logger.warning("Source not found: %s — skipping", src)
            continue

        shutil.copy2(src, dst)
        logger.info("Copied %s → %s", src_name, dst)


def validate_files() -> None:
    from src.processing.cleaner import load_ohlcv_df

    logger.info("Validating files...")
    for _, dst_name in TARGETS:
        path = DEST_DIR / dst_name
        if not path.exists():
            logger.warning("Missing: %s", path)
            continue

        df = load_ohlcv_df(path)
        if df.empty:
            logger.error("Empty after load: %s", path)
            continue

        start_dt = df["datetime"].iloc[0]
        end_dt   = df["datetime"].iloc[-1]
        logger.info(
            "%s: %d candles | %s → %s | close range [%.2f, %.2f]",
            dst_name, len(df),
            start_dt.strftime("%Y-%m-%d"),
            end_dt.strftime("%Y-%m-%d"),
            df["close"].min(), df["close"].max(),
        )

        # Check key events are present
        for ev_start, ev_end, label in KEY_EVENTS:
            ev_start_ts = int(ev_start.timestamp())
            ev_end_ts   = int(ev_end.timestamp())
            count = ((df["timestamp"] >= ev_start_ts) & (df["timestamp"] <= ev_end_ts)).sum()
            status = "OK" if count > 0 else "MISSING"
            logger.info("  [%s] %s — %d candles", status, label, count)


def print_summary() -> None:
    from src.processing.cleaner import load_ohlcv_df, detect_significant_events

    logger.info("\n=== EVENT SUMMARY ===")
    for _, dst_name in [("XBTUSD_60.csv", "XBTUSD_60.csv"), ("ETHUSD_60.csv", "ETHUSD_60.csv")]:
        path = DEST_DIR / dst_name
        if not path.exists():
            continue

        df = load_ohlcv_df(path)
        events = detect_significant_events(df, price_drop_threshold=-5.0)
        logger.info(
            "%s — significant events (drops >5%%): %d",
            dst_name, len(events[events["event_type"].str.contains("price_drop", na=False)])
        )

        # Print the 5 worst single-candle drops
        drops = events[events["event_type"].str.contains("price_drop", na=False)].nsmallest(5, "price_chg_pct")
        for _, row in drops.iterrows():
            logger.info(
                "  %s  close=%.2f  chg=%.2f%%  rsi=%.1f  vol_z=%.1f  type=%s",
                row["datetime"].strftime("%Y-%m-%d %H:%M"),
                row["close"], row["price_chg_pct"],
                row["rsi14"], row["vol_zscore"],
                row["event_type"],
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SENTINEL-8004 historical data")
    parser.add_argument(
        "--raw-dir",
        default="/home/nursena/Downloads/Kraken_OHLCVT/master_q4",
        help="Directory containing raw Kraken CSV files",
    )
    parser.add_argument("--skip-copy", action="store_true", help="Skip copy (already done)")
    parser.add_argument("--skip-summary", action="store_true", help="Skip event summary")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not args.skip_copy:
        if not raw_dir.exists():
            logger.error("Raw data directory not found: %s", raw_dir)
            sys.exit(1)
        copy_files(raw_dir)

    validate_files()

    if not args.skip_summary:
        print_summary()

    logger.info("Done. Data ready in: %s", DEST_DIR)


if __name__ == "__main__":
    main()
