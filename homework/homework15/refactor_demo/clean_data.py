#!/usr/bin/env python3
"""
Refactor Demo — Clean & Save CSV (CLI)
- Reads an input CSV, drops rows with any NA in required columns, and writes a clean CSV.
- Logs to refactor_demo/clean_data.log
Usage:
  python clean_data.py --input data/raw.csv --output data/clean.csv --required col1 col2
"""
import argparse, logging
from pathlib import Path
import pandas as pd

LOG = Path(__file__).with_suffix(".log")
logging.basicConfig(filename=LOG, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

def clean_and_save(input_csv: str, output_csv: str, required_cols=None) -> int:
    p_in, p_out = Path(input_csv), Path(output_csv)
    logging.info("start clean_and_save input=%s output=%s required=%s", p_in, p_out, required_cols)
    df = pd.read_csv(p_in)
    if required_cols:
        before = len(df)
        df = df.dropna(subset=required_cols)
        logging.info("dropped rows with NA in %s: %d -> %d", required_cols, before, len(df))
    p_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p_out, index=False)
    logging.info("wrote %s rows to %s", len(df), p_out)
    logging.info("done")
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input CSV")
    ap.add_argument("--output", required=True, help="Path to output CSV")
    ap.add_argument("--required", nargs="*", default=[], help="Column names that must be non-null")
    args = ap.parse_args()
    return clean_and_save(args.input, args.output, args.required)

if __name__ == "__main__":
    raise SystemExit(main())
