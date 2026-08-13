import json
from pathlib import Path

import yfinance as yf

from settings import data_file


def load_nifty50_map(path=None) -> dict:
    if path is None:
        path = data_file("nifty50_map")
    with Path(path).open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def check_ticker(sym: str) -> tuple[int, str | None]:
    """Return (rows_in_last_month, error_message) for a ticker symbol."""
    try:
        hist = yf.Ticker(sym).history(period="1mo")
        return len(hist), None
    except Exception as exc:
        return 0, str(exc)


def main():
    mapping = load_nifty50_map()
    tickers = list(mapping.keys())
    print(f"Checking {len(tickers)} tickers from nifty50_map.json...\n")

    missing = []
    for sym in tickers:
        rows, error = check_ticker(sym)
        if error is not None or rows == 0:
            missing.append(sym)
            detail = f"ERROR -> {error}" if error else "NO recent data (empty history)"
            print(f"  {sym}: {detail}")
        else:
            print(f"  {sym}: OK ({rows} rows in last month)")

    print()
    if missing:
        print(f"Missing/stale tickers ({len(missing)}): {', '.join(missing)}")
    else:
        print("All tickers look healthy.")


if __name__ == "__main__":
    main()
