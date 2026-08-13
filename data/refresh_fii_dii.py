"""Refresh data/fii_dii_flows.csv from the NSE FII/DII activity API.

The NSE publishes daily cash-market net FII/FPI and DII figures. This script
fetches the latest values with a browser-like session (cookies + headers) and
merges them into a rolling history CSV that the dashboard reads.

Delivery-volume data is also attempted via the NSE daily delivery-position
endpoint; that endpoint is frequently WAF-blocked (HTTP 503), so the script
degrades gracefully: if the fetch fails the existing CSV (if any) is left in
place and a warning is printed.

Run with:
    python data/refresh_fii_dii.py

Outputs:
    data/fii_dii_flows.csv   columns: date, category, buy_value_crore, sell_value_crore, net_value_crore
    data/delivery_volume.csv columns: date, symbol, series, security, quantity, deliverable_quantity, delivery_pct  (best effort)
"""
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from settings import SETTINGS, data_file

FII_DII_OUT = data_file("fii_dii")
DELIVERY_OUT = data_file("delivery_volume")

_CFG = SETTINGS["data_refresh"]["fii_dii"]
NSE_HOME = _CFG["nse_home"]
FII_DII_URL = _CFG["fiidii_url"]
DELIVERY_URL = _CFG["delivery_url"]
TIMEOUT_SECONDS = _CFG["timeout_seconds"]
DELIVERY_TIMEOUT_SECONDS = _CFG["delivery_timeout_seconds"]
HOMEPAGE_SLEEP_SECONDS = _CFG["homepage_sleep_seconds"]

HEADERS = {
    "User-Agent": _CFG["user_agent"],
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def nse_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    # First hit the homepage so NSE sets the cookies the API expects.
    session.get(NSE_HOME, timeout=TIMEOUT_SECONDS)
    time.sleep(HOMEPAGE_SLEEP_SECONDS)
    return session


def parse_fii_dii_payload(payload):
    """Normalize the raw NSE FII/DII list into a DataFrame, or raise ValueError."""
    if not isinstance(payload, list) or not payload:
        raise ValueError("FII/DII payload was not a list of records")
    rows = []
    for rec in payload:
        if "category" not in rec:
            continue
        rows.append(
            {
                "date": rec.get("date"),
                "category": str(rec.get("category")).upper(),
                "buy_value_crore": float(rec.get("buyValue", 0) or 0),
                "sell_value_crore": float(rec.get("sellValue", 0) or 0),
                "net_value_crore": float(rec.get("netValue", 0) or 0),
            }
        )
    if not rows:
        raise ValueError("No FII/DII records parsed from payload")
    return pd.DataFrame(rows)


def fetch_fii_dii(session: requests.Session):
    """Return a DataFrame of {date, category, buy/sell/net} crore, or raise."""
    resp = session.get(FII_DII_URL, timeout=TIMEOUT_SECONDS)
    resp.raise_for_status()
    return parse_fii_dii_payload(resp.json())


def merge_fii_dii(new_df: pd.DataFrame, out_path: Path):
    """Append new rows, overwriting any existing (date, category) pair."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame()
    if out_path.exists():
        try:
            history = pd.read_csv(out_path)
        except Exception:
            history = pd.DataFrame()
    merged = pd.concat([history, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "category"], keep="last")
    merged = merged.sort_values(["date", "category"]).reset_index(drop=True)
    merged.to_csv(out_path, index=False)
    return merged


def fetch_delivery(session: requests.Session, target: date):
    """Return a DataFrame of delivery stats for ``target``, or raise."""
    params = {"date": target.strftime("%d-%m-%Y")}
    resp = session.get(DELIVERY_URL, params=params, timeout=DELIVERY_TIMEOUT_SECONDS)
    resp.raise_for_status()
    payload = resp.json()
    rows = payload.get("data") or []
    if not rows:
        raise ValueError("Delivery payload contained no rows")
    out = []
    for rec in rows:
        qty = float(rec.get("quantity", 0) or 0)
        deliv = float(rec.get("deliverableQuantity", 0) or 0)
        out.append(
            {
                "date": target.isoformat(),
                "symbol": rec.get("symbol"),
                "series": rec.get("series"),
                "security": rec.get("security"),
                "quantity": qty,
                "deliverable_quantity": deliv,
                "delivery_pct": (deliv / qty) if qty else None,
            }
        )
    return pd.DataFrame(out)


def last_weekday() -> date:
    today = date.today()
    days = 0
    while days < 7 and (today - timedelta(days=days)).weekday() >= 5:
        days += 1
    return today - timedelta(days=days)


def main() -> None:
    print("Connecting to NSE ...")
    session = nse_session()

    print("Fetching FII/DII flows ...")
    try:
        fii_dii = fetch_fii_dii(session)
    except Exception as exc:
        print(f"  ! FII/DII fetch failed: {exc}")
        print("  Leaving any existing data/fii_dii_flows.csv in place.")
        fii_dii = None

    if fii_dii is not None and not fii_dii.empty:
        merged = merge_fii_dii(fii_dii, FII_DII_OUT)
        print(f"  Wrote {len(merged)} rows to {FII_DII_OUT}")
        latest = merged.sort_values("date").groupby("category").tail(1)
        for _, row in latest.iterrows():
            print(
                f"  {row['category']:8s} {row['date']}: "
                f"net {row['net_value_crore']:,.1f} crore "
                f"(buy {row['buy_value_crore']:,.1f} / sell {row['sell_value_crore']:,.1f})"
            )

    print("Attempting delivery-volume report (often WAF-blocked) ...")
    try:
        delivery = fetch_delivery(session, last_weekday())
        DELIVERY_OUT.parent.mkdir(parents=True, exist_ok=True)
        delivery.to_csv(DELIVERY_OUT, index=False)
        print(f"  Wrote {len(delivery)} rows to {DELIVERY_OUT}")
    except Exception as exc:
        print(f"  ! Delivery fetch failed ({exc}).")
        print("  This endpoint is frequently blocked by NSE's WAF; the dashboard "
              "will show 'unavailable' until NSE unblocks it.")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"Network error: {exc}")
        sys.exit(1)
