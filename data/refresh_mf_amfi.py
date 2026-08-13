"""Rebuild data/mf_sector_flows.csv from AMFI monthly portfolio disclosures.

For each curated equity scheme it downloads the monthly portfolio (company,
sector, quantity, market value) for the latest month and for 3/6/12/24 months
ago, then computes sector-wise buying and selling by diffing the holdings
across each period. Results are written in INR crore.

Run with:
    python data/refresh_mf_amfi.py

Note: the AMFI portal endpoint has been unstable (it sometimes returns only the
scheme listing instead of holdings). If a request returns no holdings table the
script prints a warning and skips that (fund, month) pair so it degrades
gracefully.
"""
import re
import sys
import time
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from settings import SETTINGS, data_file

OUT_PATH = data_file("mf_flows")

_CFG = SETTINGS["data_refresh"]["mf_amfi"]
AMFI_PORTFOLIO_URL = _CFG["portfolio_url"]
PERIODS = _CFG["periods"]
MONTHS_BACK = _CFG["months_back"]
RETRIES = _CFG["retries"]
BACKOFF_SECONDS = _CFG["backoff_seconds"]
TIMEOUT_SECONDS = _CFG["timeout_seconds"]
FUND_SLEEP_SECONDS = _CFG["fund_sleep_seconds"]
CURATED_FUNDS = _CFG["curated_funds"]
SECTOR_MAP = _CFG["sector_map"]

HEADERS = {
    "User-Agent": _CFG["user_agent"]
}


def shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    total = year * 12 + (month - 1) - delta
    return total // 12, total % 12 + 1


def _fetch(url, params, retries=RETRIES, backoff_seconds=BACKOFF_SECONDS):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT_SECONDS)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            last_err = exc
            time.sleep(backoff_seconds * attempt)
    print(f"  ! request failed after {retries} attempts: {last_err}")
    return None


def _find_header_row(df):
    """Locate the row whose cells look like a holdings header."""
    for i in range(min(3, len(df))):
        joined = " ".join(str(v).strip().lower() for v in df.iloc[i])
        if re.search(r"company|instrument", joined) and "market value" in joined:
            return i
    return None


def _to_numeric(series):
    cleaned = series.astype(str).str.replace(",", "").str.replace("%", "")
    return pd.to_numeric(cleaned, errors="coerce")


def _normalize_holdings(text):
    """Extract a holdings table (company, sector, quantity, market value, %AUM)
    from an AMFI disclosure response, or return None if no holdings are present.
    """
    if not text:
        return None
    try:
        if "<table" in text.lower():
            df = pd.read_html(StringIO(text), header=None)[0]
        else:
            df = pd.read_csv(StringIO(text), encoding="utf-8-sig", on_bad_lines="skip")
    except Exception:
        return None
    if df is None or df.empty:
        return None

    header_idx = _find_header_row(df)
    if header_idx is not None:
        df.columns = [str(v).strip() for v in df.iloc[header_idx]]
        df = df.iloc[header_idx + 1 :].reset_index(drop=True)

    cols = {str(c).strip().lower(): c for c in df.columns}
    company_col = next((c for c in cols if re.search(r"company|instrument|name of the", c)), None)
    mv_col = next(
        (c for c in cols if "market value" in c and ("lakh" in c or "rs" in c or "(" in c)),
        None,
    )
    if company_col is None or mv_col is None:
        return None

    sector_col = next((c for c in cols if c == "sector" or "sector" in c), None)
    qty_col = next((c for c in cols if re.search(r"quantity|units|no\.?\s+of", c)), None)
    pct_col = next((c for c in cols if ("%" in c or "aum" in c or "net assets" in c)), None)

    def col(key):
        return df[cols[key]]

    out = pd.DataFrame({"company": col(company_col).astype(str).str.strip()})
    out["sector"] = col(sector_col).astype(str).str.strip() if sector_col else "Other"
    out["quantity"] = _to_numeric(col(qty_col)) if qty_col else None
    out["market_value_lakhs"] = _to_numeric(col(mv_col)).fillna(0.0)
    out["pct_aum"] = _to_numeric(col(pct_col)) if pct_col else None
    return out[out["company"] != ""].dropna(subset=["company"])


def fetch_scheme_portfolio(code, year, month):
    """Return normalized holdings for a scheme/month, or None."""
    text = _fetch(
        AMFI_PORTFOLIO_URL,
        {"mession": "24", "mession_code": code, "mf": month, "yr": year, "myession": "S"},
    )
    return _normalize_holdings(text)


def compute_period_flows(snapshots):
    """snapshots: { (year, month): DataFrame[company, sector, market_value_lakhs] }

    Returns a list of dicts with sector-level buy/sell for every period that has
    both a latest and a 'months ago' snapshot.
    """
    keys = sorted(snapshots.keys(), reverse=True)
    if not keys:
        return []
    latest_key = keys[0]
    latest = snapshots[latest_key]
    rows = []
    for label, delta_months in PERIODS.items():
        prev_key = shift_month(latest_key[0], latest_key[1], delta_months)
        prev = snapshots.get(prev_key)
        if prev is None:
            print(f"  ! no snapshot {prev_key} for period {label}; skipping")
            continue
        merged = pd.merge(
            latest,
            prev,
            on="company",
            how="outer",
            suffixes=("_l", "_p"),
        )
        merged["delta"] = (merged["market_value_lakhs_l"].fillna(0.0) - merged["market_value_lakhs_p"].fillna(0.0)) / 100.0
        merged["sector"] = merged["sector_l"].fillna(merged["sector_p"])
        merged["sector"] = merged["sector"].map(lambda s: SECTOR_MAP.get(s, s))
        by_sector = (
            merged.groupby("sector")["delta"]
            .agg(buy=lambda s: s[s > 0].sum(), sell=lambda s: -s[s < 0].sum(), net="sum")
            .reset_index()
        )
        for _, rec in by_sector.iterrows():
            rows.append(
                {
                    "sector": rec["sector"],
                    "period": label,
                    "buy_crore": round(rec["buy"]),
                    "sell_crore": round(rec["sell"]),
                    "net_crore": round(rec["net"]),
                }
            )
    return rows


def main():
    today = date.today()
    fetch_keys = [today.year * 12 + today.month - 1 - delta for delta in sorted(set(MONTHS_BACK))]
    months = [(k // 12, k % 12 + 1) for k in fetch_keys]

    snapshots = {}
    for fund in CURATED_FUNDS:
        print(f"Processing {fund['name']} ({fund['code']})")
        for year, month in months:
            print(f"  month {year:04d}-{month:02d} ...", end=" ")
            holdings = fetch_scheme_portfolio(fund["code"], year, month)
            if holdings is None or holdings.empty:
                print("no holdings returned")
                continue
            snapshots[(year, month)] = pd.concat(
                [snapshots[(year, month)], holdings], ignore_index=True
            ) if (year, month) in snapshots else holdings
            print(f"{len(holdings)} rows")
        time.sleep(FUND_SLEEP_SECONDS)

    if not snapshots:
        print("No holdings could be fetched. The AMFI endpoint may be returning "
              "only the scheme listing; please try again later.")
        return

    rows = compute_period_flows(snapshots)
    if not rows:
        print("Could not compute any period flows from the available snapshots.")
        return

    df = pd.DataFrame(rows, columns=["sector", "period", "buy_crore", "sell_crore", "net_crore"])
    df = df.sort_values(["period", "sector"]).reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_PATH}")
    print(df.groupby("period")[["buy_crore", "sell_crore", "net_crore"]].sum().to_string())


if __name__ == "__main__":
    main()
