import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.refresh_mf_amfi import _normalize_holdings, compute_period_flows, shift_month

HOLDINGS_HTML = """
<html><body><table>
<tr><td>Company Name</td><td>Sector</td><td>Quantity</td><td>Market Value (Rs. Lakh)</td><td>% to Net Assets</td></tr>
<tr><td>RELIANCE INDUSTRIES LTD</td><td>Energy</td><td>1,00,000</td><td>15,000.50</td><td>8.10</td></tr>
<tr><td>HDFC BANK LTD</td><td>Banks</td><td>50,000</td><td>9,200.00</td><td>4.90</td></tr>
</table></body></html>
"""


def test_shift_month():
    assert shift_month(2026, 8, 3) == (2026, 5)
    assert shift_month(2026, 1, 1) == (2025, 12)
    assert shift_month(2024, 8, 24) == (2022, 8)


def test_normalize_holdings_parses_html_table():
    df = _normalize_holdings(HOLDINGS_HTML)
    assert df is not None
    assert len(df) == 2
    assert set(df["company"]) == {"RELIANCE INDUSTRIES LTD", "HDFC BANK LTD"}
    assert df["market_value_lakhs"].sum() == pytest.approx(24200.5)
    assert df.loc[0, "sector"] == "Energy"


def test_normalize_holdings_ignores_scheme_metadata():
    csv = "AMC,Code,Scheme Name,Scheme Type\nDSP,100077,DSP Bond Fund,Open Ended\n"
    assert _normalize_holdings(csv) is None


def test_normalize_holdings_handles_empty():
    assert _normalize_holdings("") is None
    assert _normalize_holdings(None) is None


def test_compute_period_flows_gross_buy_sell():
    latest = pd.DataFrame(
        {
            "company": ["RELIANCE INDUSTRIES LTD", "HDFC BANK LTD", "TCS LTD"],
            "sector": ["Energy", "Banks", "IT"],
            "market_value_lakhs": [20000.0, 9000.0, 5000.0],
        }
    )
    prev = pd.DataFrame(
        {
            "company": ["RELIANCE INDUSTRIES LTD", "HDFC BANK LTD", "ICICI BANK LTD"],
            "sector": ["Energy", "Banks", "Banks"],
            "market_value_lakhs": [12000.0, 12000.0, 3000.0],
        }
    )
    snapshots = {(2026, 8): latest, (2026, 5): prev}
    rows = compute_period_flows(snapshots)
    by_sector = {(r["sector"], r["period"]): r for r in rows}

    energy = by_sector[("Energy", "3M")]
    assert energy["buy_crore"] == 80
    assert energy["sell_crore"] == 0
    assert energy["net_crore"] == 80

    financials = by_sector[("Financials", "3M")]
    assert financials["buy_crore"] == 0
    assert financials["sell_crore"] == 60
    assert financials["net_crore"] == -60

    it = by_sector[("IT", "3M")]
    assert it["buy_crore"] == 50
    assert it["net_crore"] == 50


def test_compute_period_flows_skips_missing_snapshot():
    latest = pd.DataFrame(
        {"company": ["A LTD"], "sector": ["Energy"], "market_value_lakhs": [100.0]}
    )
    rows = compute_period_flows({(2026, 8): latest})
    assert rows == []
