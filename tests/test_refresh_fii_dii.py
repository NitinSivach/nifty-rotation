import pandas as pd
import pytest

from data import refresh_fii_dii as rfd


def test_parse_fii_dii_payload_normalizes_rows():
    payload = [
        {"buyValue": "15006.72", "category": "DII", "date": "11-Aug-2026",
         "netValue": "24.77", "sellValue": "14981.95"},
        {"buyValue": "14628.47", "category": "FII/FPI", "date": "11-Aug-2026",
         "netValue": "258.55", "sellValue": "14369.92"},
    ]
    df = rfd.parse_fii_dii_payload(payload)
    assert list(df.columns) == [
        "date", "category", "buy_value_crore", "sell_value_crore", "net_value_crore"
    ]
    assert df["category"].tolist() == ["DII", "FII/FPI"]
    assert df["net_value_crore"].tolist() == [24.77, 258.55]
    assert df["date"].tolist() == ["11-Aug-2026", "11-Aug-2026"]


def test_parse_fii_dii_payload_handles_missing_values():
    payload = [{"category": "DII", "date": "10-Aug-2026"}]
    df = rfd.parse_fii_dii_payload(payload)
    assert df["buy_value_crore"].iloc[0] == 0.0
    assert df["net_value_crore"].iloc[0] == 0.0


def test_parse_fii_dii_payload_rejects_bad_shape():
    with pytest.raises(ValueError, match="list of records"):
        rfd.parse_fii_dii_payload({})
    with pytest.raises(ValueError, match="list of records"):
        rfd.parse_fii_dii_payload([])
    with pytest.raises(ValueError, match="No FII/DII"):
        rfd.parse_fii_dii_payload([{"other": "x"}])


def test_merge_fii_dii_overwrites_same_day_category(tmp_path):
    out = tmp_path / "flows.csv"
    day1 = pd.DataFrame(
        {
            "date": ["11-Aug-2026", "11-Aug-2026"],
            "category": ["DII", "FII/FPI"],
            "buy_value_crore": [15006.72, 14628.47],
            "sell_value_crore": [14981.95, 14369.92],
            "net_value_crore": [24.77, 258.55],
        }
    )
    rfd.merge_fii_dii(day1, out)

    # Re-fetch for the same day with newer numbers must replace the old rows.
    day1b = pd.DataFrame(
        {
            "date": ["11-Aug-2026", "12-Aug-2026"],
            "category": ["DII", "DII"],
            "buy_value_crore": [9999.0, 10000.0],
            "sell_value_crore": [5000.0, 6000.0],
            "net_value_crore": [4999.0, 4000.0],
        }
    )
    merged = rfd.merge_fii_dii(day1b, out)
    assert len(merged) == 3  # 1 replaced + 1 appended
    rows = merged.set_index(["date", "category"])
    assert rows.loc[("11-Aug-2026", "DII"), "net_value_crore"] == 4999.0
    assert rows.loc[("12-Aug-2026", "DII"), "net_value_crore"] == 4000.0
    assert rows.loc[("11-Aug-2026", "FII/FPI"), "net_value_crore"] == 258.55
