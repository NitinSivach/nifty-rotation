"""Tests for data/generate_mf_snapshot.py (deterministic bundled MF snapshot)."""
import pandas as pd

from data.generate_mf_snapshot import build_weights, main


def test_build_weights_covers_all_sectors_and_normalizes():
    weights = build_weights()
    # Every nifty50_map sector must appear with a normalized weight.
    assert len(weights) >= 10
    assert all(w > 0 for w in weights.values())
    assert sum(weights.values()) == 1.0


def test_build_weights_prefers_configured_weights():
    weights = build_weights()
    # Financials carries the largest configured weight in config.json.
    assert max(weights, key=weights.get) == "Financials"


def test_main_writes_deterministic_csv(tmp_path, monkeypatch):
    # Keep output inside tmp_path and validate shape/content.
    out = tmp_path / "mf_sector_flows.csv"
    monkeypatch.setattr(
        "data.generate_mf_snapshot.OUT_PATH",
        out,
    )
    monkeypatch.setattr(
        "data.generate_mf_snapshot.DATA_DIR",
        tmp_path,
    )
    main()

    assert out.exists()
    df = pd.read_csv(out)
    assert list(df.columns) == ["sector", "period", "buy_crore", "sell_crore", "net_crore"]
    # One row per configured period per sector.
    periods = ["3M", "6M", "1Y", "2Y"]
    assert set(df["period"]) == set(periods)
    assert len(df) == len(df["period"].unique()) * df["sector"].nunique()


def test_main_output_is_reproducible(tmp_path, monkeypatch):
    out = tmp_path / "a.csv"
    monkeypatch.setattr("data.generate_mf_snapshot.OUT_PATH", out)
    monkeypatch.setattr("data.generate_mf_snapshot.DATA_DIR", tmp_path)
    main()
    first = pd.read_csv(out)

    out2 = tmp_path / "b.csv"
    monkeypatch.setattr("data.generate_mf_snapshot.OUT_PATH", out2)
    main()
    second = pd.read_csv(out2)

    pd.testing.assert_frame_equal(first, second)