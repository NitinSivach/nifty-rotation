import numpy as np
import pandas as pd
import pytest

import backtest


def deterministic_close(n=300):
    """Three sectors: one steadily rising, one falling, one flat."""
    index = pd.date_range("2023-01-02", periods=n, freq="B")
    a = 100 * np.linspace(1.0, 1.5, n)
    b = 100 * np.linspace(1.0, 0.7, n)
    c = np.full(n, 100.0)
    return pd.DataFrame({"A.NS": a, "B.NS": b, "C.NS": c}, index=index)


def make_ohlcv(close):
    """Neutral CMF (mfm = 0 on every row) so momentum drives the score."""
    ohlcv = {}
    for ticker in close.columns:
        df = close[[ticker]].rename(columns={ticker: "Close"})
        df["Open"] = df["Close"]
        df["High"] = df["Close"] * 1.02
        df["Low"] = df["Close"] * 0.98
        df["Volume"] = 1_000_000.0
        ohlcv[ticker] = df[["Open", "High", "Low", "Close", "Volume"]]
    return ohlcv


SECTOR_MAP = {"A.NS": "Grow", "B.NS": "Fall", "C.NS": "Flat"}


def make_ohlcv_positive_accumulation(n=30):
    index = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": np.full(n, 105.0),
            "High": np.full(n, 110.0),
            "Low": np.full(n, 90.0),
            "Close": np.full(n, 105.0),
            "Volume": np.full(n, 1000.0),
        },
        index=index,
    )


def test_rolling_cmf_positive_steady_flow():
    cmf = backtest.rolling_cmf(make_ohlcv_positive_accumulation(), 10)
    assert cmf.iloc[-1] == pytest.approx(0.5)


def test_rolling_cmf_neutral_when_close_centered():
    close = deterministic_close(60)
    cmf = backtest.rolling_cmf(make_ohlcv(close)["A.NS"], 20)
    assert cmf.dropna().abs().max() < 1e-9


def test_rolling_cmf_insufficient_data_is_nan():
    cmf = backtest.rolling_cmf(make_ohlcv_positive_accumulation(5), 10)
    assert cmf.isna().all()


def test_composite_scores_ranks_by_momentum():
    close = deterministic_close()
    ohlcv = make_ohlcv(close)
    comp = backtest.composite_scores(close, ohlcv, SECTOR_MAP, 20, 20)
    last = comp.iloc[-1].dropna().sort_values(ascending=False)
    assert list(last.index) == ["Grow", "Flat", "Fall"]


def test_composite_scores_mf_overlay_shifts_ranking():
    close = deterministic_close()
    ohlcv = make_ohlcv(close)
    flows = pd.DataFrame(
        {"sector": ["Grow", "Flat", "Fall"], "net_crore": [-100, 0, 500]}
    )
    comp = backtest.composite_scores(
        close, ohlcv, SECTOR_MAP, 20, 20, 0.6, 0.4, mf_flows=flows, mf_weight=0.3
    )
    last = comp.iloc[-1].dropna().sort_values(ascending=False)
    # the huge 'Fall' flow overlay should not fully outweigh momentum (0.6 weight)
    assert last.index[0] == "Grow"


def test_run_backtest_beats_benchmark_on_trending_sector():
    close = deterministic_close()
    ohlcv = make_ohlcv(close)
    result = backtest.run_backtest(close, ohlcv, SECTOR_MAP, 20, 20, 0.6, 0.4, top_n=1)
    metrics = result["metrics"]
    assert set(metrics) >= {
        "total_return",
        "benchmark_total_return",
        "hit_rate",
        "avg_excess_return",
        "max_drawdown",
        "sharpe",
        "n_periods",
    }
    assert metrics["total_return"] > metrics["benchmark_total_return"]
    assert metrics["avg_excess_return"] > 0
    assert 0 <= metrics["hit_rate"] <= 1
    assert result["value"].notna().all()


def test_run_backtest_no_lookahead_first_rows_flat():
    close = deterministic_close()
    ohlcv = make_ohlcv(close)
    result = backtest.run_backtest(close, ohlcv, SECTOR_MAP, 20, 20, top_n=1)
    value = result["value"]
    first = result["period_stats"]["date"].iloc[0]
    pre = value.loc[:first]
    # no position before the first rebalance -> value stays flat at 1.0
    assert pre.min() == pytest.approx(1.0)
    assert pre.max() == pytest.approx(1.0)


def test_signal_reliability_columns_and_ranges():
    close = deterministic_close()
    ohlcv = make_ohlcv(close)
    rel = backtest.signal_reliability(close, ohlcv, SECTOR_MAP, lookbacks=(10, 20))
    assert list(rel["lookback"]) == [10, 20]
    expected_cols = {
        "lookback",
        "n_samples",
        "hit_rate",
        "avg_sector_return",
        "avg_benchmark_return",
        "avg_excess_return",
        "median_excess_return",
        "win_rate_positive",
    }
    assert expected_cols <= set(rel.columns)
    assert rel["n_samples"].gt(0).all()
    assert rel["hit_rate"].between(0, 1).all()


def test_signal_reliability_detects_trending_sector():
    close = deterministic_close()
    ohlcv = make_ohlcv(close)
    rel = backtest.signal_reliability(
        close, ohlcv, SECTOR_MAP, lookbacks=(20,), top_n=1
    )
    row = rel.iloc[0]
    assert row["hit_rate"] == pytest.approx(1.0)
    assert row["avg_excess_return"] > 0


def test_current_ranking_returns_sorted_series():
    close = deterministic_close()
    ohlcv = make_ohlcv(close)
    ranking = backtest.current_ranking(close, ohlcv, SECTOR_MAP, 20, 20)
    assert list(ranking.index) == ["Grow", "Flat", "Fall"]
    assert ranking.is_monotonic_decreasing


def test_top_indices_ignores_nan():
    vals = np.array([np.nan, 3.0, 1.0, np.nan, 2.0])
    assert backtest._top_indices(vals, 2) == [1, 4]
    assert backtest._top_indices(vals, 4) == [1, 4, 2]
    assert backtest._top_indices(np.array([np.nan, np.nan]), 2) == []


def test_mf_percentiles_maps_sectors():
    flows = pd.DataFrame(
        {"sector": ["A", "B", "C"], "net_crore": [10, 20, 30]}
    )
    pct = backtest._mf_percentiles(flows, ["A", "B", "C", "Z"])
    assert pct["A"] < pct["B"] < pct["C"]
    assert np.isnan(pct["Z"])


def test_compute_metrics_empty_periods():
    assert backtest.compute_metrics(
        pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame()
    ) == {}


def test_run_backtest_too_short_data_raises():
    close = deterministic_close(25)
    ohlcv = make_ohlcv(close)
    with pytest.raises(ValueError, match="No valid composite scores"):
        backtest.run_backtest(close, ohlcv, SECTOR_MAP, 60, 60)


def high_low_vol_close(n=300):
    """Two sectors with the same underlying path but different daily volatility."""
    index = pd.date_range("2023-01-02", periods=n, freq="B")
    base = 100 * np.linspace(1.0, 1.5, n)
    rng = np.random.default_rng(1)
    low = base + rng.normal(0, 0.5, n)
    high = base + rng.normal(0, 5.0, n)
    return pd.DataFrame({"LO.NS": low, "HI.NS": high}, index=index)


def test_vol_adjusted_momentum_penalizes_volatility():
    close = high_low_vol_close()
    vam = backtest.volatility_adjusted_momentum(close, 60)
    # both paths have comparable 60-day momentum, but HI.NS is ~10x more
    # volatile per day, so risk-adjusted momentum must rank it lower
    raw = close.pct_change(60).iloc[-1]
    assert raw["LO.NS"] > raw["HI.NS"] * 0.5  # raw momentum is comparable
    assert vam["LO.NS"].iloc[-1] > vam["HI.NS"].iloc[-1]


def test_sector_vol_adj_momentum_has_sector_columns():
    close = deterministic_close()
    vam = backtest.sector_vol_adj_momentum(close, SECTOR_MAP, 20)
    assert list(vam.columns) == ["Grow", "Fall", "Flat"]
    assert vam.index.equals(close.index)


def test_compute_adx_trending_series_is_high():
    close = deterministic_close()
    ohlcv = make_ohlcv(close)
    adx = backtest.compute_adx(
        ohlcv["A.NS"]["High"], ohlcv["A.NS"]["Low"], ohlcv["A.NS"]["Close"]
    )
    assert adx.dropna().iloc[-1] > backtest.REGIME_ADX_THRESHOLD


def test_compute_regime_trending_when_above_smas():
    close = deterministic_close()
    ohlcv = make_ohlcv(close)
    regime = backtest.compute_regime(close["A.NS"], ohlcv["A.NS"])
    assert regime.iloc[-1]
    # warm-up window (slow SMA not yet available) is not trending
    assert not regime.iloc[: backtest.REGIME_SMA_SLOW - 1].any()


def test_compute_regime_flat_is_false():
    close = deterministic_close()
    regime = backtest.compute_regime(close["C.NS"])
    assert not regime.any()


def test_compute_regime_without_ohlcv_uses_trend_only():
    close = deterministic_close()
    regime = backtest.compute_regime(close["A.NS"])
    assert regime.iloc[-1]


def test_compute_regime_trend_only_is_less_strict():
    close = deterministic_close()
    ohlcv = make_ohlcv(close)
    trend_only = backtest.compute_regime(close["A.NS"], ohlcv["A.NS"], require_adx=False)
    strict = backtest.compute_regime(close["A.NS"], ohlcv["A.NS"], require_adx=True)
    # trend-only must never be more conservative than the ADX-confirmed variant
    assert (trend_only >= strict).all()
    assert trend_only.sum() >= strict.sum()


def test_run_backtest_regime_all_cash_is_flat():
    close = deterministic_close()
    ohlcv = make_ohlcv(close)
    regime = pd.Series(False, index=close.index)
    result = backtest.run_backtest(close, ohlcv, SECTOR_MAP, 20, 20, top_n=1, regime=regime)
    assert result["value"].iloc[-1] == pytest.approx(1.0)
    assert "in_market" in result["period_stats"].columns
    assert not result["period_stats"]["in_market"].any()


def test_run_backtest_regime_reduces_exposure():
    close = deterministic_close()
    ohlcv = make_ohlcv(close)
    regime = pd.Series(True, index=close.index)
    regime.iloc[len(close) // 2 :] = False
    gated = backtest.run_backtest(close, ohlcv, SECTOR_MAP, 20, 20, top_n=1, regime=regime)
    always = backtest.run_backtest(close, ohlcv, SECTOR_MAP, 20, 20, top_n=1)
    ps = gated["period_stats"]
    assert ps["in_market"].sum() < len(ps)
    assert ps["in_market"].sum() > 0
    cash_rows = ps[~ps["in_market"]]
    assert not cash_rows.empty
    assert (cash_rows["portfolio_return"].abs() < 1e-9).all()
    assert gated["metrics"]["total_return"] < always["metrics"]["total_return"]


def test_composite_scores_vol_adjust_ranks_trending_first():
    close = deterministic_close()
    ohlcv = make_ohlcv(close)
    comp = backtest.composite_scores(close, ohlcv, SECTOR_MAP, 20, 20, use_vol_adjust=True)
    last = comp.iloc[-1].dropna().sort_values(ascending=False)
    assert last.index[0] == "Grow"


def test_signal_reliability_vol_adjust_runs():
    close = deterministic_close()
    ohlcv = make_ohlcv(close)
    rel = backtest.signal_reliability(
        close, ohlcv, SECTOR_MAP, lookbacks=(20,), use_vol_adjust=True
    )
    assert rel["n_samples"].iloc[0] > 0


def test_load_sector_indices_returns_sectors():
    m = backtest.load_sector_indices()
    assert "^NSEBANK" in m
    assert m["^CNXIT"] == "IT"
    assert len(m) >= 10
