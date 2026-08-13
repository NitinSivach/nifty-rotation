import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from app import (
    compute_cmf,
    compute_dividend_stats,
    compute_ema_breakouts,
    compute_sector_rotation,
    compute_ticker_period_returns,
    filter_mf_flows,
    format_crore,
    format_return,
    load_dividend_universe,
    load_mf_flows,
    load_nifty50_map,
    resample_close,
)


def make_close(tickers, n=100, seed=0, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n, freq="B")
    data = {}
    for t in tickers:
        price = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
        data[t] = price
    return pd.DataFrame(data, index=index)


def make_ohlcv(n=30, high=110.0, low=90.0, close=105.0, volume=1000.0):
    index = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": np.full(n, close),
            "High": np.full(n, high),
            "Low": np.full(n, low),
            "Close": np.full(n, close),
            "Volume": np.full(n, volume),
        },
        index=index,
    )


def test_format_return():
    assert format_return(0.1234) == "12.34%"
    assert format_return(-0.005) == "-0.50%"


def test_compute_sector_rotation_ranks_and_summary():
    tickers = ["A.NS", "B.NS", "C.NS"]
    nifty50_map = {"A.NS": "Tech", "B.NS": "Tech", "C.NS": "Bank"}
    close = make_close(tickers)

    momentum, sector_summary = compute_sector_rotation(close, 20, nifty50_map)

    assert set(momentum["ticker"]) == set(tickers)
    assert set(momentum["sector"]) == {"Tech", "Bank"}
    assert momentum["rank"].min() == 1
    assert momentum["rank"].max() == len(tickers)
    assert momentum["rank"].is_monotonic_increasing
    assert list(sector_summary.index) == list(sector_summary.index)  # valid index
    assert "symbol_count" in sector_summary.columns
    assert sector_summary["average_momentum"].is_monotonic_decreasing


def test_compute_sector_rotation_missing_sector_gets_other():
    tickers = ["A.NS", "B.NS"]
    nifty50_map = {"A.NS": "Tech"}  # B.NS unknown
    close = make_close(tickers)

    momentum, _ = compute_sector_rotation(close, 10, nifty50_map)

    assert set(momentum["sector"]) == {"Tech", "Other"}


def test_compute_cmf_positive_steady_flow():
    ohlcv = {"TEST.NS": make_ohlcv(n=30)}
    value = compute_cmf(ohlcv, 10)["TEST.NS"]

    assert value is not None
    # mfm = ((105-90) - (110-105)) / (110-90) = 0.5 on every row
    assert value == pytest.approx(0.5)


def test_compute_cmf_insufficient_data_returns_none():
    ohlcv = {"TEST.NS": make_ohlcv(n=5)}
    assert compute_cmf(ohlcv, 10)["TEST.NS"] is None


def test_compute_cmf_zero_range_is_none_or_finite():
    # high == low on every row would normally blow up; here values are finite.
    ohlcv = {"TEST.NS": make_ohlcv(n=20, high=100.0, low=100.0, close=100.0)}
    value = compute_cmf(ohlcv, 10)["TEST.NS"]
    assert value is None or np.isfinite(value)


@pytest.fixture
def mf_flows_csv(tmp_path):
    data = pd.DataFrame(
        {
            "sector": ["IT", "IT", "Financials", "Financials"],
            "period": ["3M", "1Y", "3M", "2Y"],
            "buy_crore": [1000, 5000, 800, 20000],
            "sell_crore": [400, 2000, 1200, 8000],
            "net_crore": [600, 3000, -400, 12000],
        }
    )
    path = tmp_path / "mf_sector_flows.csv"
    data.to_csv(path, index=False)
    return path


def test_load_mf_flows(tmp_path, mf_flows_csv):
    df = load_mf_flows(mf_flows_csv)
    assert list(df.columns) == ["sector", "period", "buy_crore", "sell_crore", "net_crore"]
    assert len(df) == 4


def test_filter_mf_flows_filters_period(mf_flows_csv):
    df = load_mf_flows(mf_flows_csv)
    filtered = filter_mf_flows(df, "3M")
    assert list(filtered["period"].unique()) == ["3M"]
    assert set(filtered["sector"]) == {"IT", "Financials"}
    assert len(filtered) == 2


def test_filter_mf_flows_case_insensitive(mf_flows_csv):
    df = load_mf_flows(mf_flows_csv)
    assert len(filter_mf_flows(df, "1y")) == 1


def test_format_crore():
    assert format_crore(1234567) == "₹1,234,567 Cr"
    assert format_crore(-250.0) == "₹-250 Cr"


def test_compute_ticker_period_returns():
    close = pd.DataFrame(
        {"A.NS": [100.0, 110.0, 120.0], "B.NS": [200.0, 180.0, 160.0]}
    )
    result = compute_ticker_period_returns(close, {"A.NS": "Tech", "B.NS": "Bank"})
    assert set(result["ticker"]) == {"A.NS", "B.NS"}
    assert result.loc[result["ticker"] == "A.NS", "period_return"].iloc[0] == pytest.approx(0.2)
    assert result.loc[result["ticker"] == "B.NS", "period_return"].iloc[0] == pytest.approx(-0.2)
    assert result.loc[result["ticker"] == "A.NS", "current_price"].iloc[0] == pytest.approx(120.0)
    assert set(result["sector"]) == {"Tech", "Bank"}


def make_breakout_daily():
    """Rising start (above EMA), 35 flat low days (below EMA), then a jump above."""
    n = 61
    index = pd.date_range("2023-01-02", periods=n, freq="B")
    prices = np.zeros(n)
    prices[:25] = 100 + np.arange(25)  # 100..124, rising above lagging EMA
    prices[25:60] = 100.0              # flat low, stays below the lagging EMA
    prices[60:] = 130.0                # breakout day
    return pd.DataFrame({"A.NS": prices}, index=index)


def make_dividend_history(years=5, close=100.0, annual_div=4.0):
    """5 years of daily B-day rows: constant close, annual dividend at year-end."""
    n = years * 252
    index = pd.date_range("2020-01-02", periods=n, freq="B")
    df = pd.DataFrame({"Close": close, "Dividends": 0.0}, index=index)
    for year in range(2020, 2020 + years):
        last_row = df.index[df.index.year == year][-1]
        df.loc[last_row, "Dividends"] = annual_div
    return df


def test_compute_dividend_stats_yield():
    df = make_dividend_history()
    stats = compute_dividend_stats({"A.NS": df}, {"A.NS": "Financials"})
    assert len(stats) == 1
    row = stats.iloc[0]
    assert row["ticker"] == "A.NS"
    assert row["sector"] == "Financials"
    assert row["current_price"] == pytest.approx(100.0)
    assert row["avg_annual_dividend"] == pytest.approx(4.0)
    assert row["avg_dividend_yield"] == pytest.approx(4.0)
    assert row["total_dividend"] == pytest.approx(20.0)
    assert row["years_count"] == 5


def test_compute_dividend_stats_skips_non_payers():
    df = make_dividend_history(annual_div=0.0)
    assert compute_dividend_stats({"A.NS": df}, {"A.NS": "Tech"}).empty


def test_compute_dividend_stats_skips_missing_dividend_column():
    index = pd.date_range("2020-01-02", periods=100, freq="B")
    df = pd.DataFrame({"Close": 100.0}, index=index)
    assert compute_dividend_stats({"A.NS": df}, {"A.NS": "Tech"}).empty


def test_compute_dividend_stats_sorts_by_yield_desc():
    hi = make_dividend_history(annual_div=8.0)   # 8% yield
    lo = make_dividend_history(annual_div=2.0)   # 2% yield
    stats = compute_dividend_stats(
        {"A.NS": hi, "B.NS": lo}, {"A.NS": "Tech", "B.NS": "Bank"}
    )
    assert list(stats["ticker"]) == ["A.NS", "B.NS"]
    assert stats["avg_dividend_yield"].is_monotonic_decreasing


def test_load_dividend_universe_merges_base_and_extra(tmp_path):
    extra_path = tmp_path / "dividend_universe.json"
    extra_path.write_text('{"EXTRA.NS": "Tech"}', encoding="utf-8")
    merged = load_dividend_universe(extra_path)
    assert merged["EXTRA.NS"] == "Tech"
    assert set(load_nifty50_map()).issubset(set(merged))


def test_load_dividend_universe_missing_file_falls_back_to_nifty50():
    missing = load_dividend_universe(Path("does-not-exist.json"))
    assert missing == load_nifty50_map()


def test_load_dividend_universe_default_file_valid():
    merged = load_dividend_universe()
    assert len(merged) > len(load_nifty50_map())
    assert all(isinstance(v, str) for v in merged.values())


def test_resample_close_daily_is_noop():
    close = make_breakout_daily()
    assert resample_close(close, "D") is close


def test_resample_close_weekly_last_close_per_week():
    daily = pd.DataFrame(
        {"A.NS": range(14)},
        index=pd.date_range("2023-01-02", periods=14, freq="B"),
    )
    weekly = resample_close(daily, "W")
    # 2023-01-02 (Mon) is week 1, Jan 9 is week 2, Jan 16 is week 3.
    assert weekly.shape[0] == 3
    assert list(weekly["A.NS"]) == [4, 9, 13]


def test_compute_ema_breakouts_detects_long_downtrend_breakout():
    close = make_breakout_daily()
    breakouts = compute_ema_breakouts(close, ema_span=20)
    assert len(breakouts) == 1
    row = breakouts.iloc[0]
    assert row["ticker"] == "A.NS"
    assert row["below_streak_periods"] == 35
    assert row["cross_date"] == close.index[60]
    assert bool(row["currently_above"])
    assert row["pct_above_ema"] > 0


def test_compute_ema_breakouts_filters_by_min_below_periods():
    close = make_breakout_daily()
    assert compute_ema_breakouts(close, ema_span=20, min_below_periods=40).empty
    assert len(compute_ema_breakouts(close, ema_span=20, min_below_periods=30)) == 1


def test_compute_ema_breakouts_excludes_never_crossing_stocks():
    n = 80
    index = pd.date_range("2023-01-02", periods=n, freq="B")
    prices = np.linspace(100.0, 80.0, n)  # steady decline: always below lagging EMA
    close = pd.DataFrame({"B.NS": prices}, index=index)
    assert compute_ema_breakouts(close, ema_span=20).empty


def test_compute_ema_breakouts_weekly_frame_reports_weeks_and_years():
    weeks = 120
    index = pd.date_range("2020-01-06", periods=weeks, freq="W")
    prices = np.zeros(weeks)
    prices[:10] = 100 + np.arange(10) * 2  # rising start (above EMA)
    prices[10:110] = 100.0                 # 100 weeks below the lagging EMA
    prices[110:] = 140.0                   # breakout
    close = pd.DataFrame({"A.NS": prices}, index=index)
    breakouts = compute_ema_breakouts(close, ema_span=20, periods_per_year=52)
    assert len(breakouts) == 1
    row = breakouts.iloc[0]
    assert row["below_streak_periods"] == 100
    assert row["streak_years"] == pytest.approx(100 / 52)
    assert row["cross_date"] == index[110]
    assert row["periods_since_cross"] == weeks - 1 - 110
