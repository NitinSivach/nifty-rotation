"""Backtest harness for the Nifty sector-rotation signals.

Implements the first, highest-impact step of the improvement roadmap: measure
whether the current signals (momentum + Chaikin Money Flow) actually predict
3-6 month sector returns before investing in fancier data sources.

Key outputs
-----------
- ``run_backtest``: 'buy top-N sectors, rebalance monthly, equal weight'
  simulation with hit-rate, average return, max drawdown and Sharpe. Supports
  volatility-adjusted momentum (``use_vol_adjust``) and an optional market
  regime filter (``regime``) that sits in cash when the Nifty 50 is not in a
  long-term uptrend.
- ``compute_regime``: trend (close above the 200-day SMA) plus optional
  ADX-confirmation regime filter used to gate the backtest defensively.
- ``signal_reliability``: for each momentum/CMF lookback, how often the top-N
  sectors beat the equal-weight benchmark over a forward (default 6-month)
  window. This is the table the AI advisor uses to calibrate confidence.
- ``composite_scores``: a single ranked sector score that combines momentum,
  CMF and (optionally) mutual-fund flows into one number per sector.
- ``sector_indices.json``: an expanded universe of Nifty sector indices that
  can be ranked in place of the Nifty 50 stock map (``--universe indices``).

The module is intentionally free of Streamlit imports so it can run headless
as a script::

    python backtest.py --period 2y --lookback 20 --top-n 3
    python backtest.py --universe indices --vol-adjust --regime
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from settings import backtest_config, data_file

_BT = backtest_config()

TRADING_DAYS_YEAR = _BT["trading_days_year"]
HOLD_DAYS_6M = _BT["hold_days_6m"]
DEFAULT_TOP_N = _BT["default_top_n"]
DEFAULT_REBALANCE_DAYS = _BT["default_rebalance_days"]
LOOKBACKS = tuple(_BT["lookbacks"])
YF_PERIODS = tuple(_BT["yf_periods"])
DEFAULT_PERIOD = _BT["default_period"]
DEFAULT_MOMENTUM_LOOKBACK = _BT["default_momentum_lookback"]
DEFAULT_CMF_LOOKBACK = _BT["default_cmf_lookback"]
DEFAULT_MOMENTUM_WEIGHT = _BT["default_momentum_weight"]
DEFAULT_CMF_WEIGHT = _BT["default_cmf_weight"]
DEFAULT_MF_WEIGHT = _BT["default_mf_weight"]
DEFAULT_DOWNLOAD_INTERVAL = _BT["download"]["interval"]
DEFAULT_DOWNLOAD_CHUNK_SIZE = _BT["download"]["chunk_size"]
DEFAULT_DOWNLOAD_RETRIES = _BT["download"]["retries"]
DEFAULT_DOWNLOAD_BACKOFF_SECONDS = _BT["download"]["backoff_seconds"]

# Market-regime filter defaults (computed on the Nifty 50 index).
REGIME_INDEX = _BT["regime_index"]
REGIME_SMA_FAST = _BT["regime_sma_fast"]
REGIME_SMA_SLOW = _BT["regime_sma_slow"]
REGIME_ADX_PERIOD = _BT["regime_adx_period"]
REGIME_ADX_THRESHOLD = _BT["regime_adx_threshold"]

__all__ = [
    "compute_adx",
    "compute_regime",
    "composite_scores",
    "compute_metrics",
    "current_ranking",
    "download_ohlcv_chunked",
    "fetch_backtest_data",
    "load_nifty50_map",
    "load_sector_indices",
    "rolling_cmf",
    "run_backtest",
    "sector_cmf",
    "sector_daily_returns",
    "sector_momentum",
    "sector_vol_adj_momentum",
    "signal_reliability",
    "volatility_adjusted_momentum",
]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_nifty50_map(path: Path | str | None = None) -> dict:
    if path is None:
        path = data_file("nifty50_map")
    with open(path, "r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def load_sector_indices(path: Path | str | None = None) -> dict:
    """Ticker -> sector label for the Nifty sector-index universe."""
    if path is None:
        path = data_file("sector_indices")
    with open(path, "r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def download_ohlcv_chunked(
    tickers,
    period,
    interval=DEFAULT_DOWNLOAD_INTERVAL,
    retries=DEFAULT_DOWNLOAD_RETRIES,
    backoff_seconds=DEFAULT_DOWNLOAD_BACKOFF_SECONDS,
    chunk_size=DEFAULT_DOWNLOAD_CHUNK_SIZE,
) -> dict:
    """Download daily OHLCV for tickers in chunks with retry/backoff.

    Returns a dict keyed by ticker -> DataFrame with
    [Open, High, Low, Close, Volume] columns.
    """
    results = {}
    for start in range(0, len(tickers), chunk_size):
        chunk = list(tickers)[start : start + chunk_size]
        data = None
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                data = yf.download(
                    tickers=chunk,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=False,
                    threads=True,
                    progress=False,
                )
                break
            except Exception as exc:
                last_err = exc
                time.sleep(backoff_seconds * attempt)
        if data is None:
            raise RuntimeError(
                f"Failed to download {chunk} after {retries} attempts: {last_err}"
            )
        cols = ["Open", "High", "Low", "Close", "Volume"]
        if isinstance(data.columns, pd.MultiIndex):
            for ticker in chunk:
                if ticker in data.columns.levels[0]:
                    df = data[ticker].dropna(how="all")
                    if not df.empty:
                        present = [c for c in cols if c in df.columns]
                        results[ticker] = df[present].copy()
        else:
            if chunk and not data.empty:
                present = [c for c in cols if c in data.columns]
                results[chunk[0]] = data[present].copy()
    return results


def fetch_backtest_data(tickers, period=DEFAULT_PERIOD):
    """Return (close, ohlcv) aligned on a common trading calendar."""
    ohlcv = download_ohlcv_chunked(list(tickers), period)
    close = pd.DataFrame(
        {t: df["Close"] for t, df in ohlcv.items() if "Close" in df.columns}
    )
    close = close.loc[:, ~close.columns.duplicated()]
    for ticker, df in ohlcv.items():
        ohlcv[ticker] = df.reindex(close.index)
    return close, ohlcv


# --------------------------------------------------------------------------- #
# Signal construction
# --------------------------------------------------------------------------- #

def rolling_cmf(df: pd.DataFrame, window: int) -> pd.Series:
    """Chaikin Money Flow over a trailing window, one value per row."""
    denom = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / denom
    money_flow = mfm * df["Volume"]
    flow_sum = money_flow.rolling(window, min_periods=window).sum()
    vol_sum = df["Volume"].rolling(window, min_periods=window).sum()
    cmf = flow_sum / vol_sum
    # Collapse sign-bit noise (-0.0) so near-zero CMF ranks as a true tie.
    return cmf.mask(cmf.abs() < 1e-9, 0.0)


def _sector_groups(nifty50_map: dict, tickers) -> dict:
    groups = {}
    for ticker in tickers:
        groups.setdefault(nifty50_map.get(ticker, "Other"), []).append(ticker)
    return groups


def sector_daily_returns(close: pd.DataFrame, nifty50_map: dict) -> pd.DataFrame:
    """Daily equal-weight sector index returns (mean of member tickers)."""
    ticker_ret = close.pct_change()
    groups = _sector_groups(nifty50_map, close.columns)
    return pd.DataFrame(
        {sector: ticker_ret[members].mean(axis=1) for sector, members in groups.items()}
    )


def sector_momentum(close: pd.DataFrame, nifty50_map: dict, lookback: int) -> pd.DataFrame:
    """Sector average momentum (fractional return over ``lookback`` days)."""
    mom = close.pct_change(lookback)
    groups = _sector_groups(nifty50_map, close.columns)
    return pd.DataFrame(
        {sector: mom[members].mean(axis=1) for sector, members in groups.items()}
    )


def sector_cmf(ohlcv: dict, nifty50_map: dict, window: int) -> pd.DataFrame:
    """Sector average Chaikin Money Flow over a trailing window."""
    cmf = {ticker: rolling_cmf(df, window) for ticker, df in ohlcv.items()}
    frame = pd.DataFrame(cmf)
    groups = _sector_groups(nifty50_map, frame.columns)
    return pd.DataFrame(
        {
            sector: frame[members].mean(axis=1, skipna=True)
            for sector, members in groups.items()
            if members
        }
    )


def volatility_adjusted_momentum(close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Per-ticker momentum divided by rolling daily-return volatility.

    Risk-normalizes raw momentum so a high-beta mover is rewarded only if it
    actually trends (the 'quality of the trend' variant of momentum).
    """
    mom = close.pct_change(lookback)
    rets = close.pct_change()
    vol = rets.rolling(lookback, min_periods=lookback).std()
    return mom / vol.replace(0, np.nan)


def sector_vol_adj_momentum(
    close: pd.DataFrame, nifty50_map: dict, lookback: int
) -> pd.DataFrame:
    """Sector average of volatility-adjusted momentum."""
    var_mom = volatility_adjusted_momentum(close, lookback)
    groups = _sector_groups(nifty50_map, close.columns)
    return pd.DataFrame(
        {sector: var_mom[members].mean(axis=1) for sector, members in groups.items()}
    )


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = REGIME_ADX_PERIOD,
) -> pd.Series:
    """Wilder's Average Directional Index from an index OHLC series."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def compute_regime(
    index_close: pd.Series,
    index_ohlcv: pd.DataFrame | None = None,
    sma_fast: int = REGIME_SMA_FAST,
    sma_slow: int = REGIME_SMA_SLOW,
    adx_period: int = REGIME_ADX_PERIOD,
    adx_threshold: float = REGIME_ADX_THRESHOLD,
    require_adx: bool = True,
) -> pd.Series:
    """Market regime filter: True = trending (risk-on), False = choppy (cash).

    The primary gate is price above the long-term (``sma_slow``) moving average.
    When ``require_adx`` is True (the conservative variant) the regime also
    requires ADX to be above ``adx_threshold`` so that slow grinds with no
    directional conviction are excluded too. Backtests on Nifty show the
    trend-only gate (``require_adx=False``) preserves more upside while still
    cutting drawdown, so the dashboard defaults it off.

    The result is aligned to ``index_close`` and is False during the SMA warm-up.
    """
    slow = index_close.rolling(sma_slow, min_periods=sma_slow).mean()
    trending = index_close > slow
    if require_adx:
        fast = index_close.rolling(sma_fast, min_periods=sma_fast).mean()
        trending = trending & (index_close > fast)
        if index_ohlcv is not None and {"High", "Low", "Close"}.issubset(index_ohlcv.columns):
            adx = compute_adx(
                index_ohlcv["High"], index_ohlcv["Low"], index_ohlcv["Close"], adx_period
            )
            trending = trending & (adx >= adx_threshold)
    return trending.fillna(False).astype(bool)


def _mf_percentiles(mf_flows: pd.DataFrame, sectors) -> dict:
    """Map each sector to its percentile rank (0..1) of net flow."""
    pct = pd.Series(
        mf_flows["net_crore"].rank(pct=True).values, index=mf_flows["sector"]
    )
    return {sector: float(pct.get(sector, np.nan)) for sector in sectors}


def composite_scores(
    close: pd.DataFrame,
    ohlcv: dict,
    nifty50_map: dict,
    momentum_lookback: int = DEFAULT_MOMENTUM_LOOKBACK,
    cmf_lookback: int = DEFAULT_CMF_LOOKBACK,
    momentum_weight: float = DEFAULT_MOMENTUM_WEIGHT,
    cmf_weight: float = DEFAULT_CMF_WEIGHT,
    mf_flows: pd.DataFrame | None = None,
    mf_weight: float = DEFAULT_MF_WEIGHT,
    use_vol_adjust: bool = False,
) -> pd.DataFrame:
    """Ranked sector composite score (higher = stronger) over time.

    Combines cross-sectional percentile ranks of momentum (raw or
    volatility-adjusted when ``use_vol_adjust`` is True) and CMF, plus an
    optional mutual-fund flow overlay (constant over time, used only for the
    current ranking since MF flows are not available historically).
    """
    smom = (
        sector_vol_adj_momentum(close, nifty50_map, momentum_lookback)
        if use_vol_adjust
        else sector_momentum(close, nifty50_map, momentum_lookback)
    )
    scmf = sector_cmf(ohlcv, nifty50_map, cmf_lookback)
    index = smom.index.intersection(scmf.index)
    smom = smom.reindex(index)
    scmf = scmf.reindex(index)
    mom_pct = smom.rank(axis=1, pct=True)
    cmf_pct = scmf.rank(axis=1, pct=True)
    comp = momentum_weight * mom_pct + cmf_weight * cmf_pct
    if mf_flows is not None and mf_weight > 0:
        overlay = _mf_percentiles(mf_flows, comp.columns)
        comp = comp.copy()
        for sector, value in overlay.items():
            if sector in comp.columns and not np.isnan(value):
                comp[sector] = comp[sector] + mf_weight * value
    return comp


def current_ranking(
    close: pd.DataFrame,
    ohlcv: dict,
    nifty50_map: dict,
    momentum_lookback: int = DEFAULT_MOMENTUM_LOOKBACK,
    cmf_lookback: int = DEFAULT_CMF_LOOKBACK,
    momentum_weight: float = DEFAULT_MOMENTUM_WEIGHT,
    cmf_weight: float = DEFAULT_CMF_WEIGHT,
    mf_flows: pd.DataFrame | None = None,
    mf_weight: float = DEFAULT_MF_WEIGHT,
    use_vol_adjust: bool = False,
) -> pd.Series:
    """Latest composite score per sector, sorted best first."""
    comp = composite_scores(
        close,
        ohlcv,
        nifty50_map,
        momentum_lookback,
        cmf_lookback,
        momentum_weight,
        cmf_weight,
        mf_flows,
        mf_weight,
        use_vol_adjust,
    )
    return comp.iloc[-1].dropna().sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #

def _first_valid_index(frame: pd.DataFrame):
    for i in range(len(frame)):
        if frame.iloc[i].notna().any():
            return i
    return None


def _top_indices(vals: np.ndarray, top_n: int):
    """Indices (into ``vals``) of the top_n largest non-NaN values."""
    valid = np.where(~np.isnan(vals))[0]
    if len(valid) == 0:
        return []
    k = min(top_n, len(valid))
    order = np.argsort(vals[valid])[::-1][:k]
    return valid[order].tolist()


def compute_metrics(
    value: pd.Series, benchmark_value: pd.Series, period_stats: pd.DataFrame
) -> dict:
    """Return performance metrics over the traded window."""
    if value.empty or period_stats.empty:
        return {}
    first = period_stats["date"].iloc[0]
    value = value.loc[first:]
    benchmark_value = benchmark_value.loc[first:]
    if len(value) < 2:
        return {}

    def _perf(series: pd.Series):
        total = float(series.iloc[-1] / series.iloc[0] - 1)
        years = len(series) / TRADING_DAYS_YEAR
        cagr = (
            float(series.iloc[-1] ** (1 / years) - 1)
            if years > 0 and series.iloc[-1] > 0
            else float("nan")
        )
        rets = series.pct_change().dropna()
        vol = float(rets.std(ddof=1))
        sharpe = (
            float(rets.mean() / vol * np.sqrt(TRADING_DAYS_YEAR)) if vol > 0 else float("nan")
        )
        drawdown = float((series / series.cummax() - 1).min())
        return total, cagr, sharpe, drawdown

    total, cagr, sharpe, drawdown = _perf(value)
    bench_total, bench_cagr, bench_sharpe, bench_drawdown = _perf(benchmark_value)
    ps = period_stats
    hit_rate = (
        float((ps["portfolio_return"] > ps["benchmark_return"]).mean()) if len(ps) else float("nan")
    )
    return {
        "total_return": total,
        "benchmark_total_return": bench_total,
        "cagr": cagr,
        "benchmark_cagr": bench_cagr,
        "sharpe": sharpe,
        "benchmark_sharpe": bench_sharpe,
        "max_drawdown": drawdown,
        "benchmark_max_drawdown": bench_drawdown,
        "hit_rate": hit_rate,
        "avg_excess_return": float(ps["excess_return"].mean()),
        "n_periods": int(len(ps)),
    }


def run_backtest(
    close: pd.DataFrame,
    ohlcv: dict,
    nifty50_map: dict,
    momentum_lookback: int = DEFAULT_MOMENTUM_LOOKBACK,
    cmf_lookback: int = DEFAULT_CMF_LOOKBACK,
    momentum_weight: float = DEFAULT_MOMENTUM_WEIGHT,
    cmf_weight: float = DEFAULT_CMF_WEIGHT,
    top_n: int = DEFAULT_TOP_N,
    rebalance_days: int = DEFAULT_REBALANCE_DAYS,
    use_vol_adjust: bool = False,
    regime: pd.Series | None = None,
) -> dict:
    """Simulate 'buy top-N sectors, rebalance periodically, equal weight'.

    ``use_vol_adjust`` swaps raw momentum for volatility-adjusted momentum.
    ``regime`` is an optional bool series (from ``compute_regime``) aligned to
    ``close.index``; on rebalance days where it is False the portfolio sits in
    cash instead of holding the top-N sectors.

    Returns a dict with keys: value, benchmark_value, portfolio_returns,
    period_stats, metrics, weights. Trades settle on the day after the signal
    (weights are shifted by one row) to avoid lookahead bias.
    """
    comp = composite_scores(
        close, ohlcv, nifty50_map, momentum_lookback, cmf_lookback,
        momentum_weight, cmf_weight, None, 0.0, use_vol_adjust,
    )
    sector_ret = sector_daily_returns(close, nifty50_map).reindex(comp.index)
    benchmark = close.pct_change().reindex(comp.index).mean(axis=1)
    sectors = list(sector_ret.columns)
    dates = comp.index
    n = len(dates)

    if regime is not None:
        regime = regime.reindex(comp.index).fillna(False)

    start = _first_valid_index(comp)
    if start is None:
        raise ValueError(
            "No valid composite scores produced — check data length vs lookbacks."
        )

    weights = np.zeros((n, len(sectors)))
    for i in range(start, n, rebalance_days):
        top_idx = _top_indices(comp.iloc[i].to_numpy(dtype=float), top_n)
        if not top_idx:
            continue
        if regime is not None and not bool(regime.iloc[i]):
            continue  # regime filter: sit in cash this period
        for k in top_idx:
            weights[i, k] = 1.0 / len(top_idx)

    holdings = pd.DataFrame(weights, index=dates, columns=sectors).shift(1).fillna(0.0)
    portfolio_ret = (holdings * sector_ret).sum(axis=1)
    value = (1 + portfolio_ret).cumprod()
    bench_value = (1 + benchmark).cumprod()

    period_rows = []
    for i in range(start, n, rebalance_days):
        top_idx = _top_indices(comp.iloc[i].to_numpy(dtype=float), top_n)
        if not top_idx:
            continue
        in_market = regime is None or bool(regime.iloc[i])
        label = ", ".join(sectors[k] for k in top_idx) if in_market else "CASH"
        j = min(i + rebalance_days, n)
        port_period = float(np.prod(1 + portfolio_ret.iloc[i:j].to_numpy()) - 1)
        bench_period = float(np.prod(1 + benchmark.iloc[i:j].to_numpy()) - 1)
        period_rows.append(
            {
                "date": dates[i],
                "top_sectors": label,
                "portfolio_return": port_period,
                "benchmark_return": bench_period,
                "excess_return": port_period - bench_period,
                "in_market": bool(in_market),
            }
        )
    period_stats = pd.DataFrame(period_rows)
    metrics = compute_metrics(value, bench_value, period_stats)
    return {
        "value": value,
        "benchmark_value": bench_value,
        "portfolio_returns": portfolio_ret,
        "period_stats": period_stats,
        "metrics": metrics,
        "weights": holdings,
    }


# --------------------------------------------------------------------------- #
# Signal reliability (what the advisor is grounded in)
# --------------------------------------------------------------------------- #

def signal_reliability(
    close: pd.DataFrame,
    ohlcv: dict,
    nifty50_map: dict,
    lookbacks=LOOKBACKS,
    hold_days: int = HOLD_DAYS_6M,
    top_n: int = DEFAULT_TOP_N,
    step_days: int = DEFAULT_REBALANCE_DAYS,
    momentum_weight: float = DEFAULT_MOMENTUM_WEIGHT,
    cmf_weight: float = DEFAULT_CMF_WEIGHT,
    use_vol_adjust: bool = False,
) -> pd.DataFrame:
    """For each lookback, how well do the top-N sectors predict forward returns.

    At every ``step_days``-th trading day, the top-N sectors by composite score
    are held for ``hold_days`` and compared with the equal-weight benchmark.
    """
    groups = _sector_groups(nifty50_map, close.columns)
    fwd = close.shift(-hold_days) / close - 1
    sector_fwd = pd.DataFrame(
        {sector: fwd[members].mean(axis=1) for sector, members in groups.items()}
    )
    bench_fwd = fwd.mean(axis=1)

    rows = []
    for lookback in lookbacks:
        comp = composite_scores(
            close, ohlcv, nifty50_map, lookback, lookback,
            momentum_weight, cmf_weight, None, 0.0, use_vol_adjust,
        )
        samples = []
        start = _first_valid_index(comp)
        if start is not None:
            for i in range(start, len(comp), step_days):
                top_idx = _top_indices(comp.iloc[i].to_numpy(dtype=float), top_n)
                if not top_idx:
                    continue
                port_vals = sector_fwd.iloc[i].to_numpy(dtype=float)[top_idx]
                if np.isnan(port_vals).all():
                    continue
                port = float(np.nanmean(port_vals))
                bench = bench_fwd.iloc[i]
                if np.isnan(port) or np.isnan(bench):
                    continue
                samples.append((float(port), float(bench)))
        arr = np.array(samples)
        if len(arr) == 0:
            rows.append(
                {
                    "lookback": lookback,
                    "n_samples": 0,
                    "hit_rate": float("nan"),
                    "avg_sector_return": float("nan"),
                    "avg_benchmark_return": float("nan"),
                    "avg_excess_return": float("nan"),
                    "median_excess_return": float("nan"),
                    "win_rate_positive": float("nan"),
                }
            )
            continue
        excess = arr[:, 0] - arr[:, 1]
        rows.append(
            {
                "lookback": lookback,
                "n_samples": int(len(arr)),
                "hit_rate": float((arr[:, 0] > arr[:, 1]).mean()),
                "avg_sector_return": float(arr[:, 0].mean()),
                "avg_benchmark_return": float(arr[:, 1].mean()),
                "avg_excess_return": float(excess.mean()),
                "median_excess_return": float(np.median(excess)),
                "win_rate_positive": float((arr[:, 0] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Backtest Nifty sector rotation signals."
    )
    parser.add_argument("--period", choices=YF_PERIODS, default=DEFAULT_PERIOD)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--rebalance-days", type=int, default=DEFAULT_REBALANCE_DAYS)
    parser.add_argument("--lookback", type=int, default=DEFAULT_MOMENTUM_LOOKBACK)
    parser.add_argument(
        "--universe",
        choices=("stocks", "indices"),
        default="stocks",
        help="Rank Nifty 50 stocks by sector, or Nifty sector indices directly.",
    )
    parser.add_argument(
        "--vol-adjust",
        action="store_true",
        help="Use volatility-adjusted momentum instead of raw momentum.",
    )
    parser.add_argument(
        "--regime",
        action="store_true",
        help="Apply the Nifty 50 trend regime filter (cash when below the 200-day MA).",
    )
    parser.add_argument(
        "--regime-adx",
        action="store_true",
        help="With --regime, additionally require ADX confirmation (stricter).",
    )
    args = parser.parse_args(argv)

    if args.universe == "indices":
        nifty50_map = load_sector_indices()
    else:
        nifty50_map = load_nifty50_map()
    tickers = list(nifty50_map)
    print(f"Downloading {len(tickers)} tickers for {args.period} ...")
    close, ohlcv = fetch_backtest_data(tickers, args.period)
    print(
        f"Data range: {close.index[0].date()} -> {close.index[-1].date()} "
        f"({len(close)} rows)"
    )

    regime = None
    if args.regime:
        print("Fetching Nifty 50 index for the regime filter ...")
        idx_close, idx_ohlcv = fetch_backtest_data([REGIME_INDEX], args.period)
        if REGIME_INDEX in idx_close.columns:
            regime = compute_regime(
                idx_close[REGIME_INDEX],
                idx_ohlcv.get(REGIME_INDEX),
                require_adx=args.regime_adx,
            )
            latest = bool(regime.iloc[-1])
            print(
                f"Regime filter: latest market state = "
                f"{'TRENDING' if latest else 'CASH/FLAT'} "
                f"(adx_gate={args.regime_adx})"
            )
        else:
            print("Warning: could not fetch Nifty 50 index; skipping regime filter.")

    reliability = signal_reliability(
        close, ohlcv, nifty50_map, top_n=args.top_n, use_vol_adjust=args.vol_adjust
    )
    print(f"\n=== Signal reliability (top-{args.top_n} vs equal-weight benchmark over ~6 months) ===")
    print(reliability.to_string(index=False))

    result = run_backtest(
        close,
        ohlcv,
        nifty50_map,
        momentum_lookback=args.lookback,
        cmf_lookback=args.lookback,
        top_n=args.top_n,
        rebalance_days=args.rebalance_days,
        use_vol_adjust=args.vol_adjust,
        regime=regime,
    )
    metrics = result["metrics"]
    print(
        f"\n=== Backtest ({args.lookback}d lookback, top-{args.top_n}, "
        f"rebalance every {args.rebalance_days}d, "
        f"universe={args.universe}, vol_adjust={args.vol_adjust}, "
        f"regime={args.regime}) ==="
    )
    for key, value in metrics.items():
        if isinstance(value, float):
            if "return" in key or "drawdown" in key:
                print(f"  {key:26s}: {value * 100:6.2f}%")
            elif key in ("hit_rate",):
                print(f"  {key:26s}: {value * 100:6.2f}%")
            else:
                print(f"  {key:26s}: {value:.4f}")
        else:
            print(f"  {key:26s}: {value}")


if __name__ == "__main__":
    main()
