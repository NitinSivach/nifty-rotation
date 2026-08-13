import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

import ai_advisor
import backtest
from settings import app_config, data_file

_APP = app_config()


def load_nifty50_map(path: Path | str = None) -> dict:
    if path is None:
        path = data_file("nifty50_map")
    with Path(path).open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def load_dividend_universe(path: Path | str = None) -> dict:
    """Expanded ticker -> sector universe: Nifty 50 + curated large caps.

    Merges ``nifty50_map.json`` with the extra tickers in
    ``dividend_universe.json`` (Nifty Next 50 members, PSUs and other
    blue-chip dividend payers). Falls back to the Nifty 50 map alone if the
    extra file is missing or invalid.
    """
    mapping = load_nifty50_map()
    if path is None:
        path = data_file("dividend_universe")
    try:
        with Path(path).open("r", encoding="utf-8-sig") as file:
            extra = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        extra = {}
    mapping.update(extra)
    return mapping


def _download_chunked(
    tickers,
    period,
    interval="1d",
    auto_adjust=True,
    retries=_APP["download_retries"],
    backoff_seconds=_APP["download_backoff_seconds"],
    actions=False,
) -> dict:
    """Download daily data for tickers in chunks with retry/backoff.

    Returns a dict keyed by ticker -> DataFrame of OHLCV columns. When
    ``actions`` is True the frames additionally carry Dividends / Stock
    Splits columns.
    """
    chunk_size = _APP["download_chunk_size"]
    results = {}
    for start in range(0, len(tickers), chunk_size):
        chunk = tickers[start : start + chunk_size]
        data = None
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                data = yf.download(
                    tickers=chunk,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=auto_adjust,
                    threads=True,
                    progress=False,
                    actions=actions,
                )
                break
            except Exception as exc:
                last_err = exc
                time.sleep(backoff_seconds * attempt)
        if data is None:
            raise RuntimeError(
                f"Failed to download {chunk} after {retries} attempts: {last_err}"
            )
        if isinstance(data.columns, pd.MultiIndex):
            for ticker in chunk:
                if ticker in data.columns.levels[0]:
                    df = data[ticker].dropna(how="all")
                    if not df.empty:
                        results[ticker] = df
        else:
            if chunk and not data.empty:
                results[chunk[0]] = data.dropna(how="all")
    return results


_TTL = _APP["cache_ttl"]


@st.cache_data(ttl=_TTL, show_spinner=False)
def fetch_close_prices(tickers, period):
    ohlcv = _download_chunked(tickers, period, interval="1d", auto_adjust=True)
    frames = {
        ticker: df["Close"]
        for ticker, df in ohlcv.items()
        if "Close" in df.columns
    }
    close = pd.DataFrame(frames)
    close = close.loc[:, ~close.columns.duplicated()]

    present = list(close.columns)
    missing = [t for t in tickers if t not in present]
    return close, missing


@st.cache_data(ttl=_TTL, show_spinner=False)
def fetch_ohlcv(tickers, period):
    """Fetch OHLCV for tickers using yfinance.download in chunks.

    Returns a dict of DataFrames keyed by ticker with columns [Open, High, Low, Close, Volume].
    """
    ohlcv = _download_chunked(tickers, period, interval="1d", auto_adjust=False)
    result = {}
    for ticker, df in ohlcv.items():
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        result[ticker] = df[cols].copy()
    return result


@st.cache_data(ttl=_TTL, show_spinner=False)
def compute_cmf(ohlcv_dict, lookback_days):
    """Compute Chaikin Money Flow per ticker over lookback_days.

    Returns a dict ticker->cmf_value (float or None).
    """
    cmf = {}
    for ticker, df in ohlcv_dict.items():
        try:
            if len(df) < lookback_days:
                cmf[ticker] = None
                continue
            recent = df.tail(lookback_days)
            high = recent["High"]
            low = recent["Low"]
            close = recent["Close"]
            vol = recent["Volume"]
            denom = high - low
            # avoid division by zero
            mfm = ((close - low) - (high - close)) / denom.replace(0, np.nan)
            money_flow = mfm * vol
            total_money_flow = money_flow.sum(skipna=True)
            total_volume = vol.sum(skipna=True)
            if pd.isna(total_money_flow) or total_volume == 0:
                cmf[ticker] = None
            else:
                cmf[ticker] = float(total_money_flow / total_volume)
        except Exception:
            cmf[ticker] = None
    return cmf


@st.cache_data(ttl=_TTL, show_spinner=False)
def compute_sector_rotation(close, lookback_days, nifty50_map):
    momentum_returns = close.pct_change(lookback_days).iloc[-1]
    period_returns = close.iloc[-1] / close.iloc[0] - 1

    momentum = pd.DataFrame(
        {
            "ticker": momentum_returns.index,
            "sector": [nifty50_map.get(ticker, "Other") for ticker in momentum_returns.index],
            "momentum": momentum_returns.values,
            "period_return": period_returns.values,
        }
    ).dropna()

    momentum["rank"] = momentum["momentum"].rank(ascending=False)
    sector_summary = (
        momentum.groupby("sector")
        .agg(
            average_momentum=("momentum", "mean"),
            median_momentum=("momentum", "median"),
            average_period_return=("period_return", "mean"),
            median_period_return=("period_return", "median"),
            symbol_count=("ticker", "count"),
        )
        .sort_values("average_momentum", ascending=False)
    )

    return momentum, sector_summary


def resample_close(close: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample daily closes to the last close of each period.

    ``freq`` uses pandas conventions: "D"/"B" (no-op), "W" (weekly) or "ME" (monthly).
    """
    if freq in (None, "", "D", "B", "1D", "1B"):
        return close
    return close.resample(freq).last().dropna(how="all")


def compute_ema_breakouts(
    close: pd.DataFrame,
    ema_span: int = _APP["default_ema_span"],
    min_below_periods: int = 0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Find stocks whose price most recently crossed above their EMA after a long below-streak.

    ``close`` may be daily or already resampled to weekly/monthly bars; all streak
    durations are expressed in that frame's periods. A long ``below_streak_periods``
    followed by a cross flags a potential multi-year downtrend breakout.

    Returns a DataFrame with columns: ticker, last_close, ema, cross_date,
    below_streak_periods, streak_years, periods_since_cross, currently_above,
    pct_above_ema (sorted by longest streak).
    """
    ema = close.ewm(span=ema_span, adjust=False).mean()
    above = close > ema
    # Note: fillna(False) on a shifted boolean frame leaves an object dtype, and
    # `~` on Python bools does ~True == -2. Force bool so NOT behaves correctly.
    was_above = above.shift(1).fillna(False).astype(bool)
    up_cross = above & ~was_above

    results = []
    for ticker in close.columns:
        abv = above[ticker]
        cr = up_cross[ticker]
        cross_dates = cr[cr].index
        if len(cross_dates) == 0:
            continue
        last_cross = cross_dates[-1]
        pos = close.index.get_loc(last_cross)
        streak = 0
        i = pos - 1
        while i >= 0 and not bool(abv.iloc[i]):
            streak += 1
            i -= 1
        results.append(
            {
                "ticker": ticker,
                "last_close": float(close[ticker].iloc[-1]),
                "ema": float(ema[ticker].iloc[-1]),
                "cross_date": last_cross,
                "below_streak_periods": streak,
                "streak_years": streak / periods_per_year,
                "periods_since_cross": len(close) - 1 - pos,
                "currently_above": bool(abv.iloc[-1]),
            }
        )

    frame = pd.DataFrame(results)
    if frame.empty:
        return frame
    frame["pct_above_ema"] = frame["last_close"] / frame["ema"] - 1
    if min_below_periods > 0:
        frame = frame[frame["below_streak_periods"] >= min_below_periods]
    return frame.sort_values("below_streak_periods", ascending=False).reset_index(drop=True)


def format_return(value: float) -> str:
    return f"{value * 100:0.2f}%"


MF_PERIODS = _APP["mf_periods"]
YF_PERIOD_MAP = _APP["yf_period_map"]

BACKTEST_PERIODS = _APP["backtest_periods"]

EMA_PERIODS = _APP["ema_periods"]
EMA_TIMEFRAMES = {k: tuple(v) for k, v in _APP["ema_timeframes"].items()}
EMA_MIN_YEARS = _APP["ema_min_years"]

DIVIDEND_PERIODS = _APP["dividend_periods"]

LOOKBACK_OPTIONS = _APP["lookback_options"]
TOP_N_OPTIONS = _APP["top_n_options"]
EMA_SPAN_OPTIONS = _APP["ema_span_options"]

DEFAULT_MOMENTUM_WEIGHT = _APP["default_momentum_weight"]
DEFAULT_CMF_WEIGHT = _APP["default_cmf_weight"]
DEFAULT_MF_WEIGHT = _APP["default_mf_weight"]
MF_FLOW_PERIOD = _APP["mf_flow_period"]
ADVISOR_RELIABILITY_PERIOD = _APP["advisor_reliability_period"]
ADVISOR_REGIME_PERIOD = _APP["advisor_regime_period"]
PROVIDER_OPTIONS = _APP["provider_options"]

PAGE_TITLE = _APP["page_title"]
PAGE_ICON = _APP["page_icon"]
PAGE_LAYOUT = _APP["layout"]
APP_TITLE = _APP["title"]
APP_SUBTITLE = _APP["subtitle"]
TAB_NAMES = _APP["tabs"]


def load_mf_flows(path: Path | str = None) -> pd.DataFrame:
    if path is None:
        path = data_file("mf_flows")
    return pd.read_csv(path)


def filter_mf_flows(df: pd.DataFrame, period: str) -> pd.DataFrame:
    return df[df["period"].astype(str).str.upper() == period.upper()].copy()


def format_crore(value: float) -> str:
    return f"₹{value:,.0f} Cr"


def compute_ticker_period_returns(close: pd.DataFrame, nifty50_map: dict) -> pd.DataFrame:
    """Ticker-level current price and period return derived from a close frame."""
    last = close.iloc[-1]
    first = close.iloc[0]
    rows = []
    for ticker in close.columns:
        rows.append(
            {
                "ticker": ticker,
                "sector": nifty50_map.get(ticker, "Other"),
                "current_price": float(last[ticker]),
                "period_return": float(last[ticker] / first[ticker] - 1),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=_TTL, show_spinner=False)
def fetch_dividend_history(tickers, period):
    """Fetch split-adjusted close + dividend (actions) history per ticker."""
    frames = _download_chunked(tickers, period, interval="1d", auto_adjust=True, actions=True)
    result = {}
    for ticker, df in frames.items():
        cols = [c for c in ["Close", "Dividends"] if c in df.columns]
        if "Close" not in cols:
            continue
        result[ticker] = df[cols].copy()
    return result


@st.cache_data(ttl=_TTL, show_spinner=False)
def compute_dividend_stats(dividend_history: dict, nifty50_map: dict) -> pd.DataFrame:
    """Per-ticker dividend summary over the trailing window.

    For every ticker the annual dividend total is divided by that year's
    average close to get a per-year yield, and the window average of those
    yields is reported as ``avg_dividend_yield`` (%). Non-payers are excluded.

    Returns a DataFrame with ticker, sector, current_price,
    avg_annual_dividend, avg_dividend_yield, total_dividend, years_count
    sorted by yield descending.
    """
    rows = []
    for ticker, df in dividend_history.items():
        try:
            if "Dividends" not in df.columns or df.empty:
                continue
            years = df.index.year
            annual_div = df.groupby(years)["Dividends"].sum()
            if float(annual_div.sum()) <= 0:
                continue
            avg_close_year = df.groupby(years)["Close"].mean().replace(0, np.nan)
            yield_pct = annual_div / avg_close_year * 100
            total_div = float(annual_div.sum())
            rows.append(
                {
                    "ticker": ticker,
                    "sector": nifty50_map.get(ticker, "Other"),
                    "current_price": float(df["Close"].iloc[-1]),
                    "avg_annual_dividend": total_div / len(annual_div),
                    "avg_dividend_yield": float(yield_pct.mean()),
                    "total_dividend": total_div,
                    "years_count": len(annual_div),
                }
            )
        except Exception:
            continue
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("avg_dividend_yield", ascending=False).reset_index(drop=True)


def render_mf_tab(nifty50_map: dict):
    st.subheader("Mutual Fund Sector Flows")
    st.markdown(
        "Sector-wise mutual fund buying and selling, aggregated across major equity "
        "schemes from monthly portfolio disclosures (in ₹ crore)."
    )
    try:
        flows = load_mf_flows()
    except FileNotFoundError:
        st.error(
            "Could not find data/mf_sector_flows.csv. Generate it with "
            "`python data/generate_mf_snapshot.py` or refresh it with "
            "`python data/refresh_mf_amfi.py`."
        )
        return
    except Exception as exc:
        st.error(f"Failed to load mutual fund flows: {exc}")
        return

    period = st.selectbox("Period", MF_PERIODS, index=0, key="mf_period")
    period_flows = filter_mf_flows(flows, period)
    if period_flows.empty:
        st.warning(f"No mutual fund flow data for period {period}.")
        return

    period_flows = period_flows.sort_values("net_crore", ascending=False).reset_index(drop=True)
    period_flows["buy_crore_str"] = period_flows["buy_crore"].map(format_crore)
    period_flows["sell_crore_str"] = period_flows["sell_crore"].map(format_crore)
    period_flows["net_crore_str"] = period_flows["net_crore"].map(format_crore)

    fig = px.bar(
        period_flows,
        x="net_crore",
        y="sector",
        orientation="h",
        text="net_crore_str",
        title=f"Net mutual fund buying/selling by sector — {period}",
        labels={"net_crore": "Net flow (₹ crore)", "sector": "Sector"},
        color="net_crore",
        color_continuous_scale=["#c0392b", "#ecf0f1", "#27ae60"],
        color_continuous_midpoint=0,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        height=max(420, 32 * period_flows.shape[0]),
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown("### Sector flows table")
    st.dataframe(
        period_flows[
            ["sector", "buy_crore_str", "sell_crore_str", "net_crore_str"]
        ].rename(
            columns={
                "sector": "Sector",
                "buy_crore_str": "Buy (₹ crore)",
                "sell_crore_str": "Sell (₹ crore)",
                "net_crore_str": "Net (₹ crore)",
            }
        ),
        width='stretch',
    )

    st.markdown("### Sector-wise related stocks")
    with st.spinner("Fetching price data from Yahoo Finance..."):
        tickers = list(nifty50_map.keys())
        close, failed = fetch_close_prices(tickers, YF_PERIOD_MAP[period])
    if failed:
        st.caption(f"Price data not found for: {', '.join(failed)}")
    if close.empty:
        st.warning("No price data available for the selected period.")
        return

    ticker_df = compute_ticker_period_returns(close, nifty50_map)
    sectors = sorted(ticker_df["sector"].dropna().unique())
    selected = st.multiselect("Sector", sectors, default=sectors)
    if selected:
        ticker_df = ticker_df[ticker_df["sector"].isin(selected)]
    ticker_df = ticker_df.sort_values(
        ["sector", "period_return"], ascending=[True, False]
    ).reset_index(drop=True)
    ticker_df["current_price"] = ticker_df["current_price"].map(lambda x: f"{x:0.2f}")
    ticker_df["period_return"] = ticker_df["period_return"].map(format_return)
    st.dataframe(
        ticker_df[["ticker", "sector", "current_price", "period_return"]],
        width='stretch',
    )


@st.cache_data(ttl=_TTL, show_spinner=False)
def cached_reliability_summary(period: str):
    """Backtest signal-reliability stats used to ground the AI advisor."""
    nifty50_map = load_nifty50_map()
    tickers = list(nifty50_map.keys())
    try:
        close, _ = fetch_close_prices(tickers, period)
        if close.empty:
            return None
        ohlcv = {t: df.reindex(close.index) for t, df in fetch_ohlcv(tickers, period).items()}
        return backtest.signal_reliability(
            close.copy(), ohlcv, nifty50_map, lookbacks=backtest.LOOKBACKS
        )
    except Exception:
        return None


@st.cache_data(ttl=_TTL, show_spinner=False)
def compute_backtest_analysis(
    universe,
    period,
    momentum_lookback,
    cmf_lookback,
    momentum_weight,
    cmf_weight,
    top_n,
    use_vol_adjust=False,
    apply_regime=False,
    regime_adx=False,
):
    """Run backtest + signal reliability + current composite ranking.

    ``universe`` is "stocks" (Nifty 50 members by sector) or "indices" (Nifty
    sector indices directly). ``use_vol_adjust`` swaps raw momentum for
    volatility-adjusted momentum; ``apply_regime`` gates positions to cash when
    the Nifty 50 regime is flat, with ``regime_adx`` additionally requiring ADX
    confirmation.

    Returns (reliability_df, backtest_result_dict, ranking_df) or None on failure.
    """
    if universe == "indices":
        nifty50_map = backtest.load_sector_indices()
    else:
        nifty50_map = load_nifty50_map()
    tickers = list(nifty50_map)
    close, _ = fetch_close_prices(tickers, period)
    if close.empty:
        return None
    close = close.copy()
    ohlcv = {t: df.reindex(close.index) for t, df in fetch_ohlcv(tickers, period).items()}

    regime = None
    if apply_regime:
        regime = compute_regime_series(period, require_adx=regime_adx)

    reliability = backtest.signal_reliability(
        close,
        ohlcv,
        nifty50_map,
        lookbacks=backtest.LOOKBACKS,
        use_vol_adjust=use_vol_adjust,
    )
    result = backtest.run_backtest(
        close,
        ohlcv,
        nifty50_map,
        momentum_lookback=momentum_lookback,
        cmf_lookback=cmf_lookback,
        momentum_weight=momentum_weight,
        cmf_weight=cmf_weight,
        top_n=top_n,
        use_vol_adjust=use_vol_adjust,
        regime=regime,
    )

    smom = (
        backtest.sector_vol_adj_momentum(close, nifty50_map, momentum_lookback).iloc[-1]
        if use_vol_adjust
        else backtest.sector_momentum(close, nifty50_map, momentum_lookback).iloc[-1]
    )
    scmf = backtest.sector_cmf(ohlcv, nifty50_map, cmf_lookback).iloc[-1]
    mf_flows = None
    try:
        flows = load_mf_flows()
        mf_flows = filter_mf_flows(flows, MF_FLOW_PERIOD)
    except Exception:
        pass
    comp = backtest.composite_scores(
        close,
        ohlcv,
        nifty50_map,
        momentum_lookback,
        cmf_lookback,
        momentum_weight,
        cmf_weight,
        mf_flows=mf_flows,
        mf_weight=DEFAULT_MF_WEIGHT,
        use_vol_adjust=use_vol_adjust,
    )
    ranking = pd.DataFrame(
        {"momentum": smom, "cmf": scmf, "composite": comp.iloc[-1]}
    )
    if mf_flows is not None and not mf_flows.empty:
        mf_map = dict(zip(mf_flows["sector"], mf_flows["net_crore"]))
        ranking["mf_net_crore"] = ranking.index.map(lambda s: mf_map.get(s))
    ranking = ranking.dropna(subset=["composite"]).sort_values(
        "composite", ascending=False
    )
    return reliability, result, ranking


@st.cache_data(ttl=_TTL, show_spinner=False)
def compute_regime_series(period, require_adx=False):
    """Market regime series (True = trending) from the Nifty 50 index."""
    idx_close, _ = fetch_close_prices([backtest.REGIME_INDEX], period)
    if idx_close.empty or backtest.REGIME_INDEX not in idx_close.columns:
        return pd.Series(dtype=bool)
    close_s = idx_close[backtest.REGIME_INDEX]
    idx_ohlcv = fetch_ohlcv([backtest.REGIME_INDEX], period).get(backtest.REGIME_INDEX)
    if idx_ohlcv is None:
        return backtest.compute_regime(close_s, require_adx=require_adx)
    return backtest.compute_regime(
        close_s, idx_ohlcv.reindex(close_s.index), require_adx=require_adx
    )


@st.cache_data(ttl=_TTL, show_spinner=False)
def compute_market_regime(period):
    """Regime series + latest Nifty 50 trend/ADX metrics for the dashboard."""
    regime = compute_regime_series(period)
    if regime.empty:
        return None
    idx_close, _ = fetch_close_prices([backtest.REGIME_INDEX], period)
    close_s = idx_close[backtest.REGIME_INDEX].reindex(regime.index)
    idx_ohlcv = fetch_ohlcv([backtest.REGIME_INDEX], period).get(backtest.REGIME_INDEX)
    adx = float("nan")
    if idx_ohlcv is not None:
        idx_ohlcv = idx_ohlcv.reindex(regime.index)
        adx = float(
            backtest.compute_adx(
                idx_ohlcv["High"], idx_ohlcv["Low"], idx_ohlcv["Close"]
            ).iloc[-1]
        )
    return {
        "regime": regime,
        "close": close_s,
        "latest": {
            "close": float(close_s.iloc[-1]),
            "sma_fast": float(close_s.rolling(backtest.REGIME_SMA_FAST).mean().iloc[-1]),
            "sma_slow": float(close_s.rolling(backtest.REGIME_SMA_SLOW).mean().iloc[-1]),
            "adx": adx,
            "in_market_now": bool(regime.iloc[-1]),
            "pct_in_market_3m": float(regime.iloc[-60:].mean()),
        },
    }


@st.cache_data(ttl=_TTL, show_spinner=False)
def load_fii_dii():
    """Latest NSE FII/DII cash flows (data/fii_dii_flows.csv) or None."""
    path = data_file("fii_dii")
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(ttl=_TTL, show_spinner=False)
def load_delivery_volume():
    """Delivery-volume report (data/delivery_volume.csv) or None."""
    path = data_file("delivery_volume")
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def render_backtest_tab(nifty50_map: dict):
    st.subheader("Backtest & Composite Score")
    st.markdown(
        "Simulates *'buy top-N sectors, rebalance periodically, equal weight'* using "
        "momentum + Chaikin Money Flow (CMF). Use the reliability table to see which "
        "lookbacks actually predicted 6-month returns before trusting them."
    )

    period = st.selectbox("Data window", BACKTEST_PERIODS, index=1, key="backtest_period")
    col1, col2, col3 = st.columns(3)
    with col1:
        momentum_lookback = st.select_slider(
            "Momentum lookback (days)", options=LOOKBACK_OPTIONS, value=20, key="bt_mom_lb"
        )
    with col2:
        cmf_lookback = st.select_slider(
            "CMF lookback (days)", options=LOOKBACK_OPTIONS, value=20, key="bt_cmf_lb"
        )
    with col3:
        top_n = st.selectbox("Top N sectors", TOP_N_OPTIONS, index=1, key="bt_top_n")
    cw1, cw2 = st.columns(2)
    with cw1:
        momentum_weight = st.slider("Momentum weight", 0.0, 1.0, DEFAULT_MOMENTUM_WEIGHT, key="bt_mom_w")
    with cw2:
        cmf_weight = st.slider("CMF weight", 0.0, 1.0, DEFAULT_CMF_WEIGHT, key="bt_cmf_w")

    with st.spinner("Running backtest over history..."):
        analysis = compute_backtest_analysis(
            "stocks",
            period,
            momentum_lookback,
            cmf_lookback,
            momentum_weight,
            cmf_weight,
            top_n,
        )
    if analysis is None:
        st.error("Backtest failed — check data availability for the selected window.")
        return
    reliability, result, ranking = analysis
    metrics = result["metrics"]
    if not metrics:
        st.warning("Not enough history for the chosen lookbacks.")
        return

    st.markdown("### Strategy performance")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total return", format_return(metrics["total_return"]))
    m2.metric("Benchmark", format_return(metrics["benchmark_total_return"]))
    m3.metric("Hit rate", f"{metrics['hit_rate']:.0%}")
    m4.metric("Avg excess/period", format_return(metrics["avg_excess_return"]))
    m5.metric("Max drawdown", format_return(metrics["max_drawdown"]))
    m6.metric("Sharpe", f"{metrics['sharpe']:.2f}")

    plot_df = pd.DataFrame(
        {
            "Strategy": result["value"],
            "Benchmark (equal weight)": result["benchmark_value"],
        }
    )
    fig = px.line(
        plot_df,
        title="Cumulative growth — top-N sector rotation vs benchmark",
        labels={"value": "Cumulative value", "index": "Date"},
    )
    fig.update_yaxes(tickformat=".2f")
    st.plotly_chart(fig, width='stretch')

    st.markdown("### Signal reliability by lookback")
    rel_display = reliability.sort_values("hit_rate", ascending=False)
    rel_display = rel_display.copy()
    for col in ["hit_rate", "avg_sector_return", "avg_benchmark_return",
                "avg_excess_return", "median_excess_return", "win_rate_positive"]:
        if col in rel_display.columns:
            rel_display[col] = rel_display[col].map(format_return)
    st.dataframe(
        rel_display.rename(
            columns={
                "lookback": "Lookback (days)",
                "n_samples": "Samples",
                "hit_rate": "Hit rate",
                "avg_sector_return": "Avg sector 6m",
                "avg_benchmark_return": "Avg benchmark 6m",
                "avg_excess_return": "Avg excess 6m",
                "median_excess_return": "Median excess 6m",
                "win_rate_positive": "Win rate (+)",
            }
        ),
        width='stretch',
    )

    st.markdown("### Current composite ranking")
    st.caption(
        "Score = weighted momentum + CMF + optional mutual-fund flow overlay (3M). "
        "Higher is stronger; use with the reliability table above."
    )
    display = ranking.copy()
    display["momentum"] = display["momentum"].map(format_return)
    display["cmf"] = display["cmf"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    display["composite"] = display["composite"].round(3)
    if "mf_net_crore" in display.columns:
        display["mf_flow"] = display["mf_net_crore"].map(format_crore)
        display = display.drop(columns=["mf_net_crore"])
    st.dataframe(display, width='stretch')

    with st.expander("Per-rebalance detail"):
        if not result["period_stats"].empty:
            detail = result["period_stats"].copy()
            detail["portfolio_return"] = detail["portfolio_return"].map(format_return)
            detail["benchmark_return"] = detail["benchmark_return"].map(format_return)
            detail["excess_return"] = detail["excess_return"].map(format_return)
            st.dataframe(
                detail.rename(
                    columns={
                        "date": "Rebalance date",
                        "top_sectors": "Top sectors",
                        "portfolio_return": "Strategy return",
                        "benchmark_return": "Benchmark return",
                        "excess_return": "Excess return",
                    }
                ),
                width='stretch',
            )


def render_expansion_tab(nifty50_map: dict):
    st.subheader("Market Regime & Expanded Universe")
    st.markdown(
        "Combines a **trend/ADX regime filter** on the Nifty 50 (the strategy sits in "
        "cash when the market is choppy), **FII/DII flow confirmation**, and an "
        "**expanded universe** of Nifty sector indices alongside the Nifty 50 stock map."
    )

    period = st.selectbox("Data window", BACKTEST_PERIODS, index=1, key="ex_period")

    with st.spinner("Detecting market regime..."):
        regime_info = compute_market_regime(period)
    if regime_info is None:
        st.error("Could not fetch the Nifty 50 index — the regime filter is unavailable.")
        return

    latest = regime_info["latest"]
    state = "TRENDING" if latest["in_market_now"] else "FLAT / CHOP"
    st.markdown("### Market regime (Nifty 50)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Regime", state)
    c2.metric("Nifty 50", f"{latest['close']:,.0f}")
    c3.metric(f"SMA{backtest.REGIME_SMA_FAST}", f"{latest['sma_fast']:,.0f}")
    c4.metric(f"SMA{backtest.REGIME_SMA_SLOW}", f"{latest['sma_slow']:,.0f}")
    c5.metric("ADX (14)", f"{latest['adx']:.1f}")
    st.caption(
        f"Regime is TRENDING when Nifty trades above SMA{backtest.REGIME_SMA_SLOW} "
        f"(optionally also SMA{backtest.REGIME_SMA_FAST} + ADX >= "
        f"{backtest.REGIME_ADX_THRESHOLD:g} when ADX confirmation is enabled). "
        f"The strategy was in market {latest['pct_in_market_3m']:.0%} of the last 60 sessions."
    )

    chart = pd.DataFrame(
        {
            "Nifty 50": regime_info["close"],
            f"SMA{backtest.REGIME_SMA_FAST}": regime_info["close"].rolling(
                backtest.REGIME_SMA_FAST
            ).mean(),
            f"SMA{backtest.REGIME_SMA_SLOW}": regime_info["close"].rolling(
                backtest.REGIME_SMA_SLOW
            ).mean(),
            "state": regime_info["regime"].map(
                {True: "In market", False: "Cash (regime off)"}
            ),
        }
    )
    fig = px.line(
        chart,
        y=["Nifty 50", f"SMA{backtest.REGIME_SMA_FAST}", f"SMA{backtest.REGIME_SMA_SLOW}"],
        title="Nifty 50 with trend SMAs (grey = regime off, i.e. cash)",
        labels={"value": "Index level", "index": "Date"},
    )
    fig.add_scatter(
        x=chart.index[chart["state"] == "Cash (regime off)"],
        y=chart["Nifty 50"][chart["state"] == "Cash (regime off)"],
        mode="markers",
        marker=dict(color="grey", size=4),
        name="Cash (regime off)",
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown("### FII / DII cash flows")
    flows = load_fii_dii()
    if flows is None or flows.empty:
        st.info(
            "No FII/DII data yet. Run `python data/refresh_fii_dii.py` to pull the "
            "latest NSE cash-market figures."
        )
    else:
        latest_flow = flows.sort_values("date").groupby("category").tail(1)
        display = latest_flow.rename(
            columns={
                "date": "Date",
                "category": "Category",
                "buy_value_crore": "Buy (₹ cr)",
                "sell_value_crore": "Sell (₹ cr)",
                "net_value_crore": "Net (₹ cr)",
            }
        )
        st.dataframe(display, width='stretch')
        st.caption(
            f"Source: NSE FII/DII activity, {flows['date'].iloc[0]} — {flows['date'].iloc[-1]}."
        )
    delivery = load_delivery_volume()
    if delivery is None or delivery.empty:
        st.caption("Delivery volume: unavailable — NSE's WAF blocks the report endpoint.")
    else:
        st.caption("Top deliverable-quantity (accumulation) today:")
        st.dataframe(
            delivery.sort_values("delivery_pct", ascending=False).head(8),
            width='stretch',
        )

    st.markdown("### Expanded universe & composite ranking")
    uni = st.radio(
        "Universe",
        ["Nifty 50 stocks", "Nifty sector indices"],
        horizontal=True,
        key="ex_universe",
    )
    v1, v2, v3 = st.columns(3)
    with v1:
        use_vol_adjust = st.checkbox("Volatility-adjusted momentum", value=True, key="ex_vol")
    with v2:
        apply_regime = st.checkbox("Regime filter (cash when flat)", value=True, key="ex_regime")
    with v3:
        top_n = st.selectbox("Top N sectors", TOP_N_OPTIONS, index=1, key="ex_top_n")
    if apply_regime:
        regime_adx = st.checkbox(
            "ADX confirmation (stricter — sits out slow grinds)",
            value=False,
            key="ex_regime_adx",
            help="Requires ADX >= 20 in addition to trading above the 200-day MA. "
                 "Backtests show this keeps the strategy out of the market too much.",
        )
    else:
        regime_adx = False
    lookback = st.select_slider(
        "Signal lookback (days)", options=LOOKBACK_OPTIONS, value=20, key="ex_lb"
    )

    universe_key = "indices" if uni == "Nifty sector indices" else "stocks"
    with st.spinner("Running backtest over the selected universe..."):
        analysis = compute_backtest_analysis(
            universe_key, period, lookback, lookback, DEFAULT_MOMENTUM_WEIGHT, DEFAULT_CMF_WEIGHT, top_n,
            use_vol_adjust, apply_regime, regime_adx,
        )
    if analysis is None:
        st.error("Backtest failed — check data availability for the selected window.")
        return
    reliability, result, ranking = analysis
    metrics = result["metrics"]
    if not metrics:
        st.warning("Not enough history for the chosen lookbacks.")
        return

    baseline = None
    if apply_regime:
        with st.spinner("Comparing against the always-in strategy..."):
            baseline = compute_backtest_analysis(
                universe_key, period, lookback, lookback, DEFAULT_MOMENTUM_WEIGHT, DEFAULT_CMF_WEIGHT, top_n,
                use_vol_adjust, False, False,
            )

    st.markdown("#### Strategy performance")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total return", format_return(metrics["total_return"]))
    m2.metric("Benchmark", format_return(metrics["benchmark_total_return"]))
    m3.metric("Hit rate", f"{metrics['hit_rate']:.0%}")
    m4.metric("Avg excess/period", format_return(metrics["avg_excess_return"]))
    m5.metric("Max drawdown", format_return(metrics["max_drawdown"]))
    m6.metric("Sharpe", f"{metrics['sharpe']:.2f}")

    if baseline is not None and baseline[1]["metrics"]:
        bm = baseline[1]["metrics"]
        cmp_df = pd.DataFrame(
            {
                "Run": ["Always in market", "Regime filter on"],
                "Total return": [format_return(bm["total_return"]), format_return(metrics["total_return"])],
                "Sharpe": [f"{bm['sharpe']:.2f}", f"{metrics['sharpe']:.2f}"],
                "Max drawdown": [format_return(bm["max_drawdown"]), format_return(metrics["max_drawdown"])],
                "Hit rate": [f"{bm['hit_rate']:.0%}", f"{metrics['hit_rate']:.0%}"],
            }
        )
        st.markdown("#### Regime filter impact")
        st.dataframe(cmp_df, width='stretch')

    plot_df = pd.DataFrame(
        {
            "Strategy": result["value"],
            "Benchmark (equal weight)": result["benchmark_value"],
        }
    )
    if baseline is not None and baseline[1]["metrics"]:
        plot_df["Always in market"] = baseline[1]["value"].reindex(plot_df.index)
    fig2 = px.line(
        plot_df,
        title="Cumulative growth — expanded universe rotation",
        labels={"value": "Cumulative value", "index": "Date"},
    )
    fig2.update_yaxes(tickformat=".2f")
    st.plotly_chart(fig2, width='stretch')

    st.markdown("#### Current composite ranking")
    display = ranking.copy()
    display["momentum"] = display["momentum"].map(format_return)
    display["cmf"] = display["cmf"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    display["composite"] = display["composite"].round(3)
    if "mf_net_crore" in display.columns:
        display["mf_flow"] = display["mf_net_crore"].map(format_crore)
        display = display.drop(columns=["mf_net_crore"])
    st.dataframe(display, width='stretch')

    st.markdown("#### Signal reliability by lookback")
    rel_display = reliability.sort_values("hit_rate", ascending=False).copy()
    for col in ["hit_rate", "avg_sector_return", "avg_benchmark_return",
                "avg_excess_return", "median_excess_return", "win_rate_positive"]:
        if col in rel_display.columns:
            rel_display[col] = rel_display[col].map(format_return)
    st.dataframe(
        rel_display.rename(
            columns={
                "lookback": "Lookback (days)",
                "n_samples": "Samples",
                "hit_rate": "Hit rate",
                "avg_sector_return": "Avg sector 6m",
                "avg_benchmark_return": "Avg benchmark 6m",
                "avg_excess_return": "Avg excess 6m",
                "median_excess_return": "Median excess 6m",
                "win_rate_positive": "Win rate (+)",
            }
        ),
        width='stretch',
    )

    with st.expander("Per-rebalance detail"):
        if not result["period_stats"].empty:
            detail = result["period_stats"].copy()
            detail["portfolio_return"] = detail["portfolio_return"].map(format_return)
            detail["benchmark_return"] = detail["benchmark_return"].map(format_return)
            detail["excess_return"] = detail["excess_return"].map(format_return)
            st.dataframe(
                detail.rename(
                    columns={
                        "date": "Rebalance date",
                        "top_sectors": "Top sectors",
                        "portfolio_return": "Strategy return",
                        "benchmark_return": "Benchmark return",
                        "excess_return": "Excess return",
                        "in_market": "In market",
                    }
                ),
                width='stretch',
            )


def render_ema_tab(nifty50_map: dict):
    st.subheader("Long-Downtrend Breakout Screener")
    st.markdown(
        "Flags stocks whose price most recently crossed **above** their EMA after "
        "spending a long stretch *below* it — a potential multi-year trend breakout. "
        "Run the scan on daily, **weekly** or **monthly** chart data."
    )

    period = st.selectbox("Data window", EMA_PERIODS, index=2, key="ema_period")
    timeframe = st.selectbox(
        "Chart timeframe", list(EMA_TIMEFRAMES), index=1, key="ema_timeframe"
    )
    freq, periods_per_year = EMA_TIMEFRAMES[timeframe]
    ema_span = st.select_slider(
        "EMA span (periods)", options=EMA_SPAN_OPTIONS, value=20, key="ema_span"
    )
    min_years_label = st.selectbox(
        "Minimum time below EMA before the breakout",
        list(EMA_MIN_YEARS),
        index=3,
        key="ema_min_years",
    )
    min_periods = EMA_MIN_YEARS[min_years_label] * periods_per_year

    with st.spinner(f"Fetching {period} of price data ({timeframe} chart)..."):
        tickers = list(nifty50_map.keys())
        close, failed = fetch_close_prices(tickers, period)
    if failed:
        st.caption(f"Price data not found for: {', '.join(failed)}")
    if close.empty:
        st.error("No price data loaded for the selected window.")
        return

    chart_close = resample_close(close, freq)
    breakouts = compute_ema_breakouts(
        chart_close,
        ema_span=ema_span,
        min_below_periods=min_periods,
        periods_per_year=periods_per_year,
    )
    if breakouts.empty:
        st.info(
            f"No stocks crossed above the {ema_span}-period EMA on the {timeframe.lower()} "
            f"chart after being below it for {min_years_label.lower()} or longer."
        )
        return

    period_label = {"Daily": "days", "Weekly": "weeks", "Monthly": "months"}[timeframe]
    display = breakouts.copy()
    display["sector"] = display["ticker"].map(lambda t: nifty50_map.get(t, "Other"))
    display["last_close"] = display["last_close"].map(lambda x: f"{x:0.2f}")
    display["ema"] = display["ema"].map(lambda x: f"{x:0.2f}")
    display["pct_above_ema"] = display["pct_above_ema"].map(format_return)
    display["cross_date"] = display["cross_date"].dt.date
    display["streak_years"] = display["streak_years"].map(lambda x: f"{x:0.1f}y")
    display["currently_above"] = display["currently_above"].map(lambda b: "Above" if b else "Below")

    columns = {
        "ticker": "Ticker",
        "sector": "Sector",
        "last_close": "Close",
        "ema": f"{ema_span}-p EMA",
        "pct_above_ema": "Close vs EMA",
        "cross_date": "Cross date",
        "below_streak_periods": f"Below streak ({period_label})",
        "streak_years": "Streak (yrs)",
        "periods_since_cross": f"Since cross ({period_label})",
        "currently_above": "Status",
    }
    st.dataframe(display[list(columns)].rename(columns=columns), width='stretch')

    st.caption(
        "Interpretation: a stock that sat below its EMA for years and then crossed above "
        "is breaking out of a long downtrend. Confirm with volume and momentum before "
        "acting — this is a screener, not a recommendation."
    )


def render_dividend_tab(nifty50_map: dict):
    st.subheader("Dividend Yield — Large-cap Dividend Stocks")
    st.markdown(
        "Dividend yield over the trailing window for the Nifty 50 plus other "
        "large-cap, liquid companies with established market value and reputation. "
        "Yield = average of *annual dividends ÷ that year's average close price*."
    )

    universe_label = st.radio(
        "Universe",
        ["Nifty 50 + large caps", "Nifty 50 only"],
        index=0,
        horizontal=True,
        key="div_universe",
    )
    dividend_map = nifty50_map if universe_label == "Nifty 50 only" else load_dividend_universe()

    period = st.selectbox("Data window", DIVIDEND_PERIODS, index=1, key="div_period")
    with st.spinner("Fetching dividend history from Yahoo Finance..."):
        tickers = list(dividend_map.keys())
        history = fetch_dividend_history(tickers, period)
    if not history:
        st.error("Could not fetch dividend data for the selected window.")
        return

    stats = compute_dividend_stats(history, dividend_map)
    if stats.empty:
        st.info("No dividend data found for the selected window.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Dividend-paying tickers", len(stats))
    c2.metric("Median yield", f"{stats['avg_dividend_yield'].median():.2f}%")
    c3.metric("Top yield", f"{stats['avg_dividend_yield'].max():.2f}%")

    top = stats.head(15)
    fig = px.bar(
        top,
        x="ticker",
        y="avg_dividend_yield",
        text="avg_dividend_yield",
        title=f"Top 15 dividend yields — last {period}",
        labels={"ticker": "Ticker", "avg_dividend_yield": "Avg dividend yield (%)"},
        color="avg_dividend_yield",
        color_continuous_scale="greens",
    )
    fig.update_traces(textposition="outside", texttemplate="%{text:.2f}%")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, width='stretch')

    sectors = sorted(stats["sector"].dropna().unique())
    selected = st.multiselect("Sector", sectors, default=sectors)
    if selected:
        stats = stats[stats["sector"].isin(selected)]

    display = stats.copy()
    display["current_price"] = display["current_price"].map(lambda x: f"{x:0.2f}")
    display["avg_annual_dividend"] = display["avg_annual_dividend"].map(lambda x: f"{x:0.2f}")
    display["avg_dividend_yield"] = display["avg_dividend_yield"].map(lambda x: f"{x:.2f}%")
    display["total_dividend"] = display["total_dividend"].map(lambda x: f"{x:0.2f}")
    st.dataframe(
        display.rename(
            columns={
                "ticker": "Ticker",
                "sector": "Sector",
                "current_price": "Current price (₹)",
                "avg_annual_dividend": "Avg annual dividend (₹)",
                "avg_dividend_yield": "Avg dividend yield",
                "total_dividend": f"Total dividend {period} (₹)",
                "years_count": "Years",
            }
        ),
        width='stretch',
    )

    st.caption(
        "Universe is the Nifty 50 plus other large-cap, liquid and well-reputed "
        "companies (Nifty Next 50 members and PSUs included). Dividends are "
        "split-adjusted and yields are trailing; past payouts do not guarantee "
        "future ones. Research/educational display only."
    )


def main():
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=PAGE_LAYOUT,
    )

    st.title(APP_TITLE)
    st.markdown(APP_SUBTITLE)

    try:
        nifty50_map = load_nifty50_map()
    except FileNotFoundError:
        st.error("Could not find nifty50_map.json in the project root.")
        return
    except json.JSONDecodeError as exc:
        st.error(f"nifty50_map.json contains invalid JSON: {exc}")
        return

    uploaded = st.file_uploader(
        "Upload ticker->sector CSV (columns: ticker,sector) to override mapping for this session",
        type="csv",
    )
    if uploaded is not None:
        try:
            df_map = pd.read_csv(uploaded)
            if "ticker" in df_map.columns and "sector" in df_map.columns:
                nifty50_map = dict(zip(df_map["ticker"].astype(str), df_map["sector"].astype(str)))
                st.success("Mapping uploaded and will be used for this session.")
            else:
                st.error("CSV must contain 'ticker' and 'sector' columns.")
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")

    tab_rotation, tab_mf, tab_backtest, tab_regime, tab_ema, tab_dividend, tab_advisor = st.tabs(
        TAB_NAMES
    )
    with tab_rotation:
        render_rotation_tab(nifty50_map)
    with tab_mf:
        render_mf_tab(nifty50_map)
    with tab_backtest:
        render_backtest_tab(nifty50_map)
    with tab_regime:
        render_expansion_tab(nifty50_map)
    with tab_ema:
        render_ema_tab(nifty50_map)
    with tab_dividend:
        render_dividend_tab(nifty50_map)
    with tab_advisor:
        render_advisor_tab(nifty50_map)


def _format_table(df, columns, pct_cols=()):
    frame = df.copy()
    for col in columns:
        if col in frame and col in pct_cols:
            frame[col] = frame[col].map(format_return)
    return frame[columns].to_string(index=False)


def build_advisor_context(nifty50_map: dict) -> str:
    rotation_period = st.session_state.get("rotation_period", _APP["default_rotation_period"])
    lookback_days = st.session_state.get("rotation_lookback", _APP["default_lookback"])
    mf_period = st.session_state.get("mf_period", _APP["default_mf_period"])

    parts = [
        f"Active dashboard filters — Nifty rotation window: {rotation_period}, "
        f"momentum lookback: {lookback_days} days, mutual fund flow period: {mf_period}."
    ]

    close, failed = fetch_close_prices(list(nifty50_map.keys()), rotation_period)
    if close.empty:
        parts.append("No Nifty 50 price data could be fetched.")
    else:
        momentum, sector_summary = compute_sector_rotation(close, lookback_days, nifty50_map)
        top = sector_summary.head(5).reset_index()
        bottom = sector_summary.tail(3).reset_index()
        ranked = momentum.sort_values("rank")
        parts.append(
            "Top sectors by average momentum:\n"
            + _format_table(top, ["sector", "symbol_count", "average_momentum", "average_period_return"], pct_cols=("average_momentum", "average_period_return"))
        )
        parts.append(
            "Weakest sectors:\n"
            + _format_table(bottom, ["sector", "symbol_count", "average_momentum", "average_period_return"], pct_cols=("average_momentum", "average_period_return"))
        )
        parts.append(
            "Top momentum tickers:\n"
            + _format_table(ranked.head(8), ["ticker", "sector", "momentum", "period_return"], pct_cols=("momentum", "period_return"))
        )
        parts.append(
            "Weakest tickers:\n"
            + _format_table(ranked.tail(8), ["ticker", "sector", "momentum", "period_return"], pct_cols=("momentum", "period_return"))
        )

    try:
        flows = load_mf_flows()
        period_flows = filter_mf_flows(flows, mf_period).sort_values("net_crore", ascending=False)
        if period_flows.empty:
            parts.append(f"No mutual fund flow data for period {mf_period}.")
        else:
            parts.append(
                f"Mutual fund sector flows ({mf_period}) — top net buying:\n"
                + _format_table(period_flows.head(5), ["sector", "buy_crore", "sell_crore", "net_crore"])
            )
            parts.append(
                f"Mutual fund sector flows ({mf_period}) — top net selling:\n"
                + _format_table(period_flows.tail(5), ["sector", "buy_crore", "sell_crore", "net_crore"])
            )
            parts.append(
                "NOTE: mutual fund flows are from a bundled illustrative snapshot "
                "unless refreshed from live AMFI disclosures."
            )
    except FileNotFoundError:
        parts.append("Mutual fund flows data is not available.")

    reliability = cached_reliability_summary(ADVISOR_RELIABILITY_PERIOD)
    if reliability is not None and not reliability.empty:
        reliability = reliability.sort_values("hit_rate", ascending=False)
        parts.append(
            "Backtest signal reliability (2y history; top-3 sectors vs equal-weight "
            "Nifty 50 benchmark, 6-month forward window):\n"
            + _format_table(
                reliability,
                [
                    "lookback",
                    "n_samples",
                    "hit_rate",
                    "avg_sector_return",
                    "avg_benchmark_return",
                    "avg_excess_return",
                    "median_excess_return",
                    "win_rate_positive",
                ],
                pct_cols=(
                    "hit_rate",
                    "avg_sector_return",
                    "avg_benchmark_return",
                    "avg_excess_return",
                    "median_excess_return",
                    "win_rate_positive",
                ),
            )
        )
        best = reliability.iloc[0]
        parts.append(
            f"Best lookback by hit-rate: {int(best['lookback'])} days with "
            f"{best['hit_rate']:.0%} hit rate and {best['avg_excess_return']:.2%} "
            "average excess return. Use these stats to calibrate your confidence "
            "in momentum recommendations."
        )

    try:
        regime_info = compute_market_regime(ADVISOR_REGIME_PERIOD)
        if regime_info is not None:
            latest = regime_info["latest"]
            state = "TRENDING" if latest["in_market_now"] else "FLAT / CHOP"
            parts.append(
                f"Market regime (Nifty 50, 2y): current state = {state}. "
                f"Nifty {latest['close']:,.0f}, SMA{backtest.REGIME_SMA_FAST} "
                f"{latest['sma_fast']:,.0f}, SMA{backtest.REGIME_SMA_SLOW} "
                f"{latest['sma_slow']:,.0f}, ADX {latest['adx']:.1f}. "
                f"In market {latest['pct_in_market_3m']:.0%} of the last 60 sessions."
            )
        flows_fii = load_fii_dii()
        if flows_fii is not None and not flows_fii.empty:
            latest_flow = flows_fii.sort_values("date").groupby("category").tail(1)
            lines = []
            for _, row in latest_flow.iterrows():
                lines.append(
                    f"{row['category']}: net {row['net_value_crore']:,.1f} crore"
                )
            parts.append("Latest NSE FII/DII cash flows: " + "; ".join(lines))
    except Exception:
        pass

    return "\n\n".join(parts)


def _provider_api_keys():
    keys = {}
    try:
        import streamlit as st_secrets_holder

        secrets = st_secrets_holder.secrets
        for provider, conf in ai_advisor.ADVISOR_PROVIDERS.items():
            secret_name = conf.get("secret_name")
            if not secret_name:
                continue
            value = secrets.get(secret_name)
            if value:
                keys[provider] = str(value)
    except Exception:
        pass
    return keys


def render_advisor_tab(nifty50_map: dict):
    st.subheader("AI Investment Advisor")
    st.caption(
        "Research assistant that reasons over the current dashboard data. "
        "Educational output only — not personalized investment advice."
    )

    provider = st.radio(
        "Provider",
        PROVIDER_OPTIONS,
        index=0,
        horizontal=True,
    )
    provider = provider.replace("Auto (fallback chain)", "Auto")

    extra_keys = _provider_api_keys()
    with st.expander("API keys (optional — overrides environment variables)"):
        openrouter_key = st.text_input("OpenRouter API key (sk-or-...)", type="password", key="key_openrouter")
        groq_key = st.text_input("Groq API key (gsk_...)", type="password", key="key_groq")
        zen_key = st.text_input("OpenCode Zen API key (sk-zen-...)", type="password", key="key_zen")
        if openrouter_key:
            extra_keys["OpenRouter"] = openrouter_key
        if groq_key:
            extra_keys["Groq"] = groq_key
        if zen_key:
            extra_keys["OpenCode Zen"] = zen_key

    if "advisor_messages" not in st.session_state:
        st.session_state.advisor_messages = []

    for message in st.session_state.advisor_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about sectors or stocks, e.g. 'Which sectors should I buy?'")
    if not prompt:
        return

    st.session_state.advisor_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Building context from current dashboard data..."):
        context = build_advisor_context(nifty50_map)

    messages = [
        {"role": "user", "content": f"Dashboard context:\n\n{context}\n\nQuestion: {prompt}"}
    ]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        reply = ""
        try:
            for chunk in ai_advisor.call_advisor(messages, provider=provider, extra_keys=extra_keys):
                reply += chunk
                placeholder.markdown(reply + "▌")
            placeholder.markdown(reply if reply else "_(no response text received)_")
        except Exception as exc:
            placeholder.error(f"Advisor error: {exc}")

    st.session_state.advisor_messages.append({"role": "assistant", "content": reply or ""})


def render_rotation_tab(nifty50_map: dict):
    period = st.selectbox(
        "Historical data window",
        _APP["rotation_periods"],
        index=0,
        key="rotation_period",
    )
    lookback_days = st.select_slider(
        "Momentum lookback (days)",
        options=LOOKBACK_OPTIONS,
        value=20,
        key="rotation_lookback",
    )
    show_symbol_table = st.checkbox("Show ticker-level momentum table", value=True)

    with st.spinner("Fetching price data from Yahoo Finance..."):
        tickers = list(nifty50_map.keys())
        close, failed_tickers = fetch_close_prices(tickers, period)

    if failed_tickers:
        st.warning(f"Price data not found for: {', '.join(failed_tickers)}")

    if close.empty:
        st.error("No price data was loaded. Please try a different timeframe.")
        return

    # Guard against a lookback that exceeds the available trading days.
    max_lookback = max(len(close) - 1, 1)
    if lookback_days > max_lookback:
        st.warning(
            f"Momentum lookback of {lookback_days} days exceeds the {len(close)} "
            f"available trading days; clamping to {max_lookback}."
        )
        lookback_days = max_lookback

    st.caption(
        f"Data loaded: {close.index[0].date()} → {close.index[-1].date()} "
        f"({len(close)} trading days). 'Momentum' uses the trailing {lookback_days}-day "
        "window, so it is identical for every data window. Switch the chart metric to "
        "'Return over window' to see the effect of the selected window."
    )

    momentum, sector_summary = compute_sector_rotation(close, lookback_days, nifty50_map)

    # Use the latest downloaded close as the current price (avoids per-ticker
    # network round-trips via Ticker.info).
    last_close_map = {t: (close[t].iloc[-1] if t in close.columns else None) for t in momentum["ticker"]}
    momentum["current_price"] = momentum["ticker"].map(lambda t: last_close_map.get(t))
    momentum["last_close"] = momentum["ticker"].map(lambda t: last_close_map.get(t))

    st.subheader("Sector Summary")
    sector_summary = sector_summary.reset_index()
    sector_summary["avg_momentum_pct"] = sector_summary["average_momentum"].map(format_return)
    sector_summary["median_momentum_pct"] = sector_summary["median_momentum"].map(format_return)
    sector_summary["avg_period_return_pct"] = sector_summary["average_period_return"].map(format_return)
    sector_summary["median_period_return_pct"] = sector_summary["median_period_return"].map(format_return)

    chart_metric = st.radio(
        "Sector chart metric",
        ["Momentum (trailing lookback)", "Return over window"],
        index=0,
        horizontal=True,
        key="rotation_chart_metric",
    )
    if chart_metric == "Return over window":
        fig_sector = px.bar(
            sector_summary,
            x="sector",
            y="average_period_return",
            text="avg_period_return_pct",
            title=f"Average sector return over the {period} window",
            labels={"sector": "Sector", "average_period_return": "Return over window"},
            color="average_period_return",
            color_continuous_scale=["#c0392b", "#ecf0f1", "#27ae60"],
            color_continuous_midpoint=0,
        )
        fig_sector.update_traces(textposition="outside")
        fig_sector.update_layout(yaxis_tickformat="%", coloraxis_showscale=False)
    else:
        fig_sector = px.bar(
            sector_summary,
            x="sector",
            y="average_momentum",
            text="avg_momentum_pct",
            title=f"Average {lookback_days}-day momentum by sector",
            labels={"sector": "Sector", "average_momentum": "Average momentum"},
        )
        fig_sector.update_traces(textposition="outside")
        fig_sector.update_layout(yaxis_tickformat="%")

    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(fig_sector, width='stretch')
    with right:
        st.markdown("### Top sectors")
        st.table(
            sector_summary.head(5)[
                ["sector", "symbol_count", "avg_momentum_pct", "median_momentum_pct", "avg_period_return_pct"]
            ]
        )

    # Money flow (Chaikin Money Flow) option
    show_cmf = st.checkbox("Show Chaikin Money Flow (CMF) by sector", value=False)
    if show_cmf:
        cmf_lookback = st.select_slider("CMF lookback (days)", options=LOOKBACK_OPTIONS, value=20)
        with st.spinner("Fetching OHLCV and computing CMF..."):
            ohlcv = fetch_ohlcv(list(momentum["ticker"]), period)
            cmf_vals = compute_cmf(ohlcv, cmf_lookback)
        cmf_df = pd.DataFrame(
            [
                {"ticker": t, "sector": nifty50_map.get(t, "Other"), "cmf": v}
                for t, v in cmf_vals.items()
            ]
        ).dropna(subset=["cmf"]).reset_index(drop=True)
        if cmf_df.empty:
            st.info("No CMF values could be computed for the selected lookback and data window.")
        else:
            sector_cmf = (
                cmf_df.groupby("sector").agg(average_cmf=("cmf", "mean"), symbol_count=("ticker", "count"))
                .reset_index()
                .sort_values("average_cmf", ascending=False)
            )
            sector_cmf["average_cmf_pct"] = sector_cmf["average_cmf"].map(lambda x: f"{x:.4f}")
            fig_cmf = px.bar(sector_cmf, x="sector", y="average_cmf", text="average_cmf_pct", title=f"CMF ({cmf_lookback}d) by sector")
            fig_cmf.update_traces(textposition="outside")
            left2, right2 = st.columns([2, 1])
            with left2:
                st.plotly_chart(fig_cmf, width='stretch')
            with right2:
                st.table(sector_cmf[["sector", "symbol_count", "average_cmf_pct"]].head(10))

    if show_symbol_table:
        st.subheader("Ticker Momentum")
        momentum_display = momentum.copy()
        momentum_display["momentum"] = momentum_display["momentum"].map(format_return)
        momentum_display["period_return"] = momentum_display["period_return"].map(format_return)
        momentum_display["current_price"] = momentum_display["current_price"].map(lambda x: f"{x:0.2f}" if pd.notna(x) else "")
        momentum_display = momentum_display.sort_values("rank")
        st.dataframe(
            momentum_display[
                ["ticker", "sector", "current_price", "last_close", "momentum", "period_return"]
            ].reset_index(drop=True),
            width='stretch',
        )

    st.markdown(
        "---\n"
        "**Notes:** data is sourced from Yahoo Finance via `yfinance`. "
        "This dashboard uses simple momentum signals and is intended for research and display purposes only."
    )


if __name__ == "__main__":
    main()
