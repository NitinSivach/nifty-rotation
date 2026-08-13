# Nifty 50 Sector Rotation

A Python/Streamlit research dashboard for rotating across Nifty 50 sectors and
Nifty sector indices using momentum + Chaikin Money Flow (CMF) signals,
mutual-fund flow overlays, an optional market-regime (trend/ADX) cash filter,
an EMA long-downtrend breakout screener, and an AI research assistant.

**Intended use:** research/education only — not personalized investment advice.

---

## 1. Overview

The app compares momentum across Nifty 50 sectors (or Nifty sector indices)
and identifies top-performing groups, then lets you:

- Backtest a *"buy top-N sectors, rebalance periodically, equal weight"*
  strategy and compare it against an equal-weight benchmark.
- Inspect per-lookback **signal reliability** (does momentum actually predict
  6-month returns?) to calibrate confidence before trusting signals.
- Overlay **mutual-fund sector flows**, **FII/DII flows**, and a **market
  regime filter** (cash when Nifty is choppy).
- Screen for stocks that have just crossed above their EMA after a long
  below-EMA streak (multi-year downtrend breakouts).
- Rank large-cap companies by **5-year dividend yield** (average of annual
  dividends ÷ that year's average close) to find income names — the Nifty 50
  plus Nifty Next 50 members, PSUs and other blue-chip dividend payers.
- Chat with an **AI advisor** grounded in the current dashboard data.

The backtest core (`backtest.py`) is intentionally free of Streamlit imports so
it can run headless as a CLI.

---

## 2. Directory layout

```
nifty-sector-rotation/
├── app.py                  # Streamlit dashboard (UI, rendering, data glue)
├── backtest.py             # Signal construction + backtest engine (headless)
├── ai_advisor.py           # OpenAI-compatible chat client with provider fallback
├── check_tickers.py        # Utility to verify ticker health against Yahoo
├── settings.py             # Loads config.json (all tunable values)
├── config.json             # App/backtest/advisor/data configuration (no secrets)
├── requirements.txt        # Python dependencies
├── nifty50_map.json        # ticker -> sector for the Nifty 50 stock universe
├── sector_indices.json     # Nifty sector-index universe (^NSEBANK, ^CNXIT, ...)
├── dividend_universe.json  # extra large-cap tickers for the dividend tab
├── data/
│   ├── mf_sector_flows.csv # sector MF buy/sell by period (₹ crore)
│   ├── fii_dii_flows.csv   # NSE FII/DII cash flows history
│   ├── delivery_volume.csv # NSE delivery-volume report (best effort)
│   ├── refresh_mf_amfi.py  # rebuild MF flows from AMFI disclosures
│   ├── refresh_fii_dii.py  # refresh FII/DII + delivery from NSE
│   └── generate_mf_snapshot.py # deterministic synthetic MF snapshot
├── .streamlit/
│   ├── config.toml         # Streamlit theme/server settings
│   └── secrets.example.toml # template for API keys (real secrets are gitignored)
├── .github/workflows/
│   └── refresh-data.yml    # daily cron refresh of the CSV snapshots
└── tests/                  # pytest suite
    ├── test_app.py
    ├── test_backtest.py
    ├── test_ai_advisor.py
    ├── test_refresh_mf_amfi.py
    └── test_refresh_fii_dii.py
```

---

## 3. Setup & running

Create a virtualenv and install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the dashboard:

```powershell
streamlit run app.py
```

Run the backtest headless:

```powershell
python backtest.py --period 2y --lookback 20 --top-n 3
python backtest.py --universe indices --vol-adjust --regime
```

Check ticker health against Yahoo:

```powershell
python check_tickers.py
```

Run tests:

```powershell
python -m pytest
```

Refresh data (see [Data scripts](#8-data-scripts) for details):

```powershell
python data/refresh_mf_amfi.py
python data/refresh_fii_dii.py
python data/generate_mf_snapshot.py
```

---

## 4. Module: `app.py` — Streamlit dashboard

The UI shell. Renders six tabs and wires up data fetching, formatting and the
AI advisor context. All data-fetching helpers are cached with
`@st.cache_data`.

### Module-level constants

| Constant | Value | Purpose |
|---|---|---|
| `MF_PERIODS` | `["3M", "6M", "1Y", "2Y"]` | Selectable MF-flow periods |
| `YF_PERIOD_MAP` | `{"3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y"}` | MF period → Yahoo period string |
| `BACKTEST_PERIODS` | `["1y", "2y", "5y"]` | Backtest data windows |
| `EMA_PERIODS` | `["2y", "3y", "5y"]` | EMA screener data windows |
| `EMA_TIMEFRAMES` | `{"Daily": ("D", 252), "Weekly": ("W", 52), "Monthly": ("ME", 12)}` | Chart timeframe → (resample freq, periods/yr) |
| `EMA_MIN_YEARS` | `{"Any": 0, "1 year": 1, "2 years": 2, "3 years": 3, "5 years": 5}` | Min below-EMA streak filter |
| `DIVIDEND_PERIODS` | `["3y", "5y", "10y"]` | Dividend tab data windows |

### Functions

**Data loading & fetching**

- `load_nifty50_map(path=None) -> dict`
  Loads `nifty50_map.json` (default) or a given path. Returns `{ticker: sector}`.

- `load_dividend_universe(path=None) -> dict`
  Expanded dividend universe: merges `nifty50_map.json` with the extra large-cap
  tickers in `dividend_universe.json` (Nifty Next 50 members, PSUs, blue-chip
  dividend payers). Falls back to the Nifty 50 map alone if the extra file is
  missing or invalid.

- `_download_chunked(tickers, period, interval="1d", auto_adjust=True, retries=3, backoff_seconds=1.0) -> dict`
  Downloads OHLCV for tickers in chunks of 10 via `yf.download`, retrying each
  chunk with linear backoff. Returns `{ticker: DataFrame}` (raises `RuntimeError`
  if a chunk fails after all retries). Handles both MultiIndex (multi-ticker)
  and single-ticker download shapes.

- `fetch_close_prices(tickers, period) -> (DataFrame, list)` *(cached)*
  Downloads **adjusted** daily closes, returns a `DataFrame` of `Close` columns
  (duplicate columns dropped) plus a list of tickers that had no data.

- `fetch_ohlcv(tickers, period) -> dict` *(cached)*
  Downloads **unadjusted** OHLCV in chunks; returns `{ticker: DataFrame}` with
  the subset of `[Open, High, Low, Close, Volume]` present.

**Computation**

- `compute_cmf(ohlcv_dict, lookback_days) -> dict` *(cached)*
  Chaikin Money Flow per ticker over `lookback_days`. MFM = ((C-L)-(H-C))/(H-L)
  (division-by-zero rows → NaN), money flow = MFM × Volume, CMF =
  Σmoney_flow / Σvolume. Returns `ticker -> float | None`.

- `compute_sector_rotation(close, lookback_days, nifty50_map) -> (DataFrame, DataFrame)` *(cached)*
  Computes per-ticker trailing-lookback momentum and period return, maps each
  ticker to a sector (`"Other"` fallback), ranks by momentum, and aggregates a
  sector summary (average/median momentum and period return, symbol count)
  sorted by average momentum descending.

- `resample_close(close, freq) -> DataFrame`
  Resamples daily closes to the last close of each period using pandas `freq`
  conventions. `"D"/"B"` (and variants) are no-ops; `"W"`/`"ME"` resample.

- `compute_ema_breakouts(close, ema_span=20, min_below_periods=0, periods_per_year=252) -> DataFrame`
  Flags tickers whose price most recently crossed **above** their EMA after
  spending at least `min_below_periods` periods below it. Computes EMA, `above`
  bool mask and `up_cross` (above & ~was_above, with explicit bool casting to
  avoid object-dtype `~` bugs). For each ticker it walks back from the last
  cross counting the below-streak. Returns a DataFrame with: `ticker`,
  `last_close`, `ema`, `cross_date`, `below_streak_periods`, `streak_years`,
  `periods_since_cross`, `currently_above`, `pct_above_ema` (sorted by longest
  streak, filtered by `min_below_periods`).

- `format_return(value) -> str`
  Formats a fractional return as a percentage string, e.g. `0.1234 -> "12.34%"`.

- `load_mf_flows(path=None) -> DataFrame`
  Reads `data/mf_sector_flows.csv` (columns `sector, period, buy_crore,
  sell_crore, net_crore`).

- `filter_mf_flows(df, period) -> DataFrame`
  Filters the flows frame to the given period (case-insensitive string match).

- `format_crore(value) -> str`
  Formats a value as `₹X,XXX Cr`.

- `compute_ticker_period_returns(close, nifty50_map) -> DataFrame`
  For each ticker returns `ticker`, `sector`, `current_price` (last close) and
  `period_return` (last/first − 1).

- `fetch_dividend_history(tickers, period) -> dict` *(cached, 24h TTL)*
  Fetches split-adjusted daily `Close` plus the `Dividends` action column per
  ticker by downloading with `yf.download(..., actions=True)`.

- `compute_dividend_stats(dividend_history, nifty50_map) -> DataFrame` *(cached)*
  Per-ticker dividend summary over the window. For each ticker it sums the
  annual dividends, divides by that year's average close for a per-year yield,
  and reports the window average as `avg_dividend_yield` (%). Non-payers are
  excluded. Columns: `ticker`, `sector`, `current_price`,
  `avg_annual_dividend`, `avg_dividend_yield`, `total_dividend`,
  `years_count`; sorted by yield descending.

**Backtest / regime glue**

- `cached_reliability_summary(period) -> DataFrame | None` *(cached)*
  Backtest signal-reliability stats (via `backtest.signal_reliability` over
  `backtest.LOOKBACKS`) used to ground the AI advisor. Returns `None` on failure.

- `compute_backtest_analysis(universe, period, momentum_lookback, cmf_lookback,
  momentum_weight, cmf_weight, top_n, use_vol_adjust=False, apply_regime=False,
  regime_adx=False) -> (reliability_df, result_dict, ranking_df) | None` *(cached)*
  Runs reliability, backtest, and current composite ranking for either the
  stock universe (`"stocks"`) or sector-index universe (`"indices"`). If
  `apply_regime` is set it fetches the regime series first, then passes it to
  `run_backtest`. Builds the ranking frame from the latest sector momentum
  (raw or vol-adjusted), sector CMF and composite score, optionally appending
  the 3M MF-flow net (`mf_net_crore`) column.

- `compute_regime_series(period, require_adx=False) -> Series` *(cached)*
  Market regime bool series (True = trending) computed on the Nifty 50 index
  (`backtest.REGIME_INDEX` = `^NSEI`) via `backtest.compute_regime`.

- `compute_market_regime(period) -> dict | None` *(cached)*
  Returns regime series, Nifty 50 close, and latest metrics: `close`,
  `sma_fast`, `sma_slow`, `adx`, `in_market_now`, `pct_in_market_3m`.

- `load_fii_dii() -> DataFrame | None` *(cached)*
  Reads `data/fii_dii_flows.csv` or returns `None`.

- `load_delivery_volume() -> DataFrame | None` *(cached)*
  Reads `data/delivery_volume.csv` or returns `None`.

**Rendering**

- `render_mf_tab(nifty50_map)` — Mutual Fund Flows tab. Selects period, loads
  flows, shows a net-flow horizontal bar chart and sector table, then fetches
  price data and shows ticker-level returns filtered by sector.

- `render_backtest_tab(nifty50_map)` — Backtest & Composite tab. Parameter
  sliders (momentum/CMF lookback, top-N, weights), runs
  `compute_backtest_analysis("stocks", ...)`, displays strategy metrics
  (total return, benchmark, hit rate, avg excess, max drawdown, Sharpe),
  cumulative-growth chart, reliability table, current composite ranking, and
  per-rebalance detail.

- `render_expansion_tab(nifty50_map)` — Regime & Expanded Universe tab. Shows
  the Nifty 50 regime state and SMA/ADX chart, FII/DII flows and delivery
  volume, then runs a backtest over the chosen universe with optional
  vol-adjust / regime / ADX options and compares against an "always in market"
  baseline.

- `render_ema_tab(nifty50_map)` — EMA Breakout tab. Selects data window,
  timeframe (Daily/Weekly/Monthly), EMA span and minimum below-streak years,
  runs `compute_ema_breakouts`, and renders the screener table.

- `render_dividend_tab(nifty50_map)` — Dividend Yield tab. Universe radio
  (Nifty 50 + large caps by default, or Nifty 50 only), data window
  (3y/5y/10y), fetches dividend history, computes per-ticker stats, shows
  headline yield metrics and a top-15 bar chart, then a sector-filterable
  table sorted by average dividend yield.

- `render_rotation_tab(nifty50_map)` — Nifty Sector Rotation tab. Selects
  window and lookback, computes sector rotation, shows momentum/period-return
  bar charts, top-sector table, optional CMF-by-sector chart, and the ticker
  momentum table. Clamps lookback to available trading days.

- `render_advisor_tab(nifty50_map)` — AI Advisor tab. Chat UI: provider
  selector (Auto / OpenRouter / OpenCode Zen / Groq), optional API-key inputs,
  builds dashboard context and streams assistant replies.

- `_format_table(df, columns, pct_cols=()) -> str`
  Renders a DataFrame subset as a plain-text table with the given percentage
  columns formatted via `format_return`.

- `build_advisor_context(nifty50_map) -> str`
  Assembles a plain-text snapshot of current dashboard state (active filters,
  sector/ticker momentum, MF flows, reliability stats, regime and FII/DII
  values) that is fed to the AI advisor.

- `_provider_api_keys() -> dict`
  Reads API keys from Streamlit secrets for OpenRouter / OpenCode Zen / Groq.

- `main()`
  Entry point: page config, loads the Nifty 50 map, optional ticker→sector CSV
  override upload, creates the seven tabs (rotation, MF flows, backtest,
  regime/expanded, EMA breakout, dividend yield, AI advisor) and dispatches to
  their renderers.

---

## 5. Module: `backtest.py` — signals & backtest engine

Headless module (no Streamlit imports). Core of the strategy: signal
construction, composite ranking, backtest simulation, regime filter, and
signal-reliability stats.

### Module-level constants

| Constant | Value | Purpose |
|---|---|---|
| `TRADING_DAYS_YEAR` | `252` | Annualization factor |
| `HOLD_DAYS_6M` | `126` | Default forward hold window for reliability |
| `DEFAULT_TOP_N` | `3` | Default number of top sectors |
| `DEFAULT_REBALANCE_DAYS` | `20` | Default rebalance cadence |
| `LOOKBACKS` | `(5, 10, 20, 40, 60, 120)` | Lookbacks tested for reliability |
| `YF_PERIODS` | `("1y", "2y", "5y")` | CLI period choices |
| `REGIME_INDEX` | `"^NSEI"` | Nifty 50 index for the regime filter |
| `REGIME_SMA_FAST` | `50` | Fast SMA for strict regime |
| `REGIME_SMA_SLOW` | `200` | Slow SMA (primary regime gate) |
| `REGIME_ADX_PERIOD` | `14` | ADX period |
| `REGIME_ADX_THRESHOLD` | `20.0` | ADX confirmation threshold |

`__all__` exposes: `compute_adx`, `compute_regime`, `composite_scores`,
`compute_metrics`, `current_ranking`, `download_ohlcv_chunked`,
`fetch_backtest_data`, `load_nifty50_map`, `load_sector_indices`, `rolling_cmf`,
`run_backtest`, `sector_cmf`, `sector_daily_returns`, `sector_momentum`,
`sector_vol_adj_momentum`, `signal_reliability`, `volatility_adjusted_momentum`.

### Data loading

- `load_nifty50_map(path=None) -> dict`
  Loads the Nifty 50 stock `{ticker: sector}` mapping (`nifty50_map.json`).

- `load_sector_indices(path=None) -> dict`
  Loads the Nifty sector-index universe (`sector_indices.json`) as
  `{index_ticker: sector}`.

- `download_ohlcv_chunked(tickers, period, interval="1d", retries=3,
  backoff_seconds=1.0, chunk_size=10) -> dict`
  Downloads **unadjusted** OHLCV in chunks of `chunk_size` with retry/backoff.
  Returns `{ticker: DataFrame[Open, High, Low, Close, Volume]}`.

- `fetch_backtest_data(tickers, period="2y") -> (DataFrame, dict)`
  Returns `(close, ohlcv)` aligned on a common trading calendar; every OHLCV
  frame is reindexed to the close frame's index.

### Signal construction

- `rolling_cmf(df, window) -> Series`
  Chaikin Money Flow over a trailing `window`, one value per row. MFM =
  ((C−L)−(H−C))/(H−L); money flow = MFM × Volume; CMF = rolling Σflow / Σvol.
  Near-zero values are collapsed to `0.0` so ties rank correctly.

- `_sector_groups(nifty50_map, tickers) -> dict`
  Groups tickers into `{sector: [tickers]}` with an `"Other"` fallback bucket.

- `sector_daily_returns(close, nifty50_map) -> DataFrame`
  Daily equal-weight sector index returns (mean of member ticker returns).

- `sector_momentum(close, nifty50_map, lookback) -> DataFrame`
  Sector-average momentum: per-ticker `pct_change(lookback)`, averaged within
  each sector.

- `sector_cmf(ohlcv, nifty50_map, window) -> DataFrame`
  Sector-average CMF: per-ticker `rolling_cmf` averaged within each sector.

- `volatility_adjusted_momentum(close, lookback) -> DataFrame`
  Per-ticker momentum divided by the rolling std of daily returns — a
  risk-normalized "quality of trend" momentum.

- `sector_vol_adj_momentum(close, nifty50_map, lookback) -> DataFrame`
  Sector average of volatility-adjusted momentum.

- `compute_adx(high, low, close, period=REGIME_ADX_PERIOD) -> Series`
  Wilder's ADX: computes +DM/−DM, true range, ATR, ±DI and DX using EMA
  smoothing with `alpha = 1/period`.

- `compute_regime(index_close, index_ohlcv=None, sma_fast=50, sma_slow=200,
  adx_period=14, adx_threshold=20.0, require_adx=True) -> Series`
  Market regime bool series: primary gate is close above the slow SMA. When
  `require_adx`, additionally requires close above the fast SMA and ADX ≥
  threshold (using `index_ohlcv` if provided). False during SMA warm-up.
  Aligned to `index_close`.

- `_mf_percentiles(mf_flows, sectors) -> dict`
  Maps each sector to its percentile rank (0..1) of net MF flow.

- `composite_scores(close, ohlcv, nifty50_map, momentum_lookback=20,
  cmf_lookback=20, momentum_weight=0.6, cmf_weight=0.4, mf_flows=None,
  mf_weight=0.2, use_vol_adjust=False) -> DataFrame`
  Ranked sector composite score over time: cross-sectional percentile ranks of
  momentum (raw or vol-adjusted) and CMF, weighted combined, plus an optional
  constant MF-flow overlay (only meaningful for the current snapshot since MF
  flows aren't available historically).

- `current_ranking(close, ohlcv, nifty50_map, momentum_lookback=20,
  cmf_lookback=20, momentum_weight=0.6, cmf_weight=0.4, mf_flows=None,
  mf_weight=0.2, use_vol_adjust=False) -> Series`
  Latest composite score per sector, sorted best-first.

### Backtest

- `_first_valid_index(frame)` — index of the first row with any non-NaN value.

- `_top_indices(vals, top_n) -> list`
  Indices of the top-`n` largest non-NaN values in a 1-D array.

- `compute_metrics(value, benchmark_value, period_stats) -> dict`
  Performance metrics over the traded window: `total_return`,
  `benchmark_total_return`, `cagr`, `benchmark_cagr`, `sharpe`,
  `benchmark_sharpe`, `max_drawdown`, `benchmark_max_drawdown`, `hit_rate`,
  `avg_excess_return`, `n_periods`. Empty if insufficient data.

- `run_backtest(close, ohlcv, nifty50_map, momentum_lookback=20,
  cmf_lookback=20, momentum_weight=0.6, cmf_weight=0.4, top_n=3,
  rebalance_days=20, use_vol_adjust=False, regime=None) -> dict`
  Simulates "buy top-N sectors, rebalance periodically, equal weight". Every
  `rebalance_days` it picks the top-N sectors by composite score; when `regime`
  is False on a rebalance day the portfolio sits in cash. Weights are shifted by
  one row so trades settle the day after the signal (no lookahead bias).
  Returns `{value, benchmark_value, portfolio_returns, period_stats, metrics,
  weights}`.

### Signal reliability

- `signal_reliability(close, ohlcv, nifty50_map, lookbacks=LOOKBACKS,
  hold_days=126, top_n=3, step_days=20, momentum_weight=0.6, cmf_weight=0.4,
  use_vol_adjust=False) -> DataFrame`
  For each lookback, at every `step_days`-th day the top-N sectors are held for
  `hold_days` and compared with the equal-weight benchmark. Returns one row per
  lookback: `lookback`, `n_samples`, `hit_rate`, `avg_sector_return`,
  `avg_benchmark_return`, `avg_excess_return`, `median_excess_return`,
  `win_rate_positive`.

### CLI

- `main(argv=None) -> None`
  `argparse` CLI: `--period`, `--top-n`, `--rebalance-days`, `--lookback`,
  `--universe {stocks|indices}`, `--vol-adjust`, `--regime`, `--regime-adx`.
  Fetches data, optionally computes the regime filter, prints the reliability
  table and backtest metrics.

---

## 6. Module: `ai_advisor.py` — AI chat client

OpenAI-compatible streaming chat client with provider fallback for the AI
advisor tab.

### Module-level data

- `ADVISOR_PROVIDERS` — dict of provider configs, each with `base_url`,
  `model`, `env_key` and `needs_key`:
  - **OpenRouter** — `deepseek/deepseek-v4-flash:free`, needs key.
  - **OpenCode Zen** — `deepseek-v4-flash-free`, no key required.
  - **Groq** — `llama-3.3-70b-versatile`, needs key.
- `AUTO_ORDER` — `["OpenRouter", "OpenCode Zen", "Groq"]`, the fallback chain.
- `SYSTEM_PROMPT` — instructs the model to act as a research assistant embedded
  in the dashboard: cite the numbers given, calibrate confidence using
  reliability hit-rates, never fabricate figures, flag synthetic data, and
  output educational (non-SEBI-advisory) content.

### Functions

- `resolve_api_key(provider, extra_keys=None) -> str`
  Returns the API key for a provider from `extra_keys` first, then the
  environment variable named by `env_key`, else `""`.

- `stream_chat(base_url, api_key, model, messages, timeout=90) -> generator`
  Streams assistant text chunks from an OpenAI-compatible
  `/chat/completions` SSE endpoint. Forces UTF-8 decoding (requests would
  otherwise fall back to ISO-8859-1 and mangle unicode), raises `RuntimeError`
  on HTTP ≥ 400, and yields `delta.content` per SSE `data:` line until
  `[DONE]`.

- `call_advisor(messages, provider="Auto", extra_keys=None,
  system_prompt=SYSTEM_PROMPT) -> generator`
  Prepends the system prompt and yields assistant text, trying providers in
  `AUTO_ORDER` (or just the chosen one) until one succeeds. Skips providers
  missing a required key and aggregates all errors into a single
  `RuntimeError` if every provider fails.

---

## 7. Module: `check_tickers.py` — ticker health check

Utility that verifies every ticker in `nifty50_map.json` against Yahoo.

- `load_nifty50_map(path=None) -> dict`
  Loads the Nifty 50 mapping (default `nifty50_map.json`).

- `check_ticker(sym) -> (int, str | None)`
  Fetches one month of history via `yf.Ticker(sym).history(period="1mo")`.
  Returns `(row_count, error)`; error is `None` on success.

- `main()`
  Iterates all mapped tickers, prints `OK (N rows)` or the failure detail, and
  summarizes any missing/stale tickers.

---

## 8. Data scripts

### `data/refresh_mf_amfi.py` — rebuild MF sector flows from AMFI

Downloads monthly portfolio holdings for 30+ curated equity schemes from the
AMFI portal for the latest month and for 3/6/12/24 months ago, diffs holdings
across each period to compute sector-wise buy/sell, and writes
`data/mf_sector_flows.csv` in ₹ crore.

Module constants: `ROOT`, `OUT_PATH`, `PERIODS` (`{"3M": 3, "6M": 6, "1Y": 12,
"2Y": 24}`), `AMFI_PORTFOLIO_URL`, `HEADERS`, `CURATED_FUNDS` (scheme code +
name list), `SECTOR_MAP` (raw AMFI sector → canonical sector).

Functions:

- `shift_month(year, month, delta) -> (int, int)` — month arithmetic for
  "delta months ago".
- `_fetch(url, params, retries=3, backoff_seconds=1.0) -> str | None` — GET
  with retry/backoff; returns text or `None`.
- `_find_header_row(df) -> int | None` — finds the row that looks like a
  holdings header (contains "company"/"instrument" and "market value").
- `_to_numeric(series)` — strips commas/percent and coerces to numeric.
- `_normalize_holdings(text) -> DataFrame | None` — parses an AMFI disclosure
  response (HTML table or CSV) into `{company, sector, quantity,
  market_value_lakhs, pct_aum}`; returns `None` if no holdings present.
- `fetch_scheme_portfolio(code, year, month) -> DataFrame | None` — fetches and
  normalizes a scheme/month portfolio.
- `compute_period_flows(snapshots) -> list[dict]` — given `{(year, month):
  holdings}` computes per-period buy/sell/net by sector (in ₹ crore), mapping
  sectors through `SECTOR_MAP`.
- `main()` — iterates curated funds × months, aggregates snapshots, writes CSV,
  prints per-period totals.

### `data/refresh_fii_dii.py` — refresh FII/DII + delivery from NSE

Pulls daily NSE cash-market FII/FPI and DII figures with a browser-like
session (cookies) and merges them into `data/fii_dii_flows.csv`
(`date, category, buy_value_crore, sell_value_crore, net_value_crore`).
Also attempts the daily delivery-position report
(`data/delivery_volume.csv`) which is frequently WAF-blocked (HTTP 503) and
degrades gracefully.

Functions:

- `nse_session() -> requests.Session` — session with browser headers that hits
  the NSE homepage first so cookies are set.
- `parse_fii_dii_payload(payload) -> DataFrame` — normalizes the raw NSE
  FII/DII list into `{date, category, buy/sell/net}` crore; raises `ValueError`
  on a bad shape.
- `fetch_fii_dii(session) -> DataFrame` — GETs the FII/DII API and parses it.
- `merge_fii_dii(new_df, out_path) -> DataFrame` — appends new rows,
  overwriting any existing `(date, category)` pair (keep="last").
- `fetch_delivery(session, target) -> DataFrame` — delivery stats for a date:
  `date, symbol, series, security, quantity, deliverable_quantity, delivery_pct`.
- `last_weekday() -> date` — most recent non-weekend day.
- `main() -> None` — orchestrates: session, FII/DII fetch+merge, best-effort
  delivery fetch, printing latest figures.

### `data/generate_mf_snapshot.py` — deterministic synthetic MF snapshot

Generates the bundled `data/mf_sector_flows.csv` so the Mutual Fund tab always
renders even without live AMFI data. Values are synthetic and deterministic
(seeded RNG).

Module constants: `PERIODS`, `MONTHS`, `BASE_NET_PER_MONTH` (`30000`),
`ACTIVITY_FACTOR` (`1.6`), `SEED` (`20260501`), `SECTOR_WEIGHTS`.

Functions:

- `build_weights() -> dict` — sector weights derived from `nifty50_map.json`
  sectors blended with `SECTOR_WEIGHTS`, normalized to sum to 1.
- `main()` — for each period and sector, generates net/activity/buy/sell
  figures (₹ crore) and writes the CSV.

---

## 9. Data files

| File | Source | Columns |
|---|---|---|
| `nifty50_map.json` | static | `{ticker: sector}` (Nifty 50 members, e.g. `RELIANCE.NS: Energy`) |
| `sector_indices.json` | static | `{index_ticker: sector}` (`^NSEBANK`, `^CNXIT`, ...) |
| `dividend_universe.json` | static | extra `{ticker: sector}` for the dividend tab (Next 50, PSUs, blue chips) |
| `data/mf_sector_flows.csv` | `refresh_mf_amfi.py` or `generate_mf_snapshot.py` | `sector, period, buy_crore, sell_crore, net_crore` |
| `data/fii_dii_flows.csv` | `refresh_fii_dii.py` | `date, category, buy_value_crore, sell_value_crore, net_value_crore` |
| `data/delivery_volume.csv` | `refresh_fii_dii.py` (best effort) | `date, symbol, series, security, quantity, deliverable_quantity, delivery_pct` |

---

## 10. Testing

Tests live in `tests/` and run with `python -m pytest`. They cover:

- **`test_backtest.py`** — CMF, composite scores (+ MF overlay, + vol-adjust),
  backtest vs benchmark, no-lookahead, regime gating, reliability stats,
  metrics, ADX/regime helpers, top-N/momentum edge cases.
- **`test_app.py`** — formatting, sector rotation ranking, CMF computation,
  MF flow loading/filtering, ticker returns, resampling, EMA breakout detection,
  dividend-yield stats.
- **`test_ai_advisor.py`** — SSE parsing, UTF-8 preservation, HTTP error
  handling, provider fallback, API-key resolution (uses a `FakeResponse` class).
- **`test_refresh_mf_amfi.py`** — month shifting, holdings parsing, period-flow
  aggregation.
- **`test_refresh_fii_dii.py`** — FII/DII payload parsing and CSV merging.

---

## 11. Notes & disclaimers

- Price data comes from Yahoo Finance via `yfinance`; MF flows are either
  live AMFI disclosures or the bundled synthetic snapshot; FII/DII comes from
  NSE APIs.
- The bundled MF snapshot is **illustrative/synthetic** unless regenerated from
  live AMFI data; the AI advisor is told to flag it as such.
- NSE endpoints are sometimes WAF-blocked; the scripts and dashboard degrade
  gracefully.
- This is a research tool. Signals are simple (momentum + CMF) and intended
  for display/research only, not personalized investment advice.