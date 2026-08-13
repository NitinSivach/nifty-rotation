# Nifty 50 Sector Rotation

A Python/Streamlit research dashboard for rotating across Nifty 50 sectors and
Nifty sector indices using momentum + Chaikin Money Flow (CMF), mutual-fund
flow overlays, an optional market-regime (trend/ADX) cash filter, an EMA
long-downtrend breakout screener, and an AI research assistant.

**Intended use:** research/education only — not personalized investment advice.

See [PROJECT.md](PROJECT.md) for the full project write-up.

## Features

- Sector momentum & CMF ranking (Nifty 50 stocks by sector, or Nifty sector indices)
- Backtest: *"buy top-N sectors, rebalance periodically, equal weight"* vs benchmark
- Per-lookback signal-reliability table (does momentum predict 6-month returns?)
- Mutual-fund sector flows, FII/DII flows, market-regime cash filter
- EMA long-downtrend breakout screener (daily/weekly/monthly)
- Dividend-yield screener for large-cap income names
- AI research assistant grounded in live dashboard data

## Local setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Backtest headless (no Streamlit needed):

```powershell
python backtest.py --period 2y --lookback 20 --top-n 3
```

## Configuration

All tunable values live in `config.json` (loaded by `settings.py`):

- `app` — page title/icon, tabs, period options, lookbacks, weights, download/cache settings
- `backtest` — trading days, hold days, lookbacks, regime filter parameters, download retries
- `advisor` — AI provider list/order, base URLs, models, system prompt, stream params
- `data_refresh` — URLs, headers, funds and sector maps used by the refresh scripts
- `data_files` — paths to the JSON/CSV data files

`config.json` contains **no secrets** and is safe to commit. For local
experiments, copy it to `config.local.json` and tweak values there —
`settings.py` auto-merges it over `config.json` (and it's gitignored).

## API keys (kept secret)

AI provider keys are **never stored in the repo or in config.json**. They are
read from environment variables (`OPENROUTER_API_KEY`, `GROQ_API_KEY`,
`OPENCODE_ZEN_API_KEY`) or from Streamlit secrets.

Local dev: copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`
and fill in your keys. `.streamlit/secrets.toml` is gitignored.

```toml
openrouter_api_key = "sk-or-..."
groq_api_key = "gsk_..."
opencode_zen_api_key = "sk-zen-..."
```

## Deploying to Streamlit Cloud

1. Push this repository to GitHub.
2. In Streamlit Cloud, create a new app pointing at the repo, with **Main file
   path** = `app.py`.
3. Add your API keys under **Settings → Secrets** (same TOML format as above).
4. Deploy.

The dashboard's price data (momentum, CMF, backtests, regime, EMA, dividends)
is fetched live from Yahoo Finance on every app run and cached for 24h, so it
stays current without any scheduled job.

### Daily CSV refresh (FII/DII, MF flows)

The committed CSVs under `data/` (`fii_dii_flows.csv`, `mf_sector_flows.csv`)
don't self-update. A GitHub Actions cron workflow (`.github/workflows/
refresh-data.yml`) runs the refresh scripts daily and commits any changes back
to the repo; Streamlit Cloud redeploys automatically on new commits.

Notes:

- AMFI disclosures are monthly, so `refresh_mf_amfi.py` usually finds nothing
  new except around month-end.
- NSE's delivery-position endpoint is frequently WAF-blocked; the script
  degrades gracefully and the dashboard shows "unavailable" in that case.

## Tests

```powershell
python -m pytest -q
```

## License

MIT — see [LICENSE](LICENSE).