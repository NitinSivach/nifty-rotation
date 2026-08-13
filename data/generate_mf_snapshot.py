"""Generate the bundled sector-wise mutual fund buy/sell snapshot.

The dashboard ships with this CSV so the Mutual Fund tab always has data to
render even when live sources are unavailable. The values are illustrative
(synthetic, deterministic) and expressed in INR crore.

Regenerate with:
    python data/generate_mf_snapshot.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from settings import SETTINGS, data_file

DATA_DIR = data_file("mf_flows").parent
OUT_PATH = DATA_DIR / "mf_sector_flows.csv"

_CFG = SETTINGS["data_refresh"]["snapshot"]
PERIODS = _CFG["periods"]
MONTHS = _CFG["months"]
BASE_NET_PER_MONTH = _CFG["base_net_per_month"]
ACTIVITY_FACTOR = _CFG["activity_factor"]
SEED = _CFG["seed"]
SECTOR_WEIGHTS = _CFG["sector_weights"]


def build_weights():
    mapping_path = data_file("nifty50_map")
    with mapping_path.open("r", encoding="utf-8-sig") as f:
        mapping = json.load(f)
    sectors = sorted({sector for sector in mapping.values()})
    weights = {sector: SECTOR_WEIGHTS.get(sector, 0.01) for sector in sectors}
    total = sum(weights.values())
    return {sector: weight / total for sector, weight in weights.items()}


def main():
    weights = build_weights()
    rng = np.random.default_rng(SEED)
    rows = []
    for period in PERIODS:
        months = MONTHS[period]
        total_net = BASE_NET_PER_MONTH * months
        total_activity = total_net * ACTIVITY_FACTOR
        for sector, weight in weights.items():
            net = total_net * weight * (1 + rng.normal(0, 0.9))
            activity = max(total_activity * weight, abs(net) * 1.05)
            buy = (activity + net) / 2
            sell = (activity - net) / 2
            rows.append(
                {
                    "sector": sector,
                    "period": period,
                    "buy_crore": round(buy),
                    "sell_crore": round(sell),
                    "net_crore": round(net),
                }
            )
    df = pd.DataFrame(rows, columns=["sector", "period", "buy_crore", "sell_crore", "net_crore"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")
    print(df.groupby("period")[["buy_crore", "sell_crore", "net_crore"]].sum().to_string())


if __name__ == "__main__":
    main()
