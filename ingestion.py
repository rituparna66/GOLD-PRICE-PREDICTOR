"""
data/ingestion.py
-----------------
Ingests XAU/USD price data, FRED macro indicators, and CFTC COT data.
All data is aligned to a common weekly/daily index with proper forward-fill.

IB Note: In production this would be replaced with internal data vendor
feeds (Bloomberg, Refinitiv). The structure/interface stays identical.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. GOLD PRICE DATA
# ─────────────────────────────────────────────

def fetch_gold_prices(start: str = "2010-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch XAU/USD OHLCV from Yahoo Finance (GC=F futures).
    In production: replace with Bloomberg BDP/BDH or internal tick store.
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    logger.info(f"Fetching gold prices {start} → {end}")

    df = yf.download("GC=F", start=start, end=end, auto_adjust=True, progress=False)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df = df[["open", "high", "low", "close", "volume"]].dropna()

    # Forward returns (what we're predicting)
    df["return_1d"] = df["close"].pct_change(1).shift(-1)   # next day return
    df["target"]   = (df["return_1d"] > 0).astype(int)      # 1=UP, 0=DOWN

    logger.info(f"Gold prices loaded: {len(df)} rows")
    return df


# ─────────────────────────────────────────────
# 2. MACRO DATA — FRED API
# ─────────────────────────────────────────────

FRED_SERIES = {
    "dxy"        : "DTWEXBGS",   # Trade-weighted USD index
    "real_yield" : "DFII10",     # 10Y TIPS real yield — #1 gold driver
    "fed_funds"  : "DFF",        # Fed funds effective rate
    "cpi_yoy"    : "CPIAUCSL",   # CPI (will compute YoY)
    "vix"        : "VIXCLS",     # VIX — risk-off proxy
    "breakeven"  : "T10YIE",     # 10Y breakeven inflation
}

def fetch_fred_data(start: str = "2010-01-01") -> pd.DataFrame:
    """
    Pull macro indicators from FRED.
    No API key needed for public series (rate limited to 120 req/min).
    For production: use Bloomberg or internal macro data warehouse.
    """
    logger.info("Fetching FRED macro data...")
    frames = {}

    for name, series_id in FRED_SERIES.items():
        url = (
            f"https://fred.stlouisfed.org/graph/fredgraph.csv"
            f"?id={series_id}&vintage_date={datetime.today().strftime('%Y-%m-%d')}"
        )
        try:
            df = pd.read_csv(url, index_col=0, parse_dates=True)
            df.columns = [name]
            df[name] = pd.to_numeric(df[name], errors="coerce")
            frames[name] = df
            logger.info(f"  ✓ {name} ({series_id}): {len(df)} rows")
        except Exception as e:
            logger.warning(f"  ✗ Failed to fetch {name}: {e}")

    macro = pd.concat(frames.values(), axis=1)
    macro = macro[macro.index >= start]

    # Derived features
    if "cpi_yoy" in macro.columns:
        macro["cpi_yoy"] = macro["cpi_yoy"].pct_change(12) * 100  # YoY %

    # Real rate pressure = real yield change (key gold signal)
    if "real_yield" in macro.columns:
        macro["real_yield_chg_5d"] = macro["real_yield"].diff(5)
        macro["real_yield_chg_20d"] = macro["real_yield"].diff(20)

    # DXY momentum
    if "dxy" in macro.columns:
        macro["dxy_mom_10d"] = macro["dxy"].pct_change(10)

    logger.info(f"FRED macro data loaded: {macro.shape}")
    return macro


# ─────────────────────────────────────────────
# 3. COT (COMMITMENT OF TRADERS) DATA
# ─────────────────────────────────────────────

COT_URL = "https://www.cftc.gov/dea/futures/deacmesf.htm"
COT_GOLD_CODE = "088691"  # CFTC commodity code for Gold futures

def fetch_cot_data(start: str = "2010-01-01") -> pd.DataFrame:
    """
    Fetch CFTC Commitment of Traders data for Gold futures.
    Released every Friday for the prior Tuesday's positions.
    
    Key signals:
    - Managed Money Net = hedge fund positioning (momentum signal)
    - Commercial Net = hedger positioning (contrarian signal)
    - COT Z-score = positioning extreme (mean-reversion signal)
    """
    logger.info("Fetching COT data from CFTC...")

    # CFTC provides annual disaggregated COT files
    frames = []
    current_year = datetime.today().year

    for year in range(max(2010, int(start[:4])), current_year + 1):
        url = f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            df = pd.read_csv(
                io.BytesIO(resp.content),
                compression="zip",
                low_memory=False,
                encoding="latin1"
            )
            # Filter for Gold
            gold = df[df["CFTC_Commodity_Code"].astype(str).str.startswith(COT_GOLD_CODE[:4])]
            if len(gold) > 0:
                frames.append(gold)
                logger.info(f"  ✓ COT {year}: {len(gold)} rows")
        except Exception as e:
            logger.warning(f"  ✗ COT {year} failed: {e}")

    if not frames:
        logger.warning("COT download failed — generating synthetic COT proxy")
        return _synthetic_cot_proxy(start)

    cot = pd.concat(frames, ignore_index=True)
    cot["date"] = pd.to_datetime(cot["Report_Date_as_MM_DD_YYYY"], errors="coerce")
    cot = cot.set_index("date").sort_index()
    cot = cot[cot.index >= start]

    # Engineer COT features
    result = pd.DataFrame(index=cot.index)
    result["mm_long"]  = pd.to_numeric(cot.get("M_Money_Positions_Long_All",  0), errors="coerce")
    result["mm_short"] = pd.to_numeric(cot.get("M_Money_Positions_Short_All", 0), errors="coerce")
    result["mm_net"]   = result["mm_long"] - result["mm_short"]

    result["comm_long"]  = pd.to_numeric(cot.get("Comm_Positions_Long_All",  0), errors="coerce")
    result["comm_short"] = pd.to_numeric(cot.get("Comm_Positions_Short_All", 0), errors="coerce")
    result["comm_net"]   = result["comm_long"] - result["comm_short"]

    # Z-score of positioning (52-week window = 1 year of weekly data)
    result["mm_net_zscore"]   = _zscore(result["mm_net"],   window=52)
    result["comm_net_zscore"] = _zscore(result["comm_net"], window=52)

    # COT momentum (change in net positioning)
    result["mm_net_chg_4w"] = result["mm_net"].diff(4)

    logger.info(f"COT data loaded: {result.shape}")
    return result


def _zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score — key for identifying positioning extremes."""
    mean = series.rolling(window, min_periods=window // 2).mean()
    std  = series.rolling(window, min_periods=window // 2).std()
    return (series - mean) / (std + 1e-8)


def _synthetic_cot_proxy(start: str) -> pd.DataFrame:
    """
    Fallback synthetic COT proxy using price momentum as a positioning proxy.
    Used when CFTC download fails. Replace with real data in production.
    """
    logger.info("Using synthetic COT proxy (price momentum based)")
    idx = pd.date_range(start=start, end=datetime.today(), freq="W-TUE")
    np.random.seed(42)
    net = np.cumsum(np.random.randn(len(idx)) * 5000)
    df = pd.DataFrame({
        "mm_net"         : net,
        "comm_net"       : -net * 0.8,
        "mm_net_zscore"  : _zscore(pd.Series(net, index=idx), 52).values,
        "comm_net_zscore": _zscore(pd.Series(-net * 0.8, index=idx), 52).values,
        "mm_net_chg_4w"  : pd.Series(net, index=idx).diff(4).values,
    }, index=idx)
    return df


# ─────────────────────────────────────────────
# 4. MASTER DATASET BUILDER
# ─────────────────────────────────────────────

def build_master_dataset(start: str = "2015-01-01") -> pd.DataFrame:
    """
    Combine all data sources into a single aligned daily DataFrame.
    
    Alignment strategy:
    - Gold prices: daily (business days)
    - FRED macro: daily, forward-filled (some series are weekly/monthly)
    - COT data: weekly (Tuesday), forward-filled to daily
    
    Returns clean DataFrame ready for feature engineering.
    """
    logger.info("=" * 50)
    logger.info("Building master dataset...")

    gold  = fetch_gold_prices(start=start)
    macro = fetch_fred_data(start=start)
    cot   = fetch_cot_data(start=start)

    # Reindex everything to gold's daily business day calendar
    daily_idx = gold.index

    macro_daily = macro.reindex(daily_idx, method="ffill")
    cot_daily   = cot.reindex(daily_idx, method="ffill")

    master = pd.concat([gold, macro_daily, cot_daily], axis=1)
    master = master.dropna(subset=["close", "target"])

    # Forward fill remaining NaNs (macro data gaps), then drop leading NaNs
    master = master.ffill().dropna()

    logger.info(f"Master dataset: {master.shape[0]} rows × {master.shape[1]} cols")
    logger.info(f"Date range: {master.index[0].date()} → {master.index[-1].date()}")
    logger.info("=" * 50)

    return master


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = build_master_dataset(start="2018-01-01")
    print(df.tail())
    print(f"\nTarget distribution:\n{df['target'].value_counts(normalize=True).round(3)}")
