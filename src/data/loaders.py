from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.identifiers import ensure_ts_code, normalize_stock_code

# Explicit Chinese->English mappings for robustness and maintainability.
DAILY_COL_MAP = {
    "日期": "date",
    "股票代码": "stock_code",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "price_change",
    "换手率": "turnover",
}

DELIST_COL_MAP = {
    "TS代码": "ts_code",
    "股票代码": "stock_code",
    "退市日期": "delist_date",
}

STOCK_LIST_COL_MAP = {
    "TS代码": "ts_code",
    "股票代码": "stock_code",
    "所属行业": "industry",
    "上市日期": "ipo_date",
}


def _read_csv(path: Path) -> pd.DataFrame:
    """Read CSV with tolerant encoding fallback for common CN datasets."""
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _parse_cn_date(series: pd.Series) -> pd.Series:
    """Parse mixed date formats commonly seen in CN equity datasets."""
    # First attempt explicit YYYYMMDD parsing for int-like columns.
    s = series.astype("string").str.strip()
    parsed = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    # Fallback to pandas general parser for values like '1991-04-03'.
    fallback = pd.to_datetime(s, errors="coerce")
    return parsed.fillna(fallback)


def _iter_daily_files(daily_path: Path) -> Iterable[Path]:
    if daily_path.is_file():
        yield daily_path
        return
    if daily_path.is_dir():
        for p in sorted(daily_path.glob("*.csv")):
            yield p
        return
    raise FileNotFoundError(f"Daily data path not found: {daily_path}")


def load_daily_data(daily_path: Path) -> pd.DataFrame:
    """Load and clean daily OHLCV-like data.

    Supports either:
    - a single large file; or
    - a directory containing per-stock files.

    Key anti-look-ahead assumption:
    - `date` is trade date; later pipeline must shift factor->position before return calc.
    """
    frames: list[pd.DataFrame] = []
    for file_path in _iter_daily_files(daily_path):
        raw = _read_csv(file_path)
        renamed = raw.rename(columns=DAILY_COL_MAP)
        required = ["date", "stock_code", "close"]
        missing = [c for c in required if c not in renamed.columns]
        if missing:
            raise ValueError(f"{file_path} missing required columns: {missing}")

        # Keep only known columns if present; this makes schema explicit and stable.
        keep_cols = [v for v in DAILY_COL_MAP.values() if v in renamed.columns]
        cleaned = renamed[keep_cols].copy()

        cleaned["date"] = _parse_cn_date(cleaned["date"])
        cleaned["stock_code"] = normalize_stock_code(cleaned["stock_code"])

        numeric_cols = [
            c
            for c in [
                "open",
                "close",
                "high",
                "low",
                "volume",
                "amount",
                "amplitude",
                "pct_change",
                "price_change",
                "turnover",
            ]
            if c in cleaned.columns
        ]
        for c in numeric_cols:
            cleaned[c] = pd.to_numeric(cleaned[c], errors="coerce")

        frames.append(cleaned.dropna(subset=["date", "stock_code"]))

    if not frames:
        raise ValueError(f"No CSV files found under daily path: {daily_path}")

    out = pd.concat(frames, ignore_index=True)
    out = ensure_ts_code(out)
    out = out.sort_values(["date", "stock_code"]).drop_duplicates(["date", "stock_code"])
    return out.reset_index(drop=True)


def load_delist_data(delist_path: Path) -> pd.DataFrame:
    """Load delisting table for alive/dead filtering.

    Missing delist date => still alive.
    """
    raw = _read_csv(delist_path)
    renamed = raw.rename(columns=DELIST_COL_MAP)

    keep = [c for c in ["ts_code", "stock_code", "delist_date"] if c in renamed.columns]
    if "stock_code" not in keep:
        raise ValueError("Delist table must contain 股票代码/stock_code")

    out = renamed[keep].copy()
    out["stock_code"] = normalize_stock_code(out["stock_code"])

    # Delist date is often int-like yyyymmdd; coerce robustly.
    out["delist_date"] = _parse_cn_date(out["delist_date"])
    out = ensure_ts_code(out)

    return out[["ts_code", "stock_code", "delist_date"]].drop_duplicates("stock_code")


def load_stock_list(stock_list_path: Path) -> pd.DataFrame:
    """Load stock metadata with explicit schema mapping.

    Keeps only the columns needed currently for:
      - listing-age filters (`ipo_date`)
      - industry neutralization (`industry`)
    """
    raw = _read_csv(stock_list_path)
    renamed = raw.rename(columns=STOCK_LIST_COL_MAP)

    required = ["stock_code", "industry", "ipo_date"]
    missing = [c for c in required if c not in renamed.columns]
    if missing:
        raise ValueError(f"Stock list missing required columns: {missing}")

    out = renamed[[c for c in ["ts_code", "stock_code", "industry", "ipo_date"] if c in renamed.columns]].copy()
    out["stock_code"] = normalize_stock_code(out["stock_code"])
    out["industry"] = out["industry"].astype("string").str.strip()
    out["ipo_date"] = _parse_cn_date(out["ipo_date"])
    out = ensure_ts_code(out)

    return out[["ts_code", "stock_code", "industry", "ipo_date"]].drop_duplicates("stock_code")
