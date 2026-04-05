from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable
from concurrent.futures import ProcessPoolExecutor

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


def _iter_daily_files_from_dir(daily_dir: Path, pattern: str = "*_daily_hfq.csv") -> Iterable[Path]:
    if not daily_dir.exists():
        raise FileNotFoundError(f"Daily data directory not found: {daily_dir}")
    if not daily_dir.is_dir():
        raise NotADirectoryError(f"Expected directory for daily data, got: {daily_dir}")

    files = sorted(daily_dir.glob(pattern))
    if not files:
        raise ValueError(f"No daily files matched pattern `{pattern}` under: {daily_dir}")

    yield from files


def _infer_stock_code_from_filename(file_path: Path) -> str | None:
    """Infer stock_code from filenames like `000001_daily_hfq.csv`."""
    m = re.match(r"^(\d{6})_daily_hfq\.csv$", file_path.name)
    if m:
        return m.group(1)
    return None


def load_daily_data_single_file(csv_path: Path) -> pd.DataFrame:
    """Adapter for potential future single-file daily format."""
    raw = _read_csv(csv_path)
    renamed = raw.rename(columns=DAILY_COL_MAP)
    required = ["date", "stock_code", "close"]
    missing = [c for c in required if c not in renamed.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    keep_cols = [v for v in DAILY_COL_MAP.values() if v in renamed.columns]
    cleaned = renamed[keep_cols].copy()
    cleaned["date"] = _parse_cn_date(cleaned["date"])
    cleaned["stock_code"] = normalize_stock_code(cleaned["stock_code"])
    for c in [k for k in ["open", "close", "high", "low", "volume", "amount", "amplitude", "pct_change", "price_change", "turnover"] if k in cleaned.columns]:
        cleaned[c] = pd.to_numeric(cleaned[c], errors="coerce")
    cleaned = cleaned.dropna(subset=["date", "stock_code"]).sort_values(["date", "stock_code"])
    cleaned = ensure_ts_code(cleaned)
    return cleaned.reset_index(drop=True)


def _clean_daily_file(file_path: Path) -> pd.DataFrame:
    """Per-file cleaner used by both serial and process-parallel loaders."""
    raw = _read_csv(file_path)
    renamed = raw.rename(columns=DAILY_COL_MAP)
    required = ["date", "close"]
    missing = [c for c in required if c not in renamed.columns]
    if missing:
        raise ValueError(f"{file_path} missing required columns: {missing}")

    # Some vendor files may omit stock code; infer from filename if needed.
    if "stock_code" not in renamed.columns:
        inferred = _infer_stock_code_from_filename(file_path)
        if inferred is None:
            raise ValueError(
                f"{file_path} has no 股票代码 column and filename does not match `000001_daily_hfq.csv`."
            )
        renamed["stock_code"] = inferred

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

    cleaned = cleaned.dropna(subset=["date", "stock_code"]).sort_values("date")
    return cleaned


def load_daily_data(
    daily_path: Path,
    use_parallel: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Load and clean daily OHLCV-like data.

    Current default mode expects a directory containing per-stock files
    matching `*_daily_hfq.csv` (e.g., `000001_daily_hfq.csv`).

    Adaptation note:
    - this function intentionally keeps per-file cleaning modular, so it can
      be extended later to support single-file input mode if needed.

    Key anti-look-ahead assumption:
    - `date` is trade date; later pipeline must shift factor->position before return calc.
    """
    files = list(_iter_daily_files_from_dir(daily_path, pattern="*_daily_hfq.csv"))
    parallel = use_parallel and n_jobs > 1 and len(files) > 1

    if parallel:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            frames = list(ex.map(_clean_daily_file, files))
    else:
        frames = [_clean_daily_file(file_path) for file_path in files]

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
