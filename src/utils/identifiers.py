from __future__ import annotations

import pandas as pd


def normalize_stock_code(series: pd.Series, width: int = 6) -> pd.Series:
    """Normalize stock codes to zero-padded strings, e.g. `1` -> `000001`."""
    out = series.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    return out.str.zfill(width)


def infer_exchange_suffix(stock_code: pd.Series) -> pd.Series:
    """Infer exchange suffix from stock-code prefix.

    This is centralized so future schema rules can be edited in one place.
    Current rule of thumb for A-shares:
      - starts with 6/9 => SH
      - else => SZ
    """
    code = normalize_stock_code(stock_code)
    return code.str[0].map(lambda c: "SH" if c in {"6", "9"} else "SZ")


def ensure_ts_code(df: pd.DataFrame, stock_code_col: str = "stock_code", ts_code_col: str = "ts_code") -> pd.DataFrame:
    """Ensure `ts_code` exists and is consistently formatted as `000001.SZ`."""
    out = df.copy()
    if stock_code_col not in out.columns:
        raise KeyError(f"Missing stock code column: {stock_code_col}")

    out[stock_code_col] = normalize_stock_code(out[stock_code_col])

    if ts_code_col in out.columns:
        raw_ts = out[ts_code_col]
        ts = raw_ts.astype("string").str.strip().str.upper()
        missing_mask = raw_ts.isna() | ts.eq("")
        if missing_mask.any():
            suffix = infer_exchange_suffix(out.loc[missing_mask, stock_code_col])
            ts.loc[missing_mask] = out.loc[missing_mask, stock_code_col] + "." + suffix
        out[ts_code_col] = ts
    else:
        suffix = infer_exchange_suffix(out[stock_code_col])
        out[ts_code_col] = out[stock_code_col] + "." + suffix

    return out
