from __future__ import annotations

import pandas as pd


def compute_momentum_factor(
    df: pd.DataFrame,
    lookback_days: int = 20,
    price_col: str = "close",
) -> pd.DataFrame:
    """Compute a simple momentum factor from close prices.

    Factor at date t uses only data up to and including t.
    Portfolio weights/signals should be shifted by 1 day before applying returns.
    """
    out = df.sort_values(["stock_code", "date"]).copy()
    out["mom_raw"] = out.groupby("stock_code", group_keys=False)[price_col].pct_change(lookback_days)
    return out


def zscore_cross_section(df: pd.DataFrame, value_col: str, group_col: str | None = None) -> pd.Series:
    """Cross-sectional z-score by date (optionally within industry)."""

    def _z(s: pd.Series) -> pd.Series:
        std = s.std(ddof=0)
        if pd.isna(std) or std == 0:
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std

    if group_col:
        return df.groupby(["date", group_col], group_keys=False)[value_col].apply(_z)
    return df.groupby("date", group_keys=False)[value_col].apply(_z)
