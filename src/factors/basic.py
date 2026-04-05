from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import pandas as pd


def _compute_single_stock_momentum(g: pd.DataFrame, lookback_days: int, price_col: str) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    g["mom_raw"] = g[price_col].pct_change(lookback_days)
    return g


def compute_momentum_factor(
    df: pd.DataFrame,
    lookback_days: int = 20,
    price_col: str = "close",
    use_parallel: bool = False,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compute a simple momentum factor from close prices.

    Factor at date t uses only data up to and including t.
    Portfolio weights/signals should be shifted by 1 day before applying returns.
    """
    out = df.sort_values(["stock_code", "date"]).copy()
    groups = [g for _, g in out.groupby("stock_code", sort=False)]
    parallel = use_parallel and n_jobs > 1 and len(groups) > 1

    if parallel:
        # Process-based parallelism is useful here because per-stock calculation
        # is independent and naturally partitioned.
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            pieces = list(ex.map(_compute_single_stock_momentum, groups, [lookback_days] * len(groups), [price_col] * len(groups)))
        out = pd.concat(pieces, ignore_index=True)
        out = out.sort_values(["stock_code", "date"])
        return out

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
