from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import pandas as pd


def _winsor_by_date(df: pd.DataFrame, col: str, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    def _winsor(s: pd.Series) -> pd.Series:
        lo, hi = s.quantile(lower), s.quantile(upper)
        return s.clip(lower=lo, upper=hi)

    return df.groupby("date", group_keys=False)[col].apply(_winsor)


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


def preprocess_single_factor(df: pd.DataFrame, raw_col: str = "factor_raw") -> pd.DataFrame:
    """Create {raw, win, z, indneu} versions for one raw factor column."""
    out = df.copy()
    out["factor_win"] = _winsor_by_date(out, raw_col)
    out["factor_z"] = zscore_cross_section(out, value_col="factor_win", group_col=None)
    out["factor_indneu"] = zscore_cross_section(out, value_col="factor_win", group_col="industry")
    return out


def _compute_single_stock_factor(g: pd.DataFrame, factor_name: str, lookback: int) -> pd.DataFrame:
    g = g.sort_values("date").copy()

    if factor_name == "rs":
        # RS_k = close / close.shift(k) - 1
        g["factor_raw"] = g["close"].pct_change(lookback)
    elif factor_name == "hl":
        # HL_k = rolling_max(high, k) / rolling_min(low, k)
        rolling_high = g["high"].rolling(lookback, min_periods=lookback).max()
        rolling_low = g["low"].rolling(lookback, min_periods=lookback).min()
        g["factor_raw"] = rolling_high / rolling_low
    elif factor_name == "vol":
        # Vol_k = rolling std of daily close-to-close return
        ret = g["close"].pct_change()
        g["factor_raw"] = ret.rolling(lookback, min_periods=lookback).std(ddof=0)
    elif factor_name == "turnover":
        # Turnover_k = rolling mean of daily turnover
        g["factor_raw"] = g["turnover"].rolling(lookback, min_periods=lookback).mean()
    elif factor_name == "improved_mom":
        # improved_momentum_k = rolling_sum(return_t * turnover_t, k)
        ret = g["close"].pct_change()
        g["factor_raw"] = (ret * g["turnover"]).rolling(lookback, min_periods=lookback).sum()
    else:
        raise ValueError(f"Unsupported factor_name: {factor_name}")

    return g


def compute_window_factor(
    df: pd.DataFrame,
    factor_name: str,
    lookback: int,
    use_parallel: bool = False,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compute a single raw factor column for window-based families."""
    out = df.sort_values(["stock_code", "date"]).copy()
    groups = [g for _, g in out.groupby("stock_code", sort=False)]
    parallel = use_parallel and n_jobs > 1 and len(groups) > 1

    if parallel:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            pieces = list(ex.map(_compute_single_stock_factor, groups, [factor_name] * len(groups), [lookback] * len(groups)))
        out = pd.concat(pieces, ignore_index=True)
    else:
        out = pd.concat([_compute_single_stock_factor(g, factor_name, lookback) for g in groups], ignore_index=True)

    return out.sort_values(["stock_code", "date"]).reset_index(drop=True)


def _compute_single_stock_macd(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    ema12 = g["close"].ewm(span=12, adjust=False).mean()
    ema26 = g["close"].ewm(span=26, adjust=False).mean()
    g["dif_raw"] = ema12 - ema26
    g["dea_raw"] = g["dif_raw"].ewm(span=9, adjust=False).mean()
    g["macd_bar_raw"] = 2.0 * (g["dif_raw"] - g["dea_raw"])
    return g


def compute_macd_family(df: pd.DataFrame, use_parallel: bool = False, n_jobs: int = 1) -> pd.DataFrame:
    """Compute classic MACD family using close prices: DIF, DEA, MACD_bar."""
    out = df.sort_values(["stock_code", "date"]).copy()
    groups = [g for _, g in out.groupby("stock_code", sort=False)]
    parallel = use_parallel and n_jobs > 1 and len(groups) > 1

    if parallel:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            pieces = list(ex.map(_compute_single_stock_macd, groups))
        out = pd.concat(pieces, ignore_index=True)
    else:
        out = pd.concat([_compute_single_stock_macd(g) for g in groups], ignore_index=True)

    out = out.sort_values(["stock_code", "date"]).reset_index(drop=True)

    # Preprocess each MACD raw series independently.
    for base in ["dif", "dea", "macd_bar"]:
        raw = f"{base}_raw"
        win = f"{base}_win"
        z = f"{base}_z"
        indneu = f"{base}_indneu"
        out[win] = _winsor_by_date(out, raw)
        out[z] = zscore_cross_section(out, value_col=win, group_col=None)
        out[indneu] = zscore_cross_section(out, value_col=win, group_col="industry")

    return out
