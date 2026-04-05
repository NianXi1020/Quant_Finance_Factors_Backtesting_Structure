from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_ic_analysis(df: pd.DataFrame, output_dir: Path) -> dict[str, float]:
    """Compute and persist IC / Rank-IC summaries and time series."""
    _ensure_dir(output_dir)

    work = df.dropna(subset=["factor_indneu", "fwd_1d_return"]).copy()
    if work.empty:
        summary = {"ic_mean": np.nan, "ic_std": np.nan, "ic_ir": np.nan, "rank_ic_mean": np.nan, "rank_ic_std": np.nan, "rank_ic_ir": np.nan}
        pd.DataFrame([summary]).to_csv(output_dir / "ic_summary.csv", index=False)
        pd.DataFrame([summary]).to_csv(output_dir / "rank_ic_summary.csv", index=False)
        return summary

    ic_by_date = work.groupby("date").apply(lambda x: x["factor_indneu"].corr(x["fwd_1d_return"], method="pearson")).rename("ic").dropna()
    rank_ic_by_date = work.groupby("date").apply(lambda x: x["factor_indneu"].corr(x["fwd_1d_return"], method="spearman")).rename("rank_ic").dropna()

    ic_by_date.to_frame().to_csv(output_dir / "ic_timeseries.csv", index=True)
    rank_ic_by_date.to_frame().to_csv(output_dir / "rank_ic_timeseries.csv", index=True)

    def _summary(s: pd.Series, prefix: str) -> dict[str, float]:
        std = s.std(ddof=0)
        ir = np.nan if pd.isna(std) or std == 0 else s.mean() / std
        return {f"{prefix}_mean": s.mean(), f"{prefix}_std": std, f"{prefix}_ir": ir}

    ic_summary = _summary(ic_by_date, "ic")
    rank_ic_summary = _summary(rank_ic_by_date, "rank_ic")
    pd.DataFrame([ic_summary]).to_csv(output_dir / "ic_summary.csv", index=False)
    pd.DataFrame([rank_ic_summary]).to_csv(output_dir / "rank_ic_summary.csv", index=False)
    return {**ic_summary, **rank_ic_summary}


def run_quantile_backtest(df: pd.DataFrame, output_dir: Path, quantiles: int = 5) -> dict[str, float]:
    """Compute quantile returns and long-short stats; persist CSV artifacts."""
    _ensure_dir(output_dir)

    work = df.dropna(subset=["factor_indneu", "fwd_1d_return"]).copy()
    if work.empty:
        metrics = {"long_short_mean": np.nan, "long_short_std": np.nan, "long_short_ir": np.nan}
        pd.DataFrame([metrics]).to_csv(output_dir / "long_short_metrics.csv", index=False)
        return metrics

    def _bucket(s: pd.Series) -> pd.Series:
        if s.nunique() < quantiles:
            return pd.Series(np.nan, index=s.index)
        return pd.qcut(s, q=quantiles, labels=False, duplicates="drop") + 1

    work["quantile"] = work.groupby("date")["factor_indneu"].transform(_bucket)
    qret = work.dropna(subset=["quantile"]).groupby(["date", "quantile"])["fwd_1d_return"].mean().unstack()
    qret.columns = [f"Q{int(c)}" for c in qret.columns]

    low = f"Q1"
    high = f"Q{quantiles}"
    qret["long_short"] = qret.get(high, np.nan) - qret.get(low, np.nan)

    qret.to_csv(output_dir / "quantile_returns.csv", index=True)
    (1 + qret.fillna(0)).cumprod().to_csv(output_dir / "quantile_nav.csv", index=True)

    ls = qret["long_short"].dropna()
    ls_std = ls.std(ddof=0)
    metrics = {
        "long_short_mean": ls.mean(),
        "long_short_std": ls_std,
        "long_short_ir": np.nan if pd.isna(ls_std) or ls_std == 0 else ls.mean() / ls_std,
    }
    pd.DataFrame([metrics]).to_csv(output_dir / "long_short_metrics.csv", index=False)

    if not ls.empty:
        yearly = ls.groupby(ls.index.year).agg(["mean", "std", "count"])
        yearly.to_csv(output_dir / "yearly_stability.csv", index=True)

    return metrics
