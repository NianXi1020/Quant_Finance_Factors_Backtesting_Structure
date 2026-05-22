from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class EvaluationParams:
    weighting: str = "equal"
    long_short_mode: str = "cross_sectional"
    rebalance_freq: str = "D"
    n_quantiles: int = 10
    min_valid_obs: int = 20


def _validate_params(params: EvaluationParams) -> None:
    if params.long_short_mode != "cross_sectional":
        raise NotImplementedError("Only long_short_mode='cross_sectional' is currently implemented.")
    if params.weighting != "equal":
        raise NotImplementedError("Only weighting='equal' is currently implemented.")
    if params.rebalance_freq != "D":
        raise NotImplementedError("Only daily rebalance (rebalance_freq='D') is currently implemented.")


def _daily_valid_subset(g: pd.DataFrame, min_obs: int) -> tuple[pd.DataFrame, str | None]:
    s = g.dropna(subset=["factor_indneu", "fwd_1d_return"]).copy()
    if len(s) < min_obs:
        return s, "insufficient_sample"
    if s["factor_indneu"].nunique(dropna=True) <= 1:
        return s, "constant_factor"
    if s["fwd_1d_return"].nunique(dropna=True) <= 1:
        return s, "constant_forward_return"
    return s, None


def run_ic_analysis(df: pd.DataFrame, output_dir: Path, params: EvaluationParams) -> tuple[dict[str, float], dict[str, int]]:
    """Compute IC/Rank-IC with defensive day-level validity checks."""
    _ensure_dir(output_dir)
    _validate_params(params)

    dates = sorted(df["date"].dropna().unique())
    ic_rows: list[dict[str, float]] = []
    rank_ic_rows: list[dict[str, float]] = []
    diagnostics = {
        "total_rebalance_dates": len(dates),
        "valid_ic_dates": 0,
        "skipped_ic_dates": 0,
        "insufficient_sample_dates": 0,
        "constant_factor_dates": 0,
        "constant_forward_return_dates": 0,
    }

    for d in dates:
        g = df.loc[df["date"] == d]
        valid, reason = _daily_valid_subset(g, params.min_valid_obs)
        if reason:
            diagnostics["skipped_ic_dates"] += 1
            if reason == "insufficient_sample":
                diagnostics["insufficient_sample_dates"] += 1
            elif reason == "constant_factor":
                diagnostics["constant_factor_dates"] += 1
            elif reason == "constant_forward_return":
                diagnostics["constant_forward_return_dates"] += 1
            continue

        diagnostics["valid_ic_dates"] += 1
        ic_rows.append({"date": d, "ic": valid["factor_indneu"].corr(valid["fwd_1d_return"], method="pearson")})
        rank_ic_rows.append({"date": d, "rank_ic": valid["factor_indneu"].corr(valid["fwd_1d_return"], method="spearman")})

    ic_df = pd.DataFrame(ic_rows)
    ric_df = pd.DataFrame(rank_ic_rows)
    ic_df.to_csv(output_dir / "ic_timeseries.csv", index=False)
    ric_df.to_csv(output_dir / "rank_ic_timeseries.csv", index=False)

    def _summary(s: pd.Series, prefix: str) -> dict[str, float]:
        if s.empty:
            return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan, f"{prefix}_ir": np.nan}
        std = s.std(ddof=0)
        ir = np.nan if pd.isna(std) or std == 0 else s.mean() / std
        return {f"{prefix}_mean": s.mean(), f"{prefix}_std": std, f"{prefix}_ir": ir}

    ic_summary = _summary(ic_df.get("ic", pd.Series(dtype=float)), "ic")
    rank_ic_summary = _summary(ric_df.get("rank_ic", pd.Series(dtype=float)), "rank_ic")
    pd.DataFrame([ic_summary]).to_csv(output_dir / "ic_summary.csv", index=False)
    pd.DataFrame([rank_ic_summary]).to_csv(output_dir / "rank_ic_summary.csv", index=False)
    pd.DataFrame([diagnostics]).to_csv(output_dir / "ic_diagnostics.csv", index=False)

    return {**ic_summary, **rank_ic_summary}, diagnostics


def run_quantile_backtest(df: pd.DataFrame, output_dir: Path, params: EvaluationParams) -> tuple[dict[str, float], dict[str, int]]:
    """Equal-weight cross-sectional quantile long-short backtest with diagnostics."""
    _ensure_dir(output_dir)
    _validate_params(params)

    dates = sorted(df["date"].dropna().unique())
    diagnostics = {
        "total_rebalance_dates": len(dates),
        "valid_quantile_dates": 0,
        "skipped_quantile_dates": 0,
        "empty_top_bottom_dates": 0,
        "quantile_assignment_fail_dates": 0,
        "insufficient_sample_dates": 0,
        "constant_factor_dates": 0,
        "constant_forward_return_dates": 0,
    }

    qrows: list[dict[str, float]] = []
    count_rows: list[dict[str, float]] = []

    for d in dates:
        g = df.loc[df["date"] == d]
        valid, reason = _daily_valid_subset(g, params.min_valid_obs)
        if reason:
            diagnostics["skipped_quantile_dates"] += 1
            if reason == "insufficient_sample":
                diagnostics["insufficient_sample_dates"] += 1
            elif reason == "constant_factor":
                diagnostics["constant_factor_dates"] += 1
            elif reason == "constant_forward_return":
                diagnostics["constant_forward_return_dates"] += 1
            continue

        try:
            valid = valid.copy()
            valid["quantile"] = pd.qcut(valid["factor_indneu"], q=params.n_quantiles, labels=False, duplicates="drop") + 1
        except ValueError:
            diagnostics["skipped_quantile_dates"] += 1
            diagnostics["quantile_assignment_fail_dates"] += 1
            continue

        low = 1
        high = int(valid["quantile"].max())
        if high < 2:
            diagnostics["skipped_quantile_dates"] += 1
            diagnostics["quantile_assignment_fail_dates"] += 1
            continue

        low_leg = valid.loc[valid["quantile"] == low, "fwd_1d_return"]
        high_leg = valid.loc[valid["quantile"] == high, "fwd_1d_return"]
        if low_leg.empty or high_leg.empty:
            diagnostics["skipped_quantile_dates"] += 1
            diagnostics["empty_top_bottom_dates"] += 1
            continue

        diagnostics["valid_quantile_dates"] += 1
        row = {"date": d, "Q1": low_leg.mean(), f"Q{high}": high_leg.mean()}
        row["long_short"] = row[f"Q{high}"] - row["Q1"]

        # keep all available quantiles as equal-weight bucket means
        means = valid.groupby("quantile")["fwd_1d_return"].mean()
        counts = valid.groupby("quantile")["fwd_1d_return"].size()
        for q, val in means.items():
            row[f"Q{int(q)}"] = val
        row["long_leg"] = row[f"Q{high}"]
        row["short_leg"] = row["Q1"]
        qrows.append(row)

        crow = {"date": d, "valid_n": len(valid)}
        for q, n in counts.items():
            crow[f"Q{int(q)}_n"] = int(n)
        count_rows.append(crow)

    qret = pd.DataFrame(qrows).sort_values("date") if qrows else pd.DataFrame(columns=["date", "long_short"])
    qret.to_csv(output_dir / "quantile_returns.csv", index=False)
    pd.DataFrame(count_rows).sort_values("date").to_csv(output_dir / "quantile_counts.csv", index=False)

    nav_df = qret.copy()
    for c in [c for c in nav_df.columns if c != "date"]:
        nav_df[c] = (1 + nav_df[c].fillna(0)).cumprod()
    nav_df.to_csv(output_dir / "quantile_nav.csv", index=False)

    ls = qret["long_short"].dropna() if "long_short" in qret.columns else pd.Series(dtype=float)
    ls_std = ls.std(ddof=0) if not ls.empty else np.nan
    metrics = {
        "long_short_mean": ls.mean() if not ls.empty else np.nan,
        "long_short_std": ls_std,
        "long_short_ir": np.nan if pd.isna(ls_std) or ls_std == 0 else ls.mean() / ls_std,
    }
    pd.DataFrame([metrics]).to_csv(output_dir / "long_short_metrics.csv", index=False)

    if not ls.empty:
        yearly = ls.groupby(pd.to_datetime(qret.loc[ls.index, "date"]).dt.year).agg(["mean", "std", "count"])
        yearly.to_csv(output_dir / "yearly_stability.csv", index=True)

    pd.DataFrame([diagnostics]).to_csv(output_dir / "quantile_diagnostics.csv", index=False)
    return metrics, diagnostics


def save_run_metadata(output_root: Path, metadata: dict) -> None:
    _ensure_dir(output_root)
    meta_path = output_root / "run_metadata.json"
    import json

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)


def params_to_dict(params: EvaluationParams) -> dict:
    return asdict(params)
