from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from src.evaluation_plots import (
    plot_ic_distribution,
    plot_ic_timeseries,
    plot_ls_drawdown,
    plot_ls_nav,
    plot_monthly_heatmap,
    plot_quantile_bar,
    plot_quantile_nav,
    plot_coverage,
)
from src.visualization_config import VisualizationConfig
from src.visualization_core import plot_feature_importance, plot_pca_explained_variance, plot_rolling_ic, plot_yearly_return_bar
from src.visualization_io import load_ic_outputs, load_ml_diagnostics, load_quantile_outputs, safe_read_csv_or_parquet


def generate_factor_plots(
    factor_name: str,
    family: str,
    output_dir: Path,
    processed_factor_path: Path | None = None,
    universe_path: Path | None = None,
    strict: bool = False,
) -> dict:
    cfg = VisualizationConfig()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    report = {"factor_name": factor_name, "family": family, "created_at": pd.Timestamp.utcnow().isoformat(), "plots_generated": [], "plots_skipped": [], "missing_inputs": [], "warnings": [], "selected_inputs": {}}

    ic_info = load_ic_outputs(output_dir)
    q_info = load_quantile_outputs(output_dir)
    ml_info = load_ml_diagnostics(output_dir)

    ic = safe_read_csv_or_parquet(ic_info["ic_path"]) if ic_info["ic_path"] else pd.DataFrame()
    ric = safe_read_csv_or_parquet(ic_info["rank_ic_path"]) if ic_info["rank_ic_path"] else pd.DataFrame()
    qret = safe_read_csv_or_parquet(q_info["quantile_path"]) if q_info["quantile_path"] else pd.DataFrame()
    count_df = safe_read_csv_or_parquet(q_info["count_path"]) if q_info["count_path"] else pd.DataFrame()

    report["selected_inputs"].update({"ic": str(ic_info["ic_path"]) if ic_info["ic_path"] else None, "rank_ic": str(ic_info["rank_ic_path"]) if ic_info["rank_ic_path"] else None, "quantile": str(q_info["quantile_path"]) if q_info["quantile_path"] else None, "counts": str(q_info["count_path"]) if q_info["count_path"] else None})

    def _try(fn, name):
        try:
            ok = fn()
            if ok is False:
                report["plots_skipped"].append(name)
            else:
                report["plots_generated"].append(name)
        except Exception as e:
            report["plots_skipped"].append(name)
            report["warnings"].append(f"[VIS WARNING] factor={factor_name} plot={name} reason='{e}'")
            if strict:
                raise

    _try(lambda: (plot_ls_nav(qret, plots_dir, factor_name) or True), "01_long_short_nav.png")
    _try(lambda: (plot_ls_drawdown(qret, plots_dir, factor_name) or True), "02_long_short_drawdown.png")
    _try(lambda: (plot_quantile_nav(qret, plots_dir, factor_name) or True), "03_quantile_nav.png")
    _try(lambda: (plot_quantile_bar(qret, plots_dir, factor_name) or True), "04_quantile_return_bar.png")
    _try(lambda: plot_yearly_return_bar(qret, plots_dir / "05_yearly_return_bar.png", cfg.annualization_days), "05_yearly_return_bar.png")
    _try(lambda: (plot_ic_timeseries(ic, plots_dir, factor_name) or True), "06_ic_timeseries.png")
    _try(lambda: plot_rolling_ic(ic, ric, plots_dir / "07_rolling_ic.png", cfg.rolling_ic_window), "07_rolling_ic.png")
    _try(lambda: (plot_ic_distribution(ic, plots_dir, factor_name) or True), "08_ic_distribution.png")
    _try(lambda: (plot_monthly_heatmap(qret, plots_dir, factor_name) or True), "09_monthly_return_heatmap.png")
    _try(lambda: (plot_coverage(count_df, plots_dir, factor_name) or True), "10_coverage_timeseries.png")

    # Turnover only if quantile membership/holding information exists (not in current artifacts).
    report["plots_skipped"].append("11_turnover_timeseries.png")
    report["warnings"].append(f"[VIS WARNING] factor={factor_name} plot=11_turnover_timeseries.png reason='missing holdings/quantile membership'")

    fi = safe_read_csv_or_parquet(ml_info["feature_importance_path"]) if ml_info["feature_importance_path"] else pd.DataFrame()
    ev = safe_read_csv_or_parquet(ml_info["explained_variance_path"]) if ml_info["explained_variance_path"] else pd.DataFrame()

    _try(lambda: plot_feature_importance(fi, plots_dir / "14_feature_importance.png"), "14_feature_importance.png")
    _try(lambda: plot_pca_explained_variance(ev, plots_dir / "15_pca_explained_variance.png"), "15_pca_explained_variance.png")

    manifest_path = plots_dir / "plot_manifest.json"
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def generate_group_comparison_plots(
    factor_records: list[dict],
    group_name: str,
    output_dir: Path,
    strict: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in factor_records:
        p = Path(r["eval_root"]) / "summaries" / "summary_metrics.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue
        one = df.iloc[0].to_dict()
        one["factor_key"] = r["factor_key"]
        one["family"] = r["family"]
        rows.append(one)

    out = {"plots_generated": [], "plots_skipped": [], "missing_inputs": [], "warnings": []}
    if not rows:
        out["warnings"].append("[VIS WARNING] no comparable factors for group comparison")
        return out

    sm = pd.DataFrame(rows)
    sm.to_csv(output_dir / f"{group_name}_summary_table.csv", index=False)

    import matplotlib.pyplot as plt

    def _bar(metric: str, fname: str):
        if metric not in sm.columns:
            out["plots_skipped"].append(fname)
            return
        fig = plt.figure(figsize=(10, 4))
        x = sm["factor_key"]
        y = sm[metric]
        plt.bar(x, y)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{group_name} - {metric}")
        fig.tight_layout()
        fig.savefig(output_dir / "plots" / fname, dpi=150)
        plt.close(fig)
        out["plots_generated"].append(fname)

    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    if group_name == "ml":
        _bar("ic_mean", "group_ml_ic_comparison.png")
        _bar("rank_ic_mean", "group_ml_rankic_comparison.png")
        _bar("long_short_mean", "group_ml_long_short_return_comparison.png")
        _bar("long_short_ir", "group_ml_sharpe_comparison.png")
    else:
        _bar("ic_mean", "factor_mean_ic_comparison.png")
        _bar("rank_ic_mean", "factor_rankic_comparison.png")
        _bar("long_short_mean", "factor_long_short_return_comparison.png")
        _bar("long_short_ir", "factor_sharpe_comparison.png")
        _bar("long_short_std", "factor_max_drawdown_comparison.png")
    return out
