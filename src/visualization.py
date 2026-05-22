from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation_plots import generate_all_evaluation_plots


REQUIRED_FILES = [
    Path("ic_analysis/ic_timeseries.csv"),
    Path("quantile_backtest/quantile_returns.csv"),
    Path("quantile_backtest/long_short_metrics.csv"),
]


def has_required_evaluation_artifacts(eval_root: Path) -> bool:
    return all((eval_root / p).exists() for p in REQUIRED_FILES)


def run_visualization_from_saved(eval_root: Path, factor_label: str) -> int:
    """Generate plots from existing evaluation artifacts only.

    Returns number of png files under plots directory after generation.
    """
    if not has_required_evaluation_artifacts(eval_root):
        return 0

    generate_all_evaluation_plots(eval_root, factor_label=factor_label, rolling_window=126)

    plots_dir = eval_root / "plots"
    if not plots_dir.exists():
        return 0
    return len(list(plots_dir.glob("*.png")))
