from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisualizationConfig:
    annualization_days: int = 252
    rolling_ic_window: int = 60
    quantiles_default: int = 10
    max_dist_sample: int = 500_000
