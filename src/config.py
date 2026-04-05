from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    """Centralized data-path config.

    Assumptions:
    - `daily_path` points to a directory containing per-stock HFQ files
      like `000001_daily_hfq.csv`.
    - loader implementation remains modular, so single-file mode can be
      supported later via the loader adapter.
    - `stock_list_path` and `delist_path` are single CSV files.
    """

    root: Path = Path("data/raw/A_share_data")
    daily_path: Path = Path("data/raw/A_share_data/daily_hfq")
    stock_list_path: Path = Path("data/raw/A_share_data/Stock_list.csv")
    delist_path: Path = Path("data/raw/A_share_data/delist.csv")


@dataclass(frozen=True)
class UniverseConfig:
    """Settings for universe construction."""

    min_listing_days: int = 60


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline settings."""

    data: DataConfig = DataConfig()
    universe: UniverseConfig = UniverseConfig()
    factor_lookback_days: int = 20
