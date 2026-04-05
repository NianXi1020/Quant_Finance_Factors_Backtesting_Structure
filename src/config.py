from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


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
    delist_path: Path = Path("data/raw/A_share_data/Delisting/delisting.csv")


@dataclass(frozen=True)
class UniverseConfig:
    """Settings for universe construction."""

    min_listing_days: int = 60


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution controls for performance/debugging."""

    use_parallel: bool = True
    # Default to all available logical CPUs minus one, minimum 1.
    n_jobs: int = max(1, (os.cpu_count() or 1) - 1)
    verbose_timing: bool = True


@dataclass(frozen=True)
class CacheConfig:
    """Cache/rebuild controls for staged artifacts."""

    use_cache: bool = True
    force_rebuild_cleaning: bool = False
    force_rebuild_universe: bool = False
    force_rebuild_labels: bool = False
    force_rebuild_factors: bool = False


@dataclass(frozen=True)
class StageConfig:
    """Run/skip control for staged workflow."""

    run_cleaning: bool = True
    run_universe: bool = True
    run_labels: bool = True
    run_factor: bool = True
    run_evaluation: bool = True
    use_cached_interim: bool = True


@dataclass(frozen=True)
class StorageConfig:
    """Standardized storage layout for interim/processed/output artifacts."""

    interim_root: Path = Path("data/interim")
    processed_root: Path = Path("data/processed")
    outputs_root: Path = Path("outputs")

    daily_panel_clean: Path = Path("data/interim/panels/daily_panel_clean.parquet")
    stock_list_clean: Path = Path("data/interim/metadata/stock_list_clean.parquet")
    delist_clean: Path = Path("data/interim/metadata/delist_clean.parquet")
    universe_basic: Path = Path("data/interim/universe/universe_basic.parquet")
    forward_returns_1d: Path = Path("data/interim/labels/forward_returns_1d.parquet")
    factor_registry: Path = Path("data/processed/manifest/factor_registry.csv")


@dataclass(frozen=True)
class FactorConfig:
    """Single-factor run selection for stage-C/D pipeline."""

    family: str = "momentum"
    name: str = "rs"
    lookback: int = 30
    quantiles: int = 5
    eval_signal_col: str = "factor_indneu"


@dataclass(frozen=True)
class LogConfig:
    """Lightweight progress logging options."""

    enabled: bool = True
    verbose: bool = False


@dataclass(frozen=True)
class DebugConfig:
    """Debug-only output toggles."""

    save_combined_panel: bool = False
    combined_panel_path: Path = Path("output/pipeline_output_debug.parquet")


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline settings."""

    data: DataConfig = DataConfig()
    universe: UniverseConfig = UniverseConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    stage: StageConfig = StageConfig()
    cache: CacheConfig = CacheConfig()
    storage: StorageConfig = StorageConfig()
    factor: FactorConfig = FactorConfig()
    log: LogConfig = LogConfig()
    debug: DebugConfig = DebugConfig()
    factor_lookback_days: int = 20
