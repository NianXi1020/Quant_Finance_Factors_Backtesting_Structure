from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

import pandas as pd

from src.config import PipelineConfig
from src.data.loaders import load_daily_data, load_delist_data, load_stock_list
from src.factors.basic import compute_momentum_factor, zscore_cross_section
from src.universe import attach_universe_flags


def run_pipeline(config: PipelineConfig | None = None) -> pd.DataFrame:
    """Run the end-to-end research pipeline.

    Output includes a no-look-ahead tradable signal:
      - factor computed at t
      - signal shifted to t+1 within each stock
    """
    cfg = config or PipelineConfig()
    n_jobs = max(1, int(cfg.runtime.n_jobs))

    t0 = time.perf_counter()
    daily = load_daily_data(
        Path(cfg.data.daily_path),
        use_parallel=cfg.runtime.use_parallel,
        n_jobs=n_jobs,
    )
    t1 = time.perf_counter()

    stock_meta = load_stock_list(Path(cfg.data.stock_list_path))
    delist = load_delist_data(Path(cfg.data.delist_path))
    t2 = time.perf_counter()

    panel = attach_universe_flags(
        daily_df=daily,
        stock_meta=stock_meta,
        delist_df=delist,
        min_listing_days=cfg.universe.min_listing_days,
    )

    panel = compute_momentum_factor(
        panel,
        lookback_days=cfg.factor_lookback_days,
        use_parallel=cfg.runtime.use_parallel,
        n_jobs=n_jobs,
    )
    panel["mom_ind_neutral"] = zscore_cross_section(panel, value_col="mom_raw", group_col="industry")
    t3 = time.perf_counter()

    # Strict anti-look-ahead: trading signal at date t uses factor from t-1.
    panel = panel.sort_values(["stock_code", "date"])
    panel["signal"] = panel.groupby("stock_code", group_keys=False)["mom_ind_neutral"].shift(1)

    # Optional convenience return field for downstream backtest (not strategy PnL yet).
    panel["fwd_1d_return"] = panel.groupby("stock_code", group_keys=False)["close"].pct_change().shift(-1)

    # Keep data only when eligible for trading universe on signal date.
    panel["tradable"] = panel["in_universe"] & panel["signal"].notna()

    if cfg.runtime.verbose_timing:
        print(
            "[timing] "
            f"daily_load={t1 - t0:.2f}s, "
            f"meta_load={t2 - t1:.2f}s, "
            f"factor_and_signal={t3 - t2:.2f}s, "
            f"total={t3 - t0:.2f}s, "
            f"use_parallel={cfg.runtime.use_parallel}, n_jobs={n_jobs}"
        )

    return panel.reset_index(drop=True)


def main() -> None:
    cfg = PipelineConfig()
    result = run_pipeline(cfg)

    output_path = Path("output/pipeline_output.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    print("Pipeline complete.")
    print(f"Config: {asdict(cfg)}")
    print(f"Rows: {len(result):,}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
