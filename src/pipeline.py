from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

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

    daily = load_daily_data(Path(cfg.data.daily_path))
    stock_meta = load_stock_list(Path(cfg.data.stock_list_path))
    delist = load_delist_data(Path(cfg.data.delist_path))

    panel = attach_universe_flags(
        daily_df=daily,
        stock_meta=stock_meta,
        delist_df=delist,
        min_listing_days=cfg.universe.min_listing_days,
    )

    panel = compute_momentum_factor(panel, lookback_days=cfg.factor_lookback_days)
    panel["mom_ind_neutral"] = zscore_cross_section(panel, value_col="mom_raw", group_col="industry")

    # Strict anti-look-ahead: trading signal at date t uses factor from t-1.
    panel = panel.sort_values(["stock_code", "date"])
    panel["signal"] = panel.groupby("stock_code", group_keys=False)["mom_ind_neutral"].shift(1)

    # Optional convenience return field for downstream backtest (not strategy PnL yet).
    panel["fwd_1d_return"] = panel.groupby("stock_code", group_keys=False)["close"].pct_change().shift(-1)

    # Keep data only when eligible for trading universe on signal date.
    panel["tradable"] = panel["in_universe"] & panel["signal"].notna()

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
