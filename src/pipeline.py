from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

import pandas as pd

from src.config import PipelineConfig
from src.data.loaders import load_daily_data, load_delist_data, load_stock_list
from src.evaluation import run_ic_analysis, run_quantile_backtest
from src.factors.basic import compute_momentum_factor, zscore_cross_section
from src.universe import attach_universe_flags
from src.utils.progress import StepLogger


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_or_build_parquet(path: Path, builder, use_cache: bool, force_rebuild: bool) -> pd.DataFrame:
    if use_cache and (not force_rebuild) and path.exists():
        return pd.read_parquet(path)
    df = builder()
    _ensure_parent(path)
    df.to_parquet(path, index=False)
    return df


def _factor_file_name(name: str, lookback: int) -> str:
    return f"{name}_{lookback}.parquet"


def _save_factor_registry(registry_path: Path, row: dict[str, str | int | float]) -> None:
    _ensure_parent(registry_path)
    if registry_path.exists():
        reg = pd.read_csv(registry_path)
    else:
        reg = pd.DataFrame(columns=["family", "name", "lookback", "file_path"])

    current = pd.DataFrame([row])
    reg = pd.concat([reg, current], ignore_index=True)
    reg = reg.drop_duplicates(subset=["family", "name", "lookback"], keep="last")
    reg.to_csv(registry_path, index=False)


def _compute_forward_returns_1d(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily[["date", "ts_code", "stock_code", "close"]].copy()
    out = out.sort_values(["stock_code", "date"])
    out["fwd_1d_return"] = out.groupby("stock_code", group_keys=False)["close"].pct_change().shift(-1)
    return out[["date", "ts_code", "stock_code", "fwd_1d_return"]]


def run_pipeline(config: PipelineConfig | None = None) -> dict[str, Path]:
    """Run staged pipeline with modular storage, cache, and evaluation outputs."""
    cfg = config or PipelineConfig()
    logger = StepLogger(enabled=cfg.log.enabled, verbose=cfg.log.verbose)
    t_pipeline = time.perf_counter()

    n_jobs = max(1, int(cfg.runtime.n_jobs))

    # ---------------- Stage A: cleaning + interim cache ----------------
    t = logger.step("Stage A: Loading/Cleaning daily panel")
    daily = _load_or_build_parquet(
        cfg.storage.daily_panel_clean,
        lambda: load_daily_data(cfg.data.daily_path, use_parallel=cfg.runtime.use_parallel, n_jobs=n_jobs),
        use_cache=cfg.cache.use_cache,
        force_rebuild=cfg.cache.force_rebuild_cleaning,
    )
    logger.done("Daily panel ready", t, extra=f"rows={len(daily):,}")

    t = logger.step("Stage A: Cleaning stock metadata")
    stock_meta = _load_or_build_parquet(
        cfg.storage.stock_list_clean,
        lambda: load_stock_list(cfg.data.stock_list_path),
        use_cache=cfg.cache.use_cache,
        force_rebuild=cfg.cache.force_rebuild_cleaning,
    )
    logger.done("Stock metadata ready", t, extra=f"rows={len(stock_meta):,}")

    t = logger.step("Stage A: Cleaning delisting metadata")
    delist = _load_or_build_parquet(
        cfg.storage.delist_clean,
        lambda: load_delist_data(cfg.data.delist_path),
        use_cache=cfg.cache.use_cache,
        force_rebuild=cfg.cache.force_rebuild_cleaning,
    )
    logger.done("Delisting metadata ready", t, extra=f"rows={len(delist):,}")

    # ---------------- Stage B: reusable universe + labels ----------------
    t = logger.step("Stage B: Building baseline universe")

    def _build_universe() -> pd.DataFrame:
        u = attach_universe_flags(
            daily_df=daily,
            stock_meta=stock_meta,
            delist_df=delist,
            min_listing_days=cfg.universe.min_listing_days,
        )
        cols = ["date", "ts_code", "stock_code", "industry", "is_alive", "is_old_enough", "in_universe"]
        return u[cols]

    universe = _load_or_build_parquet(
        cfg.storage.universe_basic,
        _build_universe,
        use_cache=cfg.cache.use_cache,
        force_rebuild=cfg.cache.force_rebuild_universe,
    )
    logger.done("Baseline universe ready", t, extra=f"rows={len(universe):,}")

    t = logger.step("Stage B: Building forward 1-day labels")
    labels = _load_or_build_parquet(
        cfg.storage.forward_returns_1d,
        lambda: _compute_forward_returns_1d(daily),
        use_cache=cfg.cache.use_cache,
        force_rebuild=cfg.cache.force_rebuild_labels,
    )
    logger.done("Forward labels ready", t, extra=f"rows={len(labels):,}")

    # ---------------- Stage C: factor file ----------------
    factor_filename = _factor_file_name(cfg.factor.name, cfg.factor.lookback)
    factor_path = cfg.storage.processed_root / "factors" / cfg.factor.family / factor_filename

    t = logger.step(f"Stage C: Computing factor {cfg.factor.family} {cfg.factor.name}_{cfg.factor.lookback}")

    def _build_factor() -> pd.DataFrame:
        base = daily[["date", "ts_code", "stock_code", "close"]].merge(
            universe[["date", "stock_code", "industry", "in_universe"]],
            on=["date", "stock_code"],
            how="left",
        )
        fac = compute_momentum_factor(
            base,
            lookback_days=cfg.factor.lookback,
            use_parallel=cfg.runtime.use_parallel,
            n_jobs=n_jobs,
        )
        fac = fac.rename(columns={"mom_raw": "factor_raw"})

        # Lightweight winsorization by date to stabilize outliers.
        def _winsor(s: pd.Series) -> pd.Series:
            lo, hi = s.quantile(0.01), s.quantile(0.99)
            return s.clip(lower=lo, upper=hi)

        fac["factor_win"] = fac.groupby("date", group_keys=False)["factor_raw"].apply(_winsor)
        fac["factor_z"] = zscore_cross_section(fac, value_col="factor_win", group_col=None)
        fac["factor_indneu"] = zscore_cross_section(fac, value_col="factor_win", group_col="industry")

        cols = ["date", "ts_code", "stock_code", "factor_raw", "factor_win", "factor_z", "factor_indneu"]
        return fac[cols].sort_values(["date", "stock_code"]).reset_index(drop=True)

    factor_df = _load_or_build_parquet(
        factor_path,
        _build_factor,
        use_cache=cfg.cache.use_cache,
        force_rebuild=cfg.cache.force_rebuild_factors,
    )
    logger.done("Factor file ready", t, extra=f"rows={len(factor_df):,}, path={factor_path}")

    _save_factor_registry(
        cfg.storage.factor_registry,
        {
            "family": cfg.factor.family,
            "name": cfg.factor.name,
            "lookback": cfg.factor.lookback,
            "file_path": str(factor_path),
        },
    )

    # ---------------- Stage D: evaluation outputs ----------------
    eval_base = (
        cfg.storage.outputs_root
        / "single_factor"
        / cfg.factor.family
        / f"{cfg.factor.name}_{cfg.factor.lookback}"
    )
    ic_dir = eval_base / "ic_analysis"
    q_dir = eval_base / "quantile_backtest"
    summaries_dir = eval_base / "summaries"
    logs_dir = eval_base / "logs"
    for d in [ic_dir, q_dir, summaries_dir, logs_dir, eval_base / "stability"]:
        d.mkdir(parents=True, exist_ok=True)

    eval_df = (
        factor_df.merge(universe[["date", "stock_code", "in_universe"]], on=["date", "stock_code"], how="left")
        .merge(labels[["date", "stock_code", "fwd_1d_return"]], on=["date", "stock_code"], how="left")
    )
    eval_df = eval_df.loc[eval_df["in_universe"] == True].copy()

    t = logger.step("Stage D: Running IC analysis")
    ic_metrics = run_ic_analysis(eval_df, ic_dir)
    logger.done("IC analysis done", t)

    t = logger.step("Stage D: Running quantile backtest")
    q_metrics = run_quantile_backtest(eval_df, q_dir, quantiles=cfg.factor.quantiles)
    logger.done("Quantile backtest done", t)

    pd.DataFrame([{**ic_metrics, **q_metrics}]).to_csv(summaries_dir / "summary_metrics.csv", index=False)
    pd.DataFrame([asdict(cfg.factor)]).to_csv(summaries_dir / "factor_config_snapshot.csv", index=False)

    # Optional debug-only giant panel export (disabled by default).
    if cfg.debug.save_combined_panel:
        t = logger.step("Debug: Saving combined panel")
        debug_panel = eval_df.sort_values(["date", "stock_code"]).reset_index(drop=True)
        _ensure_parent(cfg.debug.combined_panel_path)
        debug_panel.to_parquet(cfg.debug.combined_panel_path, index=False)
        logger.done("Debug combined panel saved", t, extra=f"path={cfg.debug.combined_panel_path}")

    logger.complete(t_pipeline)

    return {
        "daily_panel_clean": cfg.storage.daily_panel_clean,
        "stock_list_clean": cfg.storage.stock_list_clean,
        "delist_clean": cfg.storage.delist_clean,
        "universe_basic": cfg.storage.universe_basic,
        "forward_returns_1d": cfg.storage.forward_returns_1d,
        "factor_file": factor_path,
        "evaluation_root": eval_base,
    }


def main() -> None:
    cfg = PipelineConfig()
    artifacts = run_pipeline(cfg)
    print("Artifacts saved:")
    for k, v in artifacts.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
