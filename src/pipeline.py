from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

import pandas as pd

from src.config import PipelineConfig
from src.data.loaders import load_daily_data, load_delist_data, load_stock_list
from src.evaluation import run_ic_analysis, run_quantile_backtest
from src.factors.basic import compute_macd_family, compute_window_factor, preprocess_single_factor
from src.universe import attach_universe_flags
from src.utils.progress import StepLogger


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _shape_info(df: pd.DataFrame) -> str:
    parts = [f"rows={len(df):,}", f"cols={df.shape[1]}"]
    if "ts_code" in df.columns:
        parts.append(f"n_ts={df['ts_code'].nunique():,}")
    if "date" in df.columns and not df["date"].dropna().empty:
        parts.append(f"date_range={df['date'].min().date()}~{df['date'].max().date()}")
    return ", ".join(parts)


def _load_cached_or_error(path: Path, stage_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{stage_name} is required but cache file was not found: {path}. "
            f"Enable the corresponding stage flag or generate the artifact first."
        )
    return pd.read_parquet(path)


def _build_or_load(
    path: Path,
    builder,
    *,
    stage_enabled: bool,
    use_cache: bool,
    force_rebuild: bool,
    stage_name: str,
) -> pd.DataFrame:
    if stage_enabled:
        if use_cache and (not force_rebuild) and path.exists():
            return pd.read_parquet(path)
        df = builder()
        _ensure_parent(path)
        df.to_parquet(path, index=False)
        return df

    if use_cache:
        return _load_cached_or_error(path, stage_name)

    raise ValueError(f"{stage_name} is disabled and use_cached_interim=False, so required data is unavailable.")


def _factor_file_name(name: str, lookback: int) -> str:
    if name == "macd":
        return "macd.parquet"
    return f"{name}_{lookback}.parquet"


def _save_factor_registry(registry_path: Path, row: dict[str, str | int | float]) -> None:
    _ensure_parent(registry_path)
    if registry_path.exists():
        reg = pd.read_csv(registry_path)
    else:
        reg = pd.DataFrame(columns=["family", "name", "lookback", "file_path", "eval_signal_col"])

    current = pd.DataFrame([row])
    reg = pd.concat([reg, current], ignore_index=True)
    reg = reg.drop_duplicates(subset=["family", "name", "lookback"], keep="last")
    reg.to_csv(registry_path, index=False)


def _compute_forward_returns_1d(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily[["date", "ts_code", "stock_code", "close"]].copy()
    out = out.sort_values(["stock_code", "date"])
    out["fwd_1d_return"] = out.groupby("stock_code", group_keys=False)["close"].pct_change().shift(-1)
    return out[["date", "ts_code", "stock_code", "fwd_1d_return"]]


def _resolve_eval_signal_col(cfg: PipelineConfig) -> str:
    if cfg.factor.name == "macd":
        return "macd_bar_indneu"
    return cfg.factor.eval_signal_col


def run_pipeline(config: PipelineConfig | None = None) -> dict[str, Path]:
    cfg = config or PipelineConfig()
    logger = StepLogger(enabled=cfg.log.enabled, verbose=cfg.log.verbose)
    t_pipeline = time.perf_counter()
    n_jobs = max(1, int(cfg.runtime.n_jobs))

    # Stage A
    t = logger.step("Stage A: Daily panel")
    daily = _build_or_load(
        cfg.storage.daily_panel_clean,
        lambda: load_daily_data(cfg.data.daily_path, use_parallel=cfg.runtime.use_parallel, n_jobs=n_jobs),
        stage_enabled=cfg.stage.run_cleaning,
        use_cache=cfg.stage.use_cached_interim,
        force_rebuild=cfg.cache.force_rebuild_cleaning,
        stage_name="Daily cleaning stage",
    )
    logger.done("Daily panel ready", t, extra=_shape_info(daily))

    t = logger.step("Stage A: Stock metadata")
    stock_meta = _build_or_load(
        cfg.storage.stock_list_clean,
        lambda: load_stock_list(cfg.data.stock_list_path),
        stage_enabled=cfg.stage.run_cleaning,
        use_cache=cfg.stage.use_cached_interim,
        force_rebuild=cfg.cache.force_rebuild_cleaning,
        stage_name="Stock metadata cleaning stage",
    )
    logger.done("Stock metadata ready", t, extra=_shape_info(stock_meta))

    t = logger.step("Stage A: Delisting metadata")
    delist = _build_or_load(
        cfg.storage.delist_clean,
        lambda: load_delist_data(cfg.data.delist_path),
        stage_enabled=cfg.stage.run_cleaning,
        use_cache=cfg.stage.use_cached_interim,
        force_rebuild=cfg.cache.force_rebuild_cleaning,
        stage_name="Delisting cleaning stage",
    )
    logger.done("Delisting metadata ready", t, extra=_shape_info(delist))

    # Stage B
    t = logger.step("Stage B: Universe")
    universe = _build_or_load(
        cfg.storage.universe_basic,
        lambda: attach_universe_flags(
            daily_df=daily,
            stock_meta=stock_meta,
            delist_df=delist,
            min_listing_days=cfg.universe.min_listing_days,
        )[["date", "ts_code", "stock_code", "industry", "is_alive", "is_old_enough", "in_universe"]],
        stage_enabled=cfg.stage.run_universe,
        use_cache=cfg.stage.use_cached_interim,
        force_rebuild=cfg.cache.force_rebuild_universe,
        stage_name="Universe stage",
    )
    logger.done("Universe ready", t, extra=_shape_info(universe))

    t = logger.step("Stage B: Forward labels")
    labels = _build_or_load(
        cfg.storage.forward_returns_1d,
        lambda: _compute_forward_returns_1d(daily),
        stage_enabled=cfg.stage.run_labels,
        use_cache=cfg.stage.use_cached_interim,
        force_rebuild=cfg.cache.force_rebuild_labels,
        stage_name="Label stage",
    )
    logger.done("Forward labels ready", t, extra=_shape_info(labels))

    # Stage C
    factor_filename = _factor_file_name(cfg.factor.name, cfg.factor.lookback)
    factor_path = cfg.storage.processed_root / "factors" / cfg.factor.family / factor_filename

    t = logger.step(f"Stage C: Factor {cfg.factor.name}_{cfg.factor.lookback}")

    def _build_factor() -> pd.DataFrame:
        base = daily[["date", "ts_code", "stock_code", "close", "high", "low", "turnover"]].merge(
            universe[["date", "stock_code", "industry", "in_universe"]],
            on=["date", "stock_code"],
            how="left",
        )

        if cfg.factor.name == "macd":
            fac = compute_macd_family(base, use_parallel=cfg.runtime.use_parallel, n_jobs=n_jobs)
            keep = [
                "date",
                "ts_code",
                "stock_code",
                "dif_raw",
                "dif_win",
                "dif_z",
                "dif_indneu",
                "dea_raw",
                "dea_win",
                "dea_z",
                "dea_indneu",
                "macd_bar_raw",
                "macd_bar_win",
                "macd_bar_z",
                "macd_bar_indneu",
            ]
            return fac[keep].sort_values(["date", "stock_code"]).reset_index(drop=True)

        fac = compute_window_factor(
            base,
            factor_name=cfg.factor.name,
            lookback=cfg.factor.lookback,
            use_parallel=cfg.runtime.use_parallel,
            n_jobs=n_jobs,
        )
        fac = preprocess_single_factor(fac, raw_col="factor_raw")
        keep = ["date", "ts_code", "stock_code", "factor_raw", "factor_win", "factor_z", "factor_indneu"]
        return fac[keep].sort_values(["date", "stock_code"]).reset_index(drop=True)

    factor_df = _build_or_load(
        factor_path,
        _build_factor,
        stage_enabled=cfg.stage.run_factor,
        use_cache=cfg.stage.use_cached_interim,
        force_rebuild=cfg.cache.force_rebuild_factors,
        stage_name="Factor stage",
    )
    logger.done("Factor file ready", t, extra=f"{_shape_info(factor_df)}, path={factor_path}")

    eval_signal_col = _resolve_eval_signal_col(cfg)
    _save_factor_registry(
        cfg.storage.factor_registry,
        {
            "family": cfg.factor.family,
            "name": cfg.factor.name,
            "lookback": cfg.factor.lookback,
            "file_path": str(factor_path),
            "eval_signal_col": eval_signal_col,
        },
    )

    # Stage D
    eval_base = (
        cfg.storage.outputs_root
        / "single_factor"
        / cfg.factor.family
        / ("macd" if cfg.factor.name == "macd" else f"{cfg.factor.name}_{cfg.factor.lookback}")
    )
    ic_dir = eval_base / "ic_analysis"
    q_dir = eval_base / "quantile_backtest"
    summaries_dir = eval_base / "summaries"
    logs_dir = eval_base / "logs"
    for d in [ic_dir, q_dir, summaries_dir, logs_dir, eval_base / "stability"]:
        d.mkdir(parents=True, exist_ok=True)

    t = logger.step("Stage D: Preparing evaluation input")
    eval_df = (
        factor_df.merge(universe[["date", "stock_code", "in_universe"]], on=["date", "stock_code"], how="left")
        .merge(labels[["date", "stock_code", "fwd_1d_return"]], on=["date", "stock_code"], how="left")
    )
    eval_df = eval_df.loc[eval_df["in_universe"] == True].copy()
    if eval_signal_col not in eval_df.columns:
        raise KeyError(f"Evaluation signal column `{eval_signal_col}` not found in factor data.")
    eval_df = eval_df.rename(columns={eval_signal_col: "factor_indneu"})
    logger.done("Evaluation input ready", t, extra=_shape_info(eval_df))

    if cfg.stage.run_evaluation:
        t = logger.step("Stage D: Running IC analysis")
        ic_metrics = run_ic_analysis(eval_df, ic_dir)
        logger.done("IC analysis done", t)

        t = logger.step("Stage D: Running quantile backtest")
        q_metrics = run_quantile_backtest(eval_df, q_dir, quantiles=cfg.factor.quantiles)
        logger.done("Quantile backtest done", t)

        pd.DataFrame([{**ic_metrics, **q_metrics}]).to_csv(summaries_dir / "summary_metrics.csv", index=False)
        pd.DataFrame([asdict(cfg.factor)]).to_csv(summaries_dir / "factor_config_snapshot.csv", index=False)
    else:
        logger.info("Evaluation stage skipped by config.")

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
