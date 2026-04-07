from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable
import time

import pandas as pd

from src.config import PipelineConfig
from src.evaluation_core import EvaluationParams, params_to_dict, run_ic_analysis, run_quantile_backtest, save_run_metadata
from src.evaluation_plots import generate_all_evaluation_plots
from src.factor_registry import FactorSpec, build_factor_registry, registry_by_key
from src.factors.basic import compute_macd_family, compute_window_factor, preprocess_single_factor
from src.pipeline import _shape_info, _build_or_load, _compute_forward_returns_1d, _save_factor_registry
from src.universe import attach_universe_flags
from src.data.loaders import load_daily_data, load_delist_data, load_stock_list
from src.utils.progress import StepLogger
from src.visualization import has_required_evaluation_artifacts, run_visualization_from_saved


def _factor_file_path(cfg: PipelineConfig, spec: FactorSpec) -> Path:
    fname = "macd.parquet" if spec.name == "macd" else f"{spec.name}_{spec.lookback}.parquet"
    return cfg.storage.processed_root / "factors" / spec.family / fname


def _factor_eval_root(cfg: PipelineConfig, spec: FactorSpec) -> Path:
    leaf = spec.key
    return cfg.storage.outputs_root / "single_factor" / spec.family / leaf


def is_factor_complete(cfg: PipelineConfig, spec: FactorSpec) -> bool:
    factor_file = _factor_file_path(cfg, spec)
    eval_root = _factor_eval_root(cfg, spec)
    key_files = [
        factor_file,
        eval_root / "summaries" / "summary_metrics.csv",
        eval_root / "ic_analysis" / "ic_summary.csv",
        eval_root / "quantile_backtest" / "long_short_metrics.csv",
        eval_root / "run_metadata.json",
    ]
    return all(p.exists() for p in key_files)


def _prepare_shared_artifacts(cfg: PipelineConfig, logger: StepLogger) -> dict[str, pd.DataFrame]:
    n_jobs = max(1, int(cfg.runtime.n_jobs))

    t = logger.step("Shared Stage A: Daily panel")
    daily = _build_or_load(
        cfg.storage.daily_panel_clean,
        lambda: load_daily_data(cfg.data.daily_path, use_parallel=cfg.runtime.use_parallel, n_jobs=n_jobs),
        stage_enabled=cfg.stage.run_cleaning,
        use_cache=cfg.stage.use_cached_interim,
        force_rebuild=cfg.cache.force_rebuild_cleaning,
        stage_name="Daily cleaning stage",
    )
    logger.done("Daily panel ready", t, extra=_shape_info(daily))

    t = logger.step("Shared Stage A: Stock metadata")
    stock_meta = _build_or_load(
        cfg.storage.stock_list_clean,
        lambda: load_stock_list(cfg.data.stock_list_path),
        stage_enabled=cfg.stage.run_cleaning,
        use_cache=cfg.stage.use_cached_interim,
        force_rebuild=cfg.cache.force_rebuild_cleaning,
        stage_name="Stock metadata cleaning stage",
    )
    logger.done("Stock metadata ready", t, extra=_shape_info(stock_meta))

    t = logger.step("Shared Stage A: Delisting metadata")
    delist = _build_or_load(
        cfg.storage.delist_clean,
        lambda: load_delist_data(cfg.data.delist_path),
        stage_enabled=cfg.stage.run_cleaning,
        use_cache=cfg.stage.use_cached_interim,
        force_rebuild=cfg.cache.force_rebuild_cleaning,
        stage_name="Delisting cleaning stage",
    )
    logger.done("Delisting metadata ready", t, extra=_shape_info(delist))

    t = logger.step("Shared Stage B: Universe")
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

    t = logger.step("Shared Stage B: Labels")
    labels = _build_or_load(
        cfg.storage.forward_returns_1d,
        lambda: _compute_forward_returns_1d(daily),
        stage_enabled=cfg.stage.run_labels,
        use_cache=cfg.stage.use_cached_interim,
        force_rebuild=cfg.cache.force_rebuild_labels,
        stage_name="Label stage",
    )
    logger.done("Forward labels ready", t, extra=_shape_info(labels))

    return {"daily": daily, "universe": universe, "labels": labels}


def _compute_factor_df(cfg: PipelineConfig, spec: FactorSpec, shared: dict[str, pd.DataFrame]) -> pd.DataFrame:
    n_jobs = max(1, int(cfg.runtime.n_jobs))
    daily = shared["daily"]
    universe = shared["universe"]

    base = daily[["date", "ts_code", "stock_code", "close", "high", "low", "turnover"]].merge(
        universe[["date", "stock_code", "industry", "in_universe"]], on=["date", "stock_code"], how="left"
    )

    if spec.name == "macd":
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
        factor_name=spec.name,
        lookback=int(spec.lookback),
        use_parallel=cfg.runtime.use_parallel,
        n_jobs=n_jobs,
    )
    fac = preprocess_single_factor(fac, raw_col="factor_raw")
    keep = ["date", "ts_code", "stock_code", "factor_raw", "factor_win", "factor_z", "factor_indneu"]
    return fac[keep].sort_values(["date", "stock_code"]).reset_index(drop=True)


def _run_one_factor(cfg: PipelineConfig, spec: FactorSpec, shared: dict[str, pd.DataFrame], logger: StepLogger) -> None:
    factor_path = _factor_file_path(cfg, spec)
    eval_root = _factor_eval_root(cfg, spec)
    for d in [eval_root / "ic_analysis", eval_root / "quantile_backtest", eval_root / "summaries", eval_root / "logs", eval_root / "stability"]:
        d.mkdir(parents=True, exist_ok=True)

    t = logger.step(f"Factor Stage: {spec.key}")
    factor_df = _build_or_load(
        factor_path,
        lambda: _compute_factor_df(cfg, spec, shared),
        stage_enabled=cfg.stage.run_factor,
        use_cache=cfg.stage.use_cached_interim,
        force_rebuild=cfg.cache.force_rebuild_factors,
        stage_name=f"Factor stage ({spec.key})",
    )
    logger.done("Factor file ready", t, extra=f"{_shape_info(factor_df)}, path={factor_path}")

    _save_factor_registry(
        cfg.storage.factor_registry,
        {
            "family": spec.family,
            "name": spec.name,
            "lookback": -1 if spec.lookback is None else spec.lookback,
            "file_path": str(factor_path),
            "eval_signal_col": spec.eval_signal_col,
        },
    )

    if not cfg.stage.run_evaluation:
        logger.info(f"Evaluation skipped for {spec.key} by config.")
        return

    t = logger.step(f"Evaluation Stage: {spec.key} input")
    eval_df = (
        factor_df.merge(shared["universe"][["date", "stock_code", "in_universe"]], on=["date", "stock_code"], how="left")
        .merge(shared["labels"][["date", "stock_code", "fwd_1d_return"]], on=["date", "stock_code"], how="left")
    )
    eval_df = eval_df.loc[eval_df["in_universe"] == True].copy()
    if spec.eval_signal_col not in eval_df.columns:
        raise KeyError(f"Signal column missing for {spec.key}: {spec.eval_signal_col}")
    eval_df = eval_df.rename(columns={spec.eval_signal_col: "factor_indneu"})
    logger.done("Evaluation input ready", t, extra=_shape_info(eval_df))

    params = EvaluationParams(
        weighting=cfg.evaluation.weighting,
        long_short_mode=cfg.evaluation.long_short_mode,
        rebalance_freq=cfg.evaluation.rebalance_freq,
        n_quantiles=cfg.evaluation.n_quantiles,
        min_valid_obs=cfg.evaluation.min_valid_obs,
    )

    t = logger.step(f"Evaluation Stage: {spec.key} IC")
    ic_metrics, ic_diag = run_ic_analysis(eval_df, eval_root / "ic_analysis", params)
    logger.done("IC analysis done", t, extra=f"valid_dates={ic_diag['valid_ic_dates']}")

    t = logger.step(f"Evaluation Stage: {spec.key} quantile")
    q_metrics, q_diag = run_quantile_backtest(eval_df, eval_root / "quantile_backtest", params)
    logger.done("Quantile backtest done", t, extra=f"valid_dates={q_diag['valid_quantile_dates']}")

    pd.DataFrame([{**ic_metrics, **q_metrics, **ic_diag, **q_diag}]).to_csv(eval_root / "summaries" / "summary_metrics.csv", index=False)

    t = logger.step(f"Evaluation Stage: {spec.key} plots")
    generate_all_evaluation_plots(eval_root, factor_label=spec.key, rolling_window=126)
    logger.done("Evaluation plots saved", t, extra=f"path={eval_root / 'plots'}")

    metadata = {
        "factor_key": spec.key,
        "factor_group": spec.family,
        "factor_name": spec.name,
        "lookback": spec.lookback,
        "eval_signal_col": spec.eval_signal_col,
        "weighting": params.weighting,
        "long_short_mode": params.long_short_mode,
        "rebalance_freq": params.rebalance_freq,
        "quantiles": params.n_quantiles,
        "shared_cache_reused": cfg.stage.use_cached_interim,
        "outputs_complete": is_factor_complete(cfg, spec),
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "evaluation_params": params_to_dict(params),
    }
    save_run_metadata(eval_root, metadata)


def select_factors(
    *,
    factors: list[str] | None,
    group: str | None,
) -> list[FactorSpec]:
    all_specs = build_factor_registry()
    by_key = registry_by_key()

    if factors:
        missing = [f for f in factors if f not in by_key]
        if missing:
            raise KeyError(f"Unknown factor(s): {missing}")
        return [by_key[f] for f in factors]

    if group:
        selected = [s for s in all_specs if s.family == group]
        if not selected:
            raise KeyError(f"No factors found for group `{group}`")
        return selected

    return all_specs


def run_factor_orchestration(
    cfg: PipelineConfig,
    *,
    factors: list[str] | None = None,
    group: str | None = None,
    only_missing: bool = True,
    force: bool = False,
) -> dict[str, list[str]]:
    logger = StepLogger(enabled=cfg.log.enabled, verbose=cfg.log.verbose)
    t_all = time.perf_counter()

    shared = _prepare_shared_artifacts(cfg, logger)

    selected = select_factors(factors=factors, group=group)
    logger.info(f"Selected factors: {[s.key for s in selected]}")

    to_run: list[FactorSpec] = []
    skipped: list[str] = []
    for spec in selected:
        complete = is_factor_complete(cfg, spec)
        if force:
            to_run.append(spec)
        elif only_missing and complete:
            skipped.append(spec.key)
        else:
            to_run.append(spec)

    if skipped:
        logger.info(f"Skipped completed factors: {skipped}")

    for spec in to_run:
        _run_one_factor(cfg, spec, shared, logger)

    logger.complete(t_all)
    return {
        "selected": [s.key for s in selected],
        "run": [s.key for s in to_run],
        "skipped": skipped,
    }


def run_visualization_only(
    cfg: PipelineConfig,
    *,
    factors: list[str] | None = None,
    group: str | None = None,
) -> dict[str, list[str]]:
    """Generate plots only from saved evaluation artifacts."""
    logger = StepLogger(enabled=cfg.log.enabled, verbose=cfg.log.verbose)
    selected = select_factors(factors=factors, group=group)
    generated: list[str] = []
    skipped: list[str] = []

    for spec in selected:
        eval_root = _factor_eval_root(cfg, spec)
        t = logger.step(f"Visualization: {spec.key}")
        if not has_required_evaluation_artifacts(eval_root):
            print("[SKIP] visualization skipped because evaluation outputs not found")
            skipped.append(spec.key)
            continue
        logger.done("Loaded evaluation artifacts", t)

        t2 = logger.step(f"Visualization plotting: {spec.key}")
        n = run_visualization_from_saved(eval_root, factor_label=spec.key)
        logger.done("Plots generated", t2, extra=f"n={n}, path={eval_root / 'plots'}")
        generated.append(spec.key)

    return {"selected": [s.key for s in selected], "generated": generated, "skipped": skipped}
