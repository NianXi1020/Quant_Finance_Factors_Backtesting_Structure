from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.config import PipelineConfig
from src.factor_registry import FactorSpec
from src.factors.basic import preprocess_single_factor
from src.ml.build_ml_dataset import build_ml_panel
from src.ml.diagnostics import save_feature_importance
from src.ml.models import build_model
from src.ml.pca_factor import rolling_pca_factor
from src.ml.ranking import add_cross_sectional_rank_target, add_top_bottom_label
from src.ml.train_utils import RollingConfig, rolling_train_predict
from src.orchestrator import _run_one_factor, _prepare_shared_artifacts
from src.utils.progress import StepLogger


def _score_to_factor(df_score: pd.DataFrame, industry_df: pd.DataFrame) -> pd.DataFrame:
    merged = df_score.merge(industry_df[["date", "stock_code", "industry"]], on=["date", "stock_code"], how="left")
    merged = merged.rename(columns={"ml_score": "factor_raw"})
    merged = preprocess_single_factor(merged, raw_col="factor_raw")
    keep = ["date", "ts_code", "stock_code", "factor_raw", "factor_win", "factor_z", "factor_indneu"]
    return merged[keep].sort_values(["date", "stock_code"]).reset_index(drop=True)


def run_ml_models(models: list[str], force: bool = False) -> None:
    cfg = PipelineConfig()
    logger = StepLogger(enabled=True, verbose=True)

    shared = _prepare_shared_artifacts(cfg, logger)
    factor_names = ["rs_30", "rs_60", "rs_90", "rs_180", "hl_30", "hl_60", "hl_90", "hl_180", "vol_30", "vol_60", "vol_90", "vol_180", "turnover_30", "turnover_60", "turnover_90", "turnover_180", "improved_mom_30", "improved_mom_60", "improved_mom_90", "improved_mom_180", "macd"]
    panel = build_ml_panel(factor_names=factor_names, feature_version="factor_z", use_cache=True, force=force, cfg=cfg)

    feature_cols = [c for c in panel.columns if c not in {"date", "ts_code", "stock_code", "in_universe", "fwd_1d_return"}]
    rcfg = RollingConfig()
    industry_df = shared["universe"][["date", "stock_code", "industry"]].drop_duplicates(["date", "stock_code"])

    for model_name in models:
        if model_name == "pca":
            scores, ev = rolling_pca_factor(panel, feature_cols=feature_cols, cfg=rcfg, n_components=3)
            ev_path = cfg.storage.outputs_root / "single_factor" / "ml" / "ml_pca" / "summaries" / "explained_variance.csv"
            ev_path.parent.mkdir(parents=True, exist_ok=True)
            ev.to_csv(ev_path, index=False)
            for key, score_df in scores.items():
                fac = _score_to_factor(score_df, industry_df)
                fpath = cfg.storage.processed_root / "factors" / "ml" / f"{key}.parquet"
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fac.to_parquet(fpath, index=False)
            continue

        train_df = panel.copy()
        target_col = "fwd_1d_return"
        if model_name == "logit_top_bottom":
            train_df = add_top_bottom_label(train_df)
            train_df = train_df.dropna(subset=["tb_label"]).copy()
            target_col = "tb_label"
        elif model_name == "rank_gbdt":
            train_df = add_cross_sectional_rank_target(train_df)
            target_col = "fwd_return_rank_pct"

        model = build_model(model_name)
        pred = rolling_train_predict(train_df, feature_cols=feature_cols, target_col=target_col, model=model, cfg=rcfg)
        fac = _score_to_factor(pred, industry_df)

        factor_key = f"ml_{model_name}"
        if model_name == "rf_return":
            factor_key = "ml_rf_return"
        elif model_name == "logit_top_bottom":
            factor_key = "ml_logit_top_bottom"
        elif model_name == "rank_gbdt":
            factor_key = "ml_rank_gbdt"

        fpath = cfg.storage.processed_root / "factors" / "ml" / f"{factor_key}.parquet"
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fac.to_parquet(fpath, index=False)

        if hasattr(model, "feature_importances_"):
            imp_path = cfg.storage.outputs_root / "single_factor" / "ml" / factor_key / "summaries" / "feature_importance.csv"
            save_feature_importance(imp_path, "rolling", feature_cols, model.feature_importances_)

        meta = {
            "model_name": model_name,
            "feature_version": "factor_z",
            "train_window": rcfg.train_window,
            "target_type": target_col,
            "rebalance_freq": rcfg.rebalance_freq,
            "created_at": pd.Timestamp.utcnow().isoformat(),
        }
        mpath = cfg.storage.outputs_root / "single_factor" / "ml" / factor_key / "run_metadata.json"
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Run ML factor extension")
    p.add_argument("--model", choices=["rf_return", "logit_top_bottom", "rank_gbdt", "pca"], default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.all:
        models = ["rf_return", "logit_top_bottom", "rank_gbdt", "pca"]
    elif args.model:
        models = [args.model]
    else:
        raise ValueError("Please specify --model or --all")

    run_ml_models(models=models, force=args.force)


if __name__ == "__main__":
    main()
