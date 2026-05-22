from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PipelineConfig


def _factor_path(cfg: PipelineConfig, factor_name: str) -> Path:
    if factor_name == "macd":
        return cfg.storage.processed_root / "factors" / "momentum" / "macd.parquet"
    return cfg.storage.processed_root / "factors" / "momentum" / f"{factor_name}.parquet"


def build_ml_panel(
    factor_names: list[str],
    feature_version: str = "factor_z",
    use_cache: bool = True,
    force: bool = False,
    cfg: PipelineConfig | None = None,
) -> pd.DataFrame:
    """Build reusable ML feature panel from existing factor files.

    Defaults to factor_z features (recommended in practical guide).
    """
    if feature_version not in {"factor_z", "factor_indneu"}:
        raise ValueError("feature_version must be one of {'factor_z','factor_indneu'}")

    cfg = cfg or PipelineConfig()
    cache_path = cfg.storage.processed_root / "ml" / "ml_feature_panel.parquet"
    if use_cache and (not force) and cache_path.exists():
        return pd.read_parquet(cache_path)

    merged: pd.DataFrame | None = None
    feature_cols: list[str] = []

    for fname in factor_names:
        fpath = _factor_path(cfg, fname)
        if not fpath.exists():
            raise FileNotFoundError(f"Factor file not found: {fpath}")
        fdf = pd.read_parquet(fpath)

        if fname == "macd":
            # Use three MACD members as separate features.
            cols = ["date", "ts_code", "stock_code", "dif_z", "dea_z", "macd_bar_z", "dif_indneu", "dea_indneu", "macd_bar_indneu"]
            keep = [c for c in cols if c in fdf.columns]
            fdf = fdf[keep].copy()
            if feature_version == "factor_z":
                ren = {"dif_z": "macd_dif", "dea_z": "macd_dea", "macd_bar_z": "macd_bar"}
            else:
                ren = {"dif_indneu": "macd_dif", "dea_indneu": "macd_dea", "macd_bar_indneu": "macd_bar"}
            fdf = fdf.rename(columns=ren)
            keep2 = ["date", "ts_code", "stock_code", *ren.values()]
            fdf = fdf[keep2]
            feature_cols.extend(list(ren.values()))
        else:
            src_col = feature_version
            if src_col not in fdf.columns:
                raise KeyError(f"Missing {src_col} in {fpath}")
            out_col = fname
            fdf = fdf[["date", "ts_code", "stock_code", src_col]].rename(columns={src_col: out_col})
            feature_cols.append(out_col)

        merged = fdf if merged is None else merged.merge(fdf, on=["date", "ts_code", "stock_code"], how="outer")

    if merged is None:
        raise ValueError("No factors provided for ML panel")

    universe = pd.read_parquet(cfg.storage.universe_basic)[["date", "stock_code", "in_universe"]]
    labels = pd.read_parquet(cfg.storage.forward_returns_1d)[["date", "ts_code", "stock_code", "fwd_1d_return"]]

    df = merged.merge(universe, on=["date", "stock_code"], how="left").merge(labels, on=["date", "ts_code", "stock_code"], how="left")
    df = df.loc[df["in_universe"] == True].copy()
    df = df.dropna(subset=["fwd_1d_return"]).sort_values(["date", "stock_code"]).reset_index(drop=True)

    # Same-date cross-sectional median imputation.
    for c in feature_cols:
        if c not in df.columns:
            continue
        med = df.groupby("date")[c].transform("median")
        df[c] = df[c].fillna(med)
        df[c] = df[c].fillna(0.0)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    print(f"[ML DATA] rows={len(df):,}, cols={df.shape[1]}, n_dates={df['date'].nunique():,}, n_stocks={df['stock_code'].nunique():,}")
    print(f"[ML DATA] date range: {df['date'].min()} to {df['date'].max()}")
    print(f"[ML DATA] missing feature ratio={df[feature_cols].isna().mean().mean():.6f}")

    return df
