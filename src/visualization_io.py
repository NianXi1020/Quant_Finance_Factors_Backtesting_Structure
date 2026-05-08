from __future__ import annotations

from pathlib import Path
import pandas as pd


def safe_read_csv_or_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.DataFrame()


def find_artifact_file(base_dir: Path, canonical: list[str], keywords: list[str], required_cols: list[str] | None = None) -> tuple[Path | None, str]:
    for name in canonical:
        p = base_dir / name
        if p.exists():
            df = safe_read_csv_or_parquet(p)
            if required_cols and not set(required_cols).issubset(df.columns):
                continue
            return p, "canonical"

    candidates = [p for p in base_dir.glob("**/*") if p.is_file() and p.suffix.lower() in {".csv", ".parquet", ".pq"}]
    candidates = [p for p in candidates if any(k.lower() in p.name.lower() for k in keywords)]
    valid = []
    for p in candidates:
        df = safe_read_csv_or_parquet(p)
        if required_cols and not set(required_cols).issubset(df.columns):
            continue
        valid.append(p)
    if not valid:
        return None, "missing"
    valid = sorted(valid, key=lambda p: p.stat().st_mtime, reverse=True)
    return valid[0], "keyword"


def load_ic_outputs(eval_root: Path) -> dict:
    p, mode = find_artifact_file(eval_root / "ic_analysis", ["ic_timeseries.csv"], ["ic"], ["date", "ic"])
    rp, _ = find_artifact_file(eval_root / "ic_analysis", ["rank_ic_timeseries.csv"], ["rank"], ["date", "rank_ic"])
    return {"ic_path": p, "rank_ic_path": rp, "mode": mode}


def load_quantile_outputs(eval_root: Path) -> dict:
    qp, mode = find_artifact_file(eval_root / "quantile_backtest", ["quantile_returns.csv"], ["quantile", "return"], ["date"])
    cp, _ = find_artifact_file(eval_root / "quantile_backtest", ["quantile_counts.csv"], ["count", "coverage"], ["date"])
    return {"quantile_path": qp, "count_path": cp, "mode": mode}


def load_summary_outputs(eval_root: Path) -> dict:
    sp, mode = find_artifact_file(eval_root / "summaries", ["summary_metrics.csv"], ["summary", "metric"], None)
    return {"summary_path": sp, "mode": mode}


def load_processed_factor_file(processed_factor_path: Path | None) -> pd.DataFrame:
    if processed_factor_path is None:
        return pd.DataFrame()
    return safe_read_csv_or_parquet(processed_factor_path)


def load_ml_diagnostics(eval_root: Path) -> dict:
    fi, _ = find_artifact_file(eval_root / "summaries", ["feature_importance.csv"], ["feature", "importance"], None)
    pca, _ = find_artifact_file(eval_root / "summaries", ["explained_variance.csv"], ["explained", "variance"], None)
    return {"feature_importance_path": fi, "explained_variance_path": pca}
