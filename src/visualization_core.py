from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_yearly_return_bar(qret: pd.DataFrame, out: Path, annualization_days: int = 252):
    if qret.empty or "long_short" not in qret.columns:
        return False
    s = qret[["date", "long_short"]].dropna().copy()
    s["date"] = pd.to_datetime(s["date"])
    yr = s.set_index("date")["long_short"].groupby(lambda x: x.year).apply(lambda x: (1 + x).prod() - 1)
    fig = plt.figure(figsize=(9, 4))
    plt.bar(yr.index.astype(str), yr.values)
    plt.title("Yearly Long-Short Return (Compounded)")
    _save(fig, out)
    return True


def plot_rolling_ic(ic: pd.DataFrame, ric: pd.DataFrame, out: Path, window: int = 60):
    if ic.empty and ric.empty:
        return False
    fig = plt.figure(figsize=(10, 4))
    if not ic.empty:
        ic = ic.copy(); ic["date"] = pd.to_datetime(ic["date"])
        plt.plot(ic["date"], ic["ic"].rolling(window).mean(), label=f"IC MA{window}")
    if not ric.empty:
        ric = ric.copy(); ric["date"] = pd.to_datetime(ric["date"])
        plt.plot(ric["date"], ric["rank_ic"].rolling(window).mean(), label=f"RankIC MA{window}")
    plt.axhline(0, color="black", lw=1)
    plt.legend(); plt.title("Rolling IC / RankIC")
    _save(fig, out)
    return True


def plot_feature_importance(fi: pd.DataFrame, out: Path):
    if fi.empty or "feature" not in fi.columns:
        return False
    if "importance" not in fi.columns:
        return False
    g = fi.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False).head(15)
    fig = plt.figure(figsize=(10, 5))
    plt.barh(g["feature"], g["importance"])
    plt.gca().invert_yaxis(); plt.title("Feature Importance (Top 15)")
    _save(fig, out)
    return True


def plot_pca_explained_variance(ev: pd.DataFrame, out: Path):
    if ev.empty or "component" not in ev.columns or "explained_variance_ratio" not in ev.columns:
        return False
    fig = plt.figure(figsize=(10, 4))
    if "train_end_date" in ev.columns:
        ev = ev.copy(); ev["train_end_date"] = pd.to_datetime(ev["train_end_date"])
        for c, g in ev.groupby("component"):
            plt.plot(g["train_end_date"], g["explained_variance_ratio"], label=f"PC{c}")
        plt.legend()
    else:
        agg = ev.groupby("component", as_index=False)["explained_variance_ratio"].mean()
        plt.bar(agg["component"].astype(str), agg["explained_variance_ratio"])
    plt.title("PCA Explained Variance")
    _save(fig, out)
    return True
