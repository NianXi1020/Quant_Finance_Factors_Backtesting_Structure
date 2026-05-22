from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def plot_ls_nav(qret: pd.DataFrame, out: Path, title: str) -> None:
    if qret.empty or "long_short" not in qret.columns:
        return
    nav_ls = (1 + qret["long_short"].fillna(0)).cumprod()
    nav_long = (1 + qret.get("long_leg", pd.Series(0.0, index=qret.index)).fillna(0)).cumprod()
    nav_short = (1 - qret.get("short_leg", pd.Series(0.0, index=qret.index)).fillna(0)).cumprod()

    plt.figure(figsize=(10, 5))
    plt.plot(qret["date"], nav_ls, label="Long-Short NAV", linewidth=2)
    plt.plot(qret["date"], nav_long, label="Long Leg NAV", linewidth=1.5)
    plt.plot(qret["date"], nav_short, label="Short Leg NAV", linewidth=1.5)
    plt.title(f"{title} - Long/Short NAV")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "ls_nav.png", dpi=150)
    plt.close()


def plot_quantile_nav(qret: pd.DataFrame, out: Path, title: str) -> None:
    qcols = sorted([c for c in qret.columns if c.startswith("Q") and "_" not in c], key=lambda x: int(x[1:]))
    if qret.empty or not qcols:
        return
    plt.figure(figsize=(10, 5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(qcols)))
    for c, color in zip(qcols, cmap):
        nav = (1 + qret[c].fillna(0)).cumprod()
        plt.plot(qret["date"], nav, label=c, color=color, linewidth=1)
    plt.title(f"{title} - Quantile NAV")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "quantile_nav.png", dpi=150)
    plt.close()


def plot_quantile_bar(qret: pd.DataFrame, out: Path, title: str) -> None:
    qcols = sorted([c for c in qret.columns if c.startswith("Q") and "_" not in c], key=lambda x: int(x[1:]))
    if qret.empty or not qcols:
        return
    means = [qret[c].mean() for c in qcols]
    labels = qcols.copy()
    vals = means.copy()
    if "long_short" in qret.columns:
        labels.append("LS")
        vals.append(qret["long_short"].mean())

    plt.figure(figsize=(10, 4))
    plt.bar(labels, vals)
    plt.title(f"{title} - Mean Quantile Return")
    plt.xlabel("Quantile")
    plt.ylabel("Mean Return")
    plt.tight_layout()
    plt.savefig(out / "quantile_bar.png", dpi=150)
    plt.close()


def plot_ls_drawdown(qret: pd.DataFrame, out: Path, title: str) -> None:
    if qret.empty or "long_short" not in qret.columns:
        return
    nav = (1 + qret["long_short"].fillna(0)).cumprod()
    dd = nav / nav.cummax() - 1
    plt.figure(figsize=(10, 4))
    plt.plot(qret["date"], dd, color="crimson")
    plt.title(f"{title} - Long/Short Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(out / "ls_drawdown.png", dpi=150)
    plt.close()


def plot_rolling_metrics(qret: pd.DataFrame, out: Path, title: str, window: int = 126) -> None:
    if qret.empty or "long_short" not in qret.columns:
        return
    r = qret["long_short"].fillna(0)
    roll_mean = r.rolling(window).mean()
    roll_std = r.rolling(window).std(ddof=0)
    roll_sharpe = np.where((roll_std == 0) | roll_std.isna(), np.nan, (roll_mean / roll_std) * np.sqrt(252))
    roll_ann_return = roll_mean * 252

    plt.figure(figsize=(10, 4))
    plt.plot(qret["date"], roll_sharpe)
    plt.title(f"{title} - Rolling Sharpe ({window}D)")
    plt.xlabel("Date")
    plt.ylabel("Sharpe")
    plt.tight_layout()
    plt.savefig(out / "rolling_sharpe.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(qret["date"], roll_ann_return)
    plt.title(f"{title} - Rolling Annualized Return ({window}D)")
    plt.xlabel("Date")
    plt.ylabel("Annualized Return")
    plt.tight_layout()
    plt.savefig(out / "rolling_return.png", dpi=150)
    plt.close()


def plot_monthly_heatmap(qret: pd.DataFrame, out: Path, title: str) -> None:
    if qret.empty or "long_short" not in qret.columns:
        return
    m = qret[["date", "long_short"]].dropna().copy()
    if m.empty:
        return
    m = m.set_index("date").resample("M").apply(lambda x: (1 + x).prod() - 1)
    m["year"] = m.index.year
    m["month"] = m.index.month
    piv = m.pivot(index="year", columns="month", values="long_short")

    plt.figure(figsize=(12, max(3, 0.4 * len(piv))))
    im = plt.imshow(piv.values, aspect="auto", cmap="RdYlGn", interpolation="nearest")
    plt.colorbar(im, label="Monthly Return")
    plt.xticks(range(12), [str(i) for i in range(1, 13)])
    plt.yticks(range(len(piv.index)), piv.index.astype(str))
    plt.title(f"{title} - Monthly Long/Short Return Heatmap")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.tight_layout()
    plt.savefig(out / "monthly_heatmap.png", dpi=150)
    plt.close()


def plot_return_distribution(qret: pd.DataFrame, out: Path, title: str) -> None:
    if qret.empty or "long_short" not in qret.columns:
        return
    s = qret["long_short"].dropna()
    if s.empty:
        return
    plt.figure(figsize=(8, 4))
    plt.hist(s, bins=50, alpha=0.8, density=True)
    plt.title(f"{title} - Long/Short Return Distribution")
    plt.xlabel("Return")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(out / "return_distribution.png", dpi=150)
    plt.close()


def plot_ic_timeseries(ic_df: pd.DataFrame, out: Path, title: str, roll_windows: tuple[int, int] = (20, 60)) -> None:
    if ic_df.empty or "ic" not in ic_df.columns:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(ic_df["date"], ic_df["ic"], alpha=0.5, label="IC")
    for w in roll_windows:
        plt.plot(ic_df["date"], ic_df["ic"].rolling(w).mean(), label=f"IC MA{w}")
    plt.title(f"{title} - IC Time Series")
    plt.xlabel("Date")
    plt.ylabel("IC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "ic_timeseries.png", dpi=150)
    plt.close()


def plot_ic_distribution(ic_df: pd.DataFrame, out: Path, title: str) -> None:
    if ic_df.empty or "ic" not in ic_df.columns:
        return
    s = ic_df["ic"].dropna()
    if s.empty:
        return
    plt.figure(figsize=(8, 4))
    plt.hist(s, bins=40, alpha=0.8, density=True)
    plt.title(f"{title} - IC Distribution")
    plt.xlabel("IC")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(out / "ic_distribution.png", dpi=150)
    plt.close()


def plot_turnover_placeholder(out: Path, title: str) -> None:
    plt.figure(figsize=(8, 3))
    plt.text(0.5, 0.5, "TODO: Portfolio turnover series not implemented yet", ha="center", va="center")
    plt.axis("off")
    plt.title(f"{title} - Turnover")
    plt.tight_layout()
    plt.savefig(out / "turnover.png", dpi=150)
    plt.close()


def plot_coverage(count_df: pd.DataFrame, out: Path, title: str) -> None:
    if count_df.empty:
        return
    plt.figure(figsize=(10, 4))
    if "valid_n" in count_df.columns:
        plt.plot(count_df["date"], count_df["valid_n"], label="valid sample size", linewidth=2)
    qn_cols = [c for c in count_df.columns if c.endswith("_n")]
    for c in qn_cols:
        plt.plot(count_df["date"], count_df[c], alpha=0.5, linewidth=1, label=c)
    plt.title(f"{title} - Quantile Coverage")
    plt.xlabel("Date")
    plt.ylabel("Count")
    if qn_cols:
        plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "coverage.png", dpi=150)
    plt.close()


def generate_all_evaluation_plots(eval_root: Path, factor_label: str, rolling_window: int = 126) -> None:
    plots_dir = eval_root / "plots"
    _ensure_dir(plots_dir)

    qret = _load_csv(eval_root / "quantile_backtest" / "quantile_returns.csv")
    ic_df = _load_csv(eval_root / "ic_analysis" / "ic_timeseries.csv")
    count_df = _load_csv(eval_root / "quantile_backtest" / "quantile_counts.csv")

    plot_ls_nav(qret, plots_dir, factor_label)
    plot_quantile_nav(qret, plots_dir, factor_label)
    plot_quantile_bar(qret, plots_dir, factor_label)
    plot_ls_drawdown(qret, plots_dir, factor_label)
    plot_rolling_metrics(qret, plots_dir, factor_label, window=rolling_window)
    plot_monthly_heatmap(qret, plots_dir, factor_label)
    plot_return_distribution(qret, plots_dir, factor_label)

    plot_ic_timeseries(ic_df, plots_dir, factor_label)
    plot_ic_distribution(ic_df, plots_dir, factor_label)

    plot_turnover_placeholder(plots_dir, factor_label)
    plot_coverage(count_df, plots_dir, factor_label)
