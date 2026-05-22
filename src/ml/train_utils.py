from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RollingConfig:
    train_window: int = 252
    min_train_dates: int = 126
    min_train_rows: int = 5000
    rebalance_freq: str = "daily"


def iter_prediction_dates(unique_dates: list[pd.Timestamp], rebalance_freq: str = "daily") -> list[pd.Timestamp]:
    if rebalance_freq == "daily":
        return unique_dates
    if rebalance_freq == "weekly":
        return [d for i, d in enumerate(unique_dates) if i % 5 == 0]
    if rebalance_freq == "monthly":
        out = []
        seen = set()
        for d in unique_dates:
            ym = (d.year, d.month)
            if ym not in seen:
                out.append(d)
                seen.add(ym)
        return out
    raise ValueError("rebalance_freq must be daily/weekly/monthly")


def rolling_train_predict(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model,
    cfg: RollingConfig,
) -> pd.DataFrame:
    dates = sorted(pd.to_datetime(df["date"]).dropna().unique())
    pred_dates = iter_prediction_dates(dates, cfg.rebalance_freq)
    out = []

    for d in pred_dates:
        idx = dates.index(d)
        if idx < cfg.min_train_dates:
            continue
        start = max(0, idx - cfg.train_window)
        train_dates = dates[start:idx]
        train = df[df["date"].isin(train_dates)].dropna(subset=[target_col]).copy()
        pred = df[df["date"] == d].copy()
        if len(train) < cfg.min_train_rows or pred.empty:
            continue

        X_train = train[feature_cols]
        y_train = train[target_col]
        X_pred = pred[feature_cols]

        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X_pred)[:, 1]
        else:
            score = model.predict(X_pred)

        pred_out = pred[["date", "ts_code", "stock_code"]].copy()
        pred_out["ml_score"] = score
        out.append(pred_out)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["date", "ts_code", "stock_code", "ml_score"])
