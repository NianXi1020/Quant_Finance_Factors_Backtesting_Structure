from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.ml.train_utils import RollingConfig, iter_prediction_dates


def rolling_pca_factor(df: pd.DataFrame, feature_cols: list[str], cfg: RollingConfig, n_components: int = 3) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    dates = sorted(pd.to_datetime(df["date"]).dropna().unique())
    pred_dates = iter_prediction_dates(dates, cfg.rebalance_freq)

    comp_rows = {k: [] for k in range(1, n_components + 1)}
    ev_rows = []

    for d in pred_dates:
        idx = dates.index(d)
        if idx < cfg.min_train_dates:
            continue
        start = max(0, idx - cfg.train_window)
        train_dates = dates[start:idx]

        train = df[df["date"].isin(train_dates)]
        pred = df[df["date"] == d]
        if len(train) < cfg.min_train_rows or pred.empty:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[feature_cols])
        X_pred = scaler.transform(pred[feature_cols])
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X_train)
        Z = pca.transform(X_pred)

        for k in range(n_components):
            tmp = pred[["date", "ts_code", "stock_code"]].copy()
            tmp["ml_score"] = Z[:, k]
            comp_rows[k + 1].append(tmp)
            ev_rows.append({"train_end_date": d, "component": k + 1, "explained_variance_ratio": float(pca.explained_variance_ratio_[k])})

    out = {
        f"ml_pca_{k}": pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "ts_code", "stock_code", "ml_score"])
        for k, rows in comp_rows.items()
    }
    return out, pd.DataFrame(ev_rows)
