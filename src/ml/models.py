from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression


def build_model(model_name: str):
    if model_name == "rf_return":
        return RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=50, max_features="sqrt", n_jobs=-1, random_state=42)
    if model_name == "rank_gbdt":
        return HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=31, learning_rate=0.05, l2_regularization=1.0, random_state=42)
    if model_name == "logit_top_bottom":
        return LogisticRegression(penalty="l2", C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
    raise ValueError(f"Unsupported model_name: {model_name}")
