from __future__ import annotations

import numpy as np
import pandas as pd


def add_top_bottom_label(df: pd.DataFrame, target_col: str = "fwd_1d_return", top_q: float = 0.7, bottom_q: float = 0.3) -> pd.DataFrame:
    out = df.copy()
    q_high = out.groupby("date")[target_col].transform(lambda s: s.quantile(top_q))
    q_low = out.groupby("date")[target_col].transform(lambda s: s.quantile(bottom_q))
    out["tb_label"] = np.where(out[target_col] >= q_high, 1, np.where(out[target_col] <= q_low, 0, np.nan))
    return out


def add_cross_sectional_rank_target(df: pd.DataFrame, target_col: str = "fwd_1d_return", rank_col: str = "fwd_return_rank_pct") -> pd.DataFrame:
    out = df.copy()
    out[rank_col] = out.groupby("date")[target_col].rank(pct=True)
    return out
