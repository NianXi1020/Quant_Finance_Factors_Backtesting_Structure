from __future__ import annotations

import pandas as pd


def build_alive_mask(dates: pd.Series, delist_date: pd.Series) -> pd.Series:
    """Return True when a stock is still alive on each `date`.

    Convention used here:
    - if delist_date is missing -> always alive
    - otherwise alive strictly before delist_date
    """
    return delist_date.isna() | (dates < delist_date)


def attach_universe_flags(
    daily_df: pd.DataFrame,
    stock_meta: pd.DataFrame,
    delist_df: pd.DataFrame,
    min_listing_days: int = 60,
) -> pd.DataFrame:
    """Merge metadata and compute universe eligibility flags.

    No look-ahead assumptions:
    - uses only date-local info (`date`, static `ipo_date`, static `delist_date`)
    - does not use future returns or future classifications.
    """
    df = daily_df.merge(
        stock_meta[["stock_code", "industry", "ipo_date"]], on="stock_code", how="left"
    ).merge(delist_df[["stock_code", "delist_date"]], on="stock_code", how="left")

    df["is_alive"] = build_alive_mask(df["date"], df["delist_date"])
    listing_age = (df["date"] - df["ipo_date"]).dt.days
    df["is_old_enough"] = listing_age >= min_listing_days
    df["in_universe"] = df["is_alive"] & df["is_old_enough"]

    return df
