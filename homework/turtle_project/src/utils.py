# src/utils.py
from __future__ import annotations
import pandas as pd

def get_summary_stats(df: pd.DataFrame, groupby_col: str | None = None) -> dict:
    """
    Compute numeric .describe() and optional groupby aggregation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    groupby_col : str | None
        Optional column to group by. Must exist in df.

    Returns
    -------
    dict
        {"describe": DataFrame, "groupby": Optional[DataFrame]}
    """
    numeric_desc = df.select_dtypes(include="number").describe().T

    group_df = None
    if groupby_col is not None and groupby_col in df.columns:
        # Example aggregation: mean of numeric columns
        group_df = df.groupby(groupby_col).mean(numeric_only=True)

    return {"describe": numeric_desc, "groupby": group_df}
