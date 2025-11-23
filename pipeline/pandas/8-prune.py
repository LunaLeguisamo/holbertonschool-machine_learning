#!/usr/bin/env python3
"""
Function that takes a pd.DataFrame and
Removes any entries where Close has NaN values.
"""


def prune(df):
    """Returns: the modified pd.DataFrame."""
    df = df.dropna(subset=["Close"])
    return df
