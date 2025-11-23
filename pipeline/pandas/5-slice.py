#!/usr/bin/env python3
"""
Function that takes a pd.DataFrame
"""


def slice(df):
    """
    Extracts the columns High, Low, Close, and Volume_BTC.
    Selects every 60th row from these columns.
    Returns: the sliced pd.DataFrame
    """
    df = df[["High", "Low", "Close", "Volume_(BTC)"]][::60]
    return df
