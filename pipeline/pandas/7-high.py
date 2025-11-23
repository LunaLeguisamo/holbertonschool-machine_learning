#!/usr/bin/env python3
"""
Function that takes a pd.DataFrame and:
Sorts it by the High price in descending order.
"""


def high(df):
    """
    Returns: the sorted pd.DataFrame.
    """
    df = df.sort_values(by="High", ascending=False)
    return df
