#!/usr/bin/env python3
"""
Function that takes a pd.DataFrame
"""


def flip_switch(df):
    """
    Sorts the data in reverse
    chronological order.
    Transposes the sorted dataframe.
    Returns: the transformed pd.DataFrame.
    """
    df = df.sort_index(ascending=False).transpose()
    return df
