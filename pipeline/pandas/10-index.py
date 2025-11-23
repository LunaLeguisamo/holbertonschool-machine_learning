#!/usr/bin/env python3
"""
Function that takes a pd.DataFrame and
Sets the Timestamp column as the index
of the dataframe.
"""


def index(df):
    """
    Returns: the modified pd.DataFrame.
    """
    df.set_index("Timestamp", inplace=True)
    return df
