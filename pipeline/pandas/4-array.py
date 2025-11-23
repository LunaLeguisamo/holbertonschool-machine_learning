#!/usr/bin/env python3
"""
Function that takes a pd.DataFrame as
input and performs the following
"""


def array(df):
    """
    df is a pd.DataFrame containing columns
    named High and Close.
    The function should select the last 10 rows
    of the High and Close columns.
    Convert these selected values into a numpy.ndarray.
    Returns: the numpy.
    """
    df = df[["High", "Close"]].tail(10).to_numpy()
    return df
