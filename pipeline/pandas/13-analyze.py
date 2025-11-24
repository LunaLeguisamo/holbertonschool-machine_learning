#!/usr/bin/env python3
"""
Function that takes a pd.DataFrame and:
Computes descriptive statistics for all columns except the Timestamp column.
Returns a new pd.DataFrame containing these statistics.
"""
import pandas as pd


def analyze(df):
    """
    Returns: a new pd.DataFrame containing descriptive statistics.
    """
    stats = df.drop(columns=["Timestamp"]).describe()
    return stats
