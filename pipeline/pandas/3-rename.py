#!/usr/bin/env python3
"""
Function that takes a pd.DataFrame
as input and performs the following
"""
import pandas as pd


def rename(df):
    """
    df is a pd.DataFrame containing a column named Timestamp.
    The function should rename the Timestamp column to Datetime.
    Convert the timestamp values to datatime values
    Display only the Datetime and Close column
    Returns: the modified pd.DataFrame
    """
    df.rename(columns={"Timestamp": "Datetime"}, inplace=True)
    pd.to_datetime(df["Datetime"])
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df[["Datetime", "Close"]]
    return df
