#!/usr/bin/env python3
"""Function that takes a pd.DataFrame and visualizes it."""
import pandas as pd
import matplotlib.pyplot as plt


def visualize(df):
    """Returns: the transformed pd.DataFrame before plotting."""
    df = df.drop(columns=["Weighted_Price"])
    df.rename(columns={"Timestamp": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], unit='s')
    df.set_index("Date", inplace=True)
    df["Close"] = df["Close"].fillna(method="ffill")
    df["High"] = df["High"].fillna(df["Close"])
    df["Low"] = df["Low"].fillna(df["Close"])
    df["Open"] = df["Open"].fillna(df["Close"])
    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)
    df = df[df.index.year >= 2017]
    df = df.resample('D').agg({
        'High': 'max',
        'Low': 'min',
        'Open': 'mean',
        'Close': 'mean',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum'
    })
    df.plot(y=['High', 'Low', 'Open', 'Close'], title='Daily OHLC from 2017')
    plt.show()
    return df
