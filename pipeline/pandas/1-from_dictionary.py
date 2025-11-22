#!/usr/bin/env python3
"""
Write a python script that creates a pd.DataFrame from a dictionary
"""

import pandas as pd

"""
The first column should be labeled First and have the values
0.0, 0.5, 1.0, and 1.5
The second column shoud be labeled Second and have the values
one, two, three, four
"""
data = {
    "First": [0.0, 0.5, 1.0, 1.5],
    "Second": ["one", "two", "three", "four"]
}

"""
The rows should be labeled A, B, C, and D, respectively
"""
index = ["A", "B", "C", "D"]

"""
The pd.DataFrame should be saved into the variable df
"""
df = pd.DataFrame(data, index=index)
