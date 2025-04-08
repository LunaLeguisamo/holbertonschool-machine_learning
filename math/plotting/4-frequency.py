#!/usr/bin/env python3

"""
Frequency
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    The x-axis should be labeled Grades
    The y-axis should be labeled Number of Students
    The x-axis should have bins every 10 units
    The title should be Project A
    The bars should be outlined in black
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 110, 10))
    plt.ylim(0, 30)
    plt.hist(student_grades, bins=range(0, 110, 10), edgecolor="black")
    plt.show()
