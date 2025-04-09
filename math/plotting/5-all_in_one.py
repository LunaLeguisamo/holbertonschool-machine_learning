#!/usr/bin/env python3
"""

"""
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """
    """
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(7, 6))

    # Primer subplot
    ax0 = plt.subplot2grid((3, 2), (0, 0))
    ax0.plot(y0, color='r')
    ax0.set_xlim(0, 10)

    # Segundo subplot
    ax1 = plt.subplot2grid((3, 2), (0, 1))
    ax1.set_title("Men's Height vs Weight", fontsize="x-small")
    ax1.set_xlabel("Height(in)", fontsize="x-small")
    ax1.set_ylabel("Weight(lbs)", fontsize="x-small")
    ax1.scatter(x1, y1, color='purple', s=8)
    ax1.set_xticks(np.arange(60, 90, 10))

    # Tercer subplot
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    ax2.set_title("Exponential Decay of C-14", fontsize="x-small")
    ax2.set_xlabel("Time (years)", fontsize="x-small")
    ax2.set_ylabel("Fraction Remaining", fontsize="x-small")
    ax2.set_xticks(np.arange(0, 28651, 10000))
    ax2.plot(x2, y2, color='skyblue')
    ax2.set_xlim(0, 28651)
    ax2.set_yscale('log')

    # Cuarto subplot
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    ax3.set_title("Exponential Decay of Radioactive Elements",
                  fontsize="x-small")
    ax3.set_xlabel("Time (years)", fontsize="x-small")
    ax3.set_ylabel("Fraction Remaining", fontsize="x-small")
    ax3.plot(x3, y31, label='C-14', linestyle='--', color='r')
    ax3.plot(x3, y32, label='Ra-226', color="green")
    ax3.set_xlim(0, 20000)
    ax3.legend(fontsize="x-small")
 
    # Quinto subplot
    ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    ax4.set_title("Project A", fontsize="x-small")
    ax4.set_xlabel("Grades", fontsize="x-small")
    ax4.set_ylabel("Number of Students", fontsize="x-small")
    ax4.hist(student_grades, bins=range(0, 110, 10), edgecolor="black")
    ax4.set_xticks(np.arange(0, 110, 10))
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 30)

    plt.suptitle("All in One")
    plt.tight_layout()
    plt.show()
