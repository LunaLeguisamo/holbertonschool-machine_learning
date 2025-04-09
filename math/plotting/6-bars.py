#!/usr/bin/env python3

"""
Function where do bar plots
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    We used np.arange to obtain de values
    of the quantity of fruits for each person,
    and plot this with 4 bar plots.
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    x = np.arange(3)
    plt.bar(x, fruit[0], color='r', label='apples', width=0.5)
    plt.bar(x, fruit[1], bottom=fruit[0], color='yellow',
            label='bananas', width=0.5)
    plt.bar(x, fruit[2], bottom=fruit[0]+fruit[1],
            color='#ff8000', label='oranges', width=0.5)
    plt.bar(x, fruit[3], bottom=fruit[0] + fruit[1] + fruit[2],
            color='#ffe5b4', label='peaches', width=0.5)

    plt.ylim(0, 80)
    plt.xticks(x, ['Farrah', 'Fred', 'Felicia'])
    plt.yticks(np.arange(0, 90, 10))

    plt.ylabel("Quantity of Fruit")
    plt.title("Number of Fruit per Person")
    plt.legend()
    plt.show()
