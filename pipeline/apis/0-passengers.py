#!/usr/bin/env python3
"""
By using the Swapi API, create a method that
returns the list of ships that can hold a given
number of passengers
"""
import requests


def availableShips(passengerCount):
    """Returns a list of ship names that can hold at least"""

    ships = []
    url = "https://swapi.dev/api/starships/"
    while url:
        response = requests.get(url)
        data = response.json()
        for ship in data['results']:
            if ship['passengers'] != 'n/a' and ship['passengers'] != 'unknown':
                # Remove commas and convert to int
                capacity = int(ship['passengers'].replace(',', ''))
                if capacity >= passengerCount:
                    ships.append(ship['name'])
        url = data['next']
    return ships
