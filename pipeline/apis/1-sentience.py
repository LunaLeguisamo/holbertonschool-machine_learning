#!/usr/bin/env python3
"""
By using the Swapi API, create a method that
returns the list of names of the home planets
of all sentient species.
"""
import requests


def sentientPlanets():
    """
    Fetches the list of names of the home planets of all sentient species.
    Sentient species are those whose 'classification' or 'designation'
    include a variation of 'sentient'.
    """
    url = "https://swapi-api.alx-tools.com/api/species/"
    planets = set()

    while url:
        response = requests.get(url)
        data = response.json()

        for species in data.get("results", []):
            classification = species.get("classification", "").lower()
            designation = species.get("designation", "").lower()

            # Identify if species is sentient
            if "sentient" in classification or "sentient" in designation:
                home = species.get("homeworld")
                if home:
                    # Fetch homeworld name
                    home_data = requests.get(home).json()
                    name = home_data.get("name")
                    if name:
                        planets.add(name)
                else:
                    planets.add("unknown")

        # pagination: move to next page
        url = data.get("next")

    return list(planets)
