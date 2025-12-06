#!/usr/bin/env python3
"""
By using the SpaceX API, create a method that
returns the details of the first upcoming launch.
"""
import requests
import sys

API = "https://api.spacexdata.com/v4"


def get_first_upcoming_launch():
    """Fetches details of the first upcoming SpaceX launch."""
    launches = requests.get(f"{API}/launches/upcoming").json()
    launches = [le for le in launches if le.get("date_unix")]

    # Orden estable por date_unix
    launches.sort(key=lambda le: le["date_unix"])

    first = launches[0]

    name = first.get("name")
    date_local = first.get("date_local")
    rocket_id = first.get("rocket")
    launchpad_id = first.get("launchpad")

    # rocket info
    rocket_name = requests.get(f"{API}/rockets/{rocket_id}").json().get("name")

    # launchpad info
    pad = requests.get(f"{API}/launchpads/{launchpad_id}").json()
    pad_name = pad.get("name")
    pad_locality = pad.get("locality")

    return f"{name} ({date_local}) {rocket_name} - {pad_name} ({pad_locality})"


if __name__ == "__main__":
    print(get_first_upcoming_launch())
