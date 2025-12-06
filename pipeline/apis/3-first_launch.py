#!/usr/bin/env python3
"""
Display the first SpaceX launch with specific information.
"""
import requests
from datetime import datetime
import sys


def get_first_launch():
    """Fetches and returns information about
    the first SpaceX launch."""
    # Fetch all launches
    launches_resp = requests.get("https://api.spacexdata.com/v4/launches")
    launches = launches_resp.json()

    # Sort by date_unix ascending
    launches = sorted(
        [le for le in launches if le.get("date_unix")],
        key=lambda x: x["date_unix"]
    )

    first = launches[0]  # first launch

    # Extract fields
    name = first.get("name")
    date_unix = first.get("date_unix")
    rocket_id = first.get("rocket")
    launchpad_id = first.get("launchpad")

    # Convert date to local time
    date_local = datetime.fromtimestamp(date_unix).astimezone().isoformat()

    # Get rocket name
    rocket_resp =\
        requests.get(f"https://api.spacexdata.com/v4/rockets/{rocket_id}")
    rocket_name =\
        rocket_resp.json().get("name")

    # Get launchpad info
    launchpad_resp =\
        requests.get(
            f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}"
            )
    launchpad_data = launchpad_resp.json()
    launchpad_name = launchpad_data.get("name")
    launchpad_locality = launchpad_data.get("locality")

    return f"{name} ({date_local}) {rocket_name}\
        - {launchpad_name} ({launchpad_locality})"


if __name__ == "__main__":
    print(get_first_launch())
