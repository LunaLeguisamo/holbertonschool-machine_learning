#!/usr/bin/env python3
"""
By using the SpaceX API, create a method that
returns a list of tuples with rocket names
and their launch frequencies, sorted by frequency
in descending order. In case of a tie, sort
alphabetically by rocket name.
"""
import requests

API = "https://api.spacexdata.com/v4"


def rocket_frequencies():
    """Returns a list of tuples with rocket names
    and their launch frequencies."""
    # Get all launches
    launches = requests.get(f"{API}/launches").json()

    # Count launches per rocket ID
    freq = {}
    for launch in launches:
        rocket_id = launch.get("rocket")
        if rocket_id:
            freq[rocket_id] = freq.get(rocket_id, 0) + 1

    # Get rocket names for each rocket ID
    rockets = requests.get(f"{API}/rockets").json()
    id_to_name = {r["id"]: r["name"] for r in rockets}

    # List of tuples: (rocket_name, launch_count)
    result = []
    for rocket_id, count in freq.items():
        name = id_to_name.get(rocket_id, "Unknown")
        result.append((name, count))

    # Sort:
    # 1) count descending
    # 2) name alphabetically for ties
    result.sort(key=lambda x: (-x[1], x[0]))

    return result


if __name__ == "__main__":
    for name, count in rocket_frequencies():
        print(f"{name}: {count}")
