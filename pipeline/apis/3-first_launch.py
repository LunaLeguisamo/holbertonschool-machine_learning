#!/usr/bin/env python3
"""
Print the first upcoming SpaceX launch in the format:
<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
"""
import requests
from datetime import datetime
import sys

API_BASE = "https://api.spacexdata.com/v4"


def get_first_upcoming_launch():
    """Fetches information about the first upcoming SpaceX launch."""
    # Pedimos los lanzamientos "upcoming" (futuros)
    resp = requests.get(f"{API_BASE}/launches/upcoming")
    resp.raise_for_status()
    launches = resp.json()

    # Filtramos los que tienen date_unix y ordenamos ascendente por date_unix.
    launches = [le for le in launches if le.get("date_unix") is not None]
    if not launches:
        return None

    launches = sorted(launches, key=lambda x: x["date_unix"])  # stable sort

    first = launches[0]

    # Datos básicos
    name = first.get("name")
    date_unix = first.get("date_unix")
    rocket_id = first.get("rocket")
    launchpad_id = first.get("launchpad")

    # Fecha en hora local con offset
    date_local = datetime.fromtimestamp(date_unix).astimezone().isoformat()

    # Rocket name
    rocket_name = None
    if rocket_id:
        r = requests.get(f"{API_BASE}/rockets/{rocket_id}")
        r.raise_for_status()
        rocket_name = r.json().get("name")

    # Launchpad name + locality
    launchpad_name = None
    launchpad_locality = None
    if launchpad_id:
        p = requests.get(f"{API_BASE}/launchpads/{launchpad_id}")
        p.raise_for_status()
        pdata = p.json()
        launchpad_name = pdata.get("name")
        launchpad_locality = pdata.get("locality")

    return {
        "name": name,
        "date": date_local,
        "rocket": rocket_name,
        "pad_name": launchpad_name,
        "pad_locality": launchpad_locality
    }


if __name__ == "__main__":
    try:
        info = get_first_upcoming_launch()
    except requests.RequestException as e:
        # En contexto de los ejercicios normalmente no se imprime nada extra,
        # pero para depuración podrías imprimir en stderr.
        sys.exit(1)

    if not info:
        sys.exit(0)

    print(f"{info['name']} ({info['date']}) {info['rocket']}\
        - {info['pad_name']} ({info['pad_locality']})")
