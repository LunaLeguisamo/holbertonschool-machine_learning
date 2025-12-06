#!/usr/bin/env python3
"""
Print the location of a GitHub user given the full API URL.

Usage:
    ./2-user_location.py https://api.github.com/users/holbertonschool
"""
import sys
import time
import math
import requests


def get_user_location(url):
    """Fetches the location of a GitHub user
    from the given API URL."""
    try:
        resp = requests.get(url)
    except requests.RequestException:
        return None, None  # network error -> treat as no data

    status = resp.status_code

    if status == 200:
        try:
            data = resp.json()
        except ValueError:
            return None, status
        # Print the location even if it's None
        return data.get('location'), status

    if status == 404:
        return "Not found", status

    if status == 403:
        # Rate limit: compute minutes until reset
        reset_header = resp.headers.get("X-RateLimit-Reset")
        minutes = 0
        if reset_header:
            try:
                reset_ts = int(reset_header)
                now_ts = int(time.time())
                diff = reset_ts - now_ts
                minutes = math.ceil(diff / 60) if diff > 0 else 0
            except (ValueError, TypeError):
                minutes = 0
        return f"Reset in {minutes} min", status

    # Other statuses: return generic message (could be extended)
    return None, status


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./2-user_location.py <github_user_api_url>")
        sys.exit(1)

    url = sys.argv[1]
    result, code = get_user_location(url)

    if code == 404:
        print("Not found")
    elif code == 403 and isinstance(result, str)\
            and result.startswith("Reset in"):
        print(result)
    elif code == 200:
        # If location is None, printing None matches
        # typical expectations
        print(result)
    else:
        # For network errors or unexpected responses,
        # print nothing or a fallback
        if result is None:
            # No output required by the spec; raising a
            # short message is helpful
            print("Not found" if code == 404 else "")
