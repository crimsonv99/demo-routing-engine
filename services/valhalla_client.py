"""
Valhalla HTTP client — replaces routing_engine.py (NetworkX).

Wraps Valhalla's /route endpoint and normalises the response
into the same RouteResult dataclass the rest of the app uses.
"""
from __future__ import annotations

import httpx
from dataclasses import dataclass, field
from typing import Optional


import os
VALHALLA_URL = os.getenv("VALHALLA_URL", "http://localhost:8002")

# Valhalla costing name per travel mode
MODE_MAP = {
    "car":        "auto",
    "motorcycle": "motorcycle",
    "bike":       "bicycle",
    "walk":       "pedestrian",
    # pass-through if already a Valhalla costing name
    "auto":       "auto",
    "bicycle":    "bicycle",
    "pedestrian": "pedestrian",
}


@dataclass
class RouteResult:
    geometry:     dict          # GeoJSON LineString {"type": "LineString", "coordinates": [...]}
    distance_km:  float
    duration_s:   float
    instructions: list[str] = field(default_factory=list)
    road_names:   list[str] = field(default_factory=list)


# ── polyline6 decoder ──────────────────────────────────────────────────────

def _decode_polyline6(encoded: str) -> list[list[float]]:
    """
    Decode Valhalla's encoded polyline (precision=6) → [[lon, lat], ...].
    Valhalla encodes as [lat, lon] internally; we swap to [lon, lat] for GeoJSON.
    """
    coords: list[list[float]] = []
    index = lat = lng = 0
    factor = 1e6

    while index < len(encoded):
        for is_lat in (True, False):
            shift = result = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else result >> 1
            if is_lat:
                lat += delta
            else:
                lng += delta
        # GeoJSON convention: [longitude, latitude]
        coords.append([lng / factor, lat / factor])

    return coords


# ── instruction builder ───────────────────────────────────────────────────

def _extract_instructions(trip: dict) -> tuple[list[str], list[str]]:
    """Pull turn-by-turn instructions and road names from a Valhalla trip leg."""
    instructions: list[str] = []
    road_names:   list[str] = []

    for leg in trip.get("legs", []):
        for maneuver in leg.get("maneuvers", []):
            instr = maneuver.get("instruction", "")
            if instr:
                instructions.append(instr)
            name = maneuver.get("street_names", [])
            road_names.extend(name)

    return instructions, list(dict.fromkeys(road_names))  # deduplicate, preserve order


# ── main client function ──────────────────────────────────────────────────

async def get_routes(
    start_lat: float,
    start_lon: float,
    end_lat:   float,
    end_lon:   float,
    mode:      str = "car",
    alternatives: int = 2,
    timeout:   float = 15.0,
) -> list[RouteResult]:
    """
    Call Valhalla /route and return a list of RouteResult objects.
    First element is the best route; subsequent ones are alternatives.
    Raises httpx.HTTPError on connection / HTTP failure.
    """
    costing = MODE_MAP.get(mode, "auto")

    body = {
        "locations": [
            {"lat": start_lat, "lon": start_lon, "type": "break"},
            {"lat": end_lat,   "lon": end_lon,   "type": "break"},
        ],
        "costing": costing,
        "alternates": alternatives,
        "directions_options": {
            "units": "kilometers",
            "language": "en-US",
        },
        "costing_options": {
            costing: {
                "use_highways": 1.0,
                "use_tolls":    0.5,
            }
        },
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{VALHALLA_URL}/route", json=body)
        resp.raise_for_status()

    data = resp.json()

    # Valhalla returns the best route under "trip" and alternatives under "alternates"
    trips = [data["trip"]] + [alt["trip"] for alt in data.get("alternates", [])]

    results: list[RouteResult] = []
    for trip in trips:
        summary  = trip["summary"]
        shape    = trip["legs"][0]["shape"]          # encoded polyline6
        coords   = _decode_polyline6(shape)          # [[lon, lat], ...]
        instrs, road_names = _extract_instructions(trip)

        results.append(RouteResult(
            geometry    = {"type": "LineString", "coordinates": coords},
            distance_km = summary["length"],
            duration_s  = summary["time"],
            instructions= instrs,
            road_names  = road_names,
        ))

    return results


async def health_check() -> bool:
    """Return True if Valhalla is reachable and responding."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{VALHALLA_URL}/status")
            return r.status_code == 200
    except Exception:
        return False
