"""
Valhalla HTTP client — replaces routing_engine.py (NetworkX).

Wraps Valhalla's /route endpoint and normalises the response
into the same RouteResult dataclass the rest of the app uses.

Cost model is tuned to match the original routing_engine.py settings:
  - SPEED_UTILIZATION_CAR        = 0.50  → top_speed=40, speed_factor=0.50
  - SPEED_UTILIZATION_MOTORCYCLE = 0.65  → top_speed=52, speed_factor=0.65
  - CAR_ROAD_PENALTY / MOTORCYCLE_ROAD_PENALTY
      → mapped to Valhalla use_* and service_penalty options
"""
from __future__ import annotations

import os
import httpx
from dataclasses import dataclass, field

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

# ── Costing options per mode ─────────────────────────────────────────────────
#
# Ported from routing_engine.py:
#
#   CAR_ROAD_PENALTY:
#     motorway=0.5 (very preferred), trunk=0.6, primary=0.8
#     residential=3.0, service=5.0, living_street=8.0 (heavily avoided)
#   → use_highways=1.0, use_living_streets=0.1, service_penalty=30, service_factor=0.3
#
#   SPEED_UTILIZATION_CAR = 0.50
#   → top_speed capped at 40 km/h (80 km/h × 0.50) for Vietnam urban traffic
#
#   MOTORCYCLE_ROAD_PENALTY:
#     more flexible — prefers primary/secondary, tolerates residential
#   → use_highways=0.5, use_living_streets=0.5, service_penalty=10
#
#   SPEED_UTILIZATION_MOTORCYCLE = 0.65
#   → top_speed capped at 52 km/h (80 km/h × 0.65)

COSTING_OPTIONS: dict[str, dict] = {
    "auto": {
        # Road type preference (mirrors CAR_ROAD_PENALTY)
        "use_highways":       1.0,   # strongly prefer motorway/trunk (penalty 0.5–0.6)
        "use_tolls":          0.5,   # neutral on tolls
        "use_living_streets": 0.1,   # strongly avoid (penalty 8.0)
        "use_tracks":         0.0,   # avoid tracks (penalty 15.0)
        "service_penalty":    30,    # heavy penalty for service roads (penalty 5.0)
        "service_factor":     0.3,   # slow down on service roads
    },
    "motorcycle": {
        # Road type preference (mirrors MOTORCYCLE_ROAD_PENALTY)
        "use_highways":       0.5,   # less highway preference than car
        "use_tolls":          0.5,
        "use_living_streets": 0.5,   # tolerates living streets (penalty 1.8)
        "use_tracks":         0.2,   # can use tracks (penalty 3.0)
        "service_penalty":    10,    # light service road penalty (penalty 1.5)
        "service_factor":     0.7,
    },
    "bicycle": {
        "use_roads":          0.5,
        "use_hills":          0.5,
        "avoid_bad_surfaces": 0.5,
    },
    "pedestrian": {
        "walking_speed":      4.0,   # km/h
        "use_living_streets": 1.0,
        "use_tracks":         0.5,
    },
}

# ── Duration correction factors ───────────────────────────────────────────────
#
# Valhalla estimates duration at near free-flow speed.
# The old routing_engine applied SPEED_UTILIZATION to reflect Vietnamese traffic:
#   SPEED_UTILIZATION_CAR        = 0.50  → duration × (1/0.50) = × 2.0
#   SPEED_UTILIZATION_MOTORCYCLE = 0.65  → duration × (1/0.65) ≈ × 1.54
#
# This makes a 4-min Valhalla result → 8 min displayed (matching real conditions).

DURATION_FACTOR: dict[str, float] = {
    "auto":       1 / 0.50,   # 2.00× — heavy urban congestion (car)
    "motorcycle": 1 / 0.65,   # 1.54× — motorbike weaves faster through traffic
    "bicycle":    1 / 0.80,   # 1.25× — minor correction for hills/stops
    "pedestrian": 1.0,        # walking speed already accurate
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
    options = COSTING_OPTIONS.get(costing, {})

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
            costing: options,
        },
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{VALHALLA_URL}/route", json=body)
        resp.raise_for_status()

    data = resp.json()

    # Valhalla returns the best route under "trip" and alternatives under "alternates"
    trips = [data["trip"]] + [alt["trip"] for alt in data.get("alternates", [])]

    factor = DURATION_FACTOR.get(costing, 1.0)

    results: list[RouteResult] = []
    for trip in trips:
        summary  = trip["summary"]
        shape    = trip["legs"][0]["shape"]          # encoded polyline6
        coords   = _decode_polyline6(shape)          # [[lon, lat], ...]
        instrs, road_names = _extract_instructions(trip)

        results.append(RouteResult(
            geometry    = {"type": "LineString", "coordinates": coords},
            distance_km = summary["length"],
            duration_s  = summary["time"] * factor,  # apply Vietnam traffic factor
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
