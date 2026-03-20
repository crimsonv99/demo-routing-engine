from __future__ import annotations
import traceback
import httpx
import polyline as polyline_lib   # pip install polyline

from benchmark_engines.base import BaseEngine, RouteResult

_VEHICLE = {
    "car":        "car",
    "motorcycle": "motorcycle",
    "bike":       "bike",
    "walk":       "foot",
}

_DURATION_FACTOR = {
    "car":        2.00,
    "motorcycle": 1.54,
    "bike":       1.25,
    "foot":       1.00,
}


class GraphHopperEngine(BaseEngine):
    NAME  = "GraphHopper"
    COLOR = "#EA580C"   # orange

    def __init__(self, url: str = "http://localhost:8989"):
        self.url = url

    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self.url}/health", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def route(self, start_lat, start_lon, end_lat, end_lon, mode="car",
              restrictions: list[dict] | None = None) -> RouteResult:
        vehicle = _VEHICLE.get(mode, "car")
        factor  = _DURATION_FACTOR.get(vehicle, 1.0)
        params  = {
            "point":       [f"{start_lat},{start_lon}", f"{end_lat},{end_lon}"],
            "vehicle":     vehicle,
            "locale":      "en",
            "points_encoded": "true",
        }
        try:
            r = httpx.get(f"{self.url}/route", params=params, timeout=15.0)
            r.raise_for_status()
            data = r.json()
            if "paths" not in data or not data["paths"]:
                return RouteResult(self.NAME, self.COLOR, 0, 0, [], error="No path returned")

            path    = data["paths"][0]
            # GraphHopper returns encoded polyline5 (lat, lon order)
            decoded = polyline_lib.decode(path["points"])  # [(lat, lon), ...]
            coords  = [[lon, lat] for lat, lon in decoded]

            return RouteResult(
                engine      = self.NAME,
                color       = self.COLOR,
                distance_km = path["distance"] / 1000.0,
                duration_s  = (path["time"] / 1000.0) * factor,  # ms → s × factor
                coordinates = coords,
            )
        except Exception as e:
            traceback.print_exc()
            return RouteResult(self.NAME, self.COLOR, 0, 0, [], error=str(e))
