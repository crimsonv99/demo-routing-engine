from __future__ import annotations
import traceback
import httpx
import polyline as polyline_lib   # pip install polyline

from benchmark_engines.base import BaseEngine, RouteResult

_PROFILE = {
    "car":        "driving",
    "motorcycle": "driving",
    "bike":       "cycling",
    "walk":       "walking",
}

_DURATION_FACTOR = {
    "driving": 2.00,
    "cycling": 1.25,
    "walking": 1.00,
}


class OSRMEngine(BaseEngine):
    NAME  = "OSRM"
    COLOR = "#DC2626"   # red

    def __init__(self, url: str = "http://localhost:5000"):
        self.url = url

    def is_available(self) -> bool:
        try:
            # OSRM health: any valid request
            r = httpx.get(f"{self.url}/route/v1/driving/0,0;1,1", timeout=2.0)
            return r.status_code in (200, 400)   # 400 means server is up but bad coords
        except Exception:
            return False

    def route(self, start_lat, start_lon, end_lat, end_lon, mode="car",
              restrictions: list[dict] | None = None) -> RouteResult:
        profile = _PROFILE.get(mode, "driving")
        factor  = _DURATION_FACTOR.get(profile, 1.0)
        coords  = f"{start_lon},{start_lat};{end_lon},{end_lat}"
        url     = f"{self.url}/route/v1/{profile}/{coords}"
        params  = {"overview": "full", "geometries": "polyline"}

        try:
            r = httpx.get(url, params=params, timeout=15.0)
            r.raise_for_status()
            data = r.json()
            if data.get("code") != "Ok" or not data.get("routes"):
                return RouteResult(self.NAME, self.COLOR, 0, 0, [],
                                   error=data.get("message", "No route"))

            route   = data["routes"][0]
            decoded = polyline_lib.decode(route["geometry"])   # [(lat, lon), ...]
            coord_list = [[lon, lat] for lat, lon in decoded]

            return RouteResult(
                engine      = self.NAME,
                color       = self.COLOR,
                distance_km = route["distance"] / 1000.0,
                duration_s  = route["duration"] * factor,
                coordinates = coord_list,
            )
        except Exception as e:
            traceback.print_exc()
            return RouteResult(self.NAME, self.COLOR, 0, 0, [], error=str(e))
