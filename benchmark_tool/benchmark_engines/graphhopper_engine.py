from __future__ import annotations
import math
import traceback
import datetime

import httpx

# Ferry routes run on fixed schedules — not affected by road traffic.
_FERRY_SPEED_KPH = 10.0

from benchmark_engines.base import BaseEngine, RouteResult

# ── Profile mapping ────────────────────────────────────────────────────────
# GraphHopper 11 uses profile names that match the custom_model_files we built.
# "motorcycle" uses car_access internally but respects motorcycle.json routing rules.
_PROFILE = {
    "car":        "car",
    "motorcycle": "motorcycle",
    "bike":       "bike",
    "walk":       "foot",
}

# ── Duration correction ────────────────────────────────────────────────────
# GraphHopper uses free-flow OSM maxspeed, same as Valhalla.
# Same correction factors apply for Vietnamese urban traffic.
_BASE_FACTOR = {
    "car":        2.50,
    "motorcycle": 2.25,
    "bike":       1.50,
    "walk":       1.00,
}

def _time_of_day_multiplier() -> float:
    h = datetime.datetime.now().hour
    if h in (7, 17, 18):   return 1.35   # peak rush
    if h in (8, 16, 19):   return 1.15   # shoulder
    if 9 <= h <= 15:        return 1.00   # daytime
    if h in (20, 21):       return 0.85   # evening
    return 0.70                           # night

def _get_factor(mode: str, multiplier_override: float | None = None) -> float:
    mult = multiplier_override if multiplier_override is not None else _time_of_day_multiplier()
    return _BASE_FACTOR.get(mode, 2.50) * mult


def _compute_gh_duration(path: dict, time_mult: float) -> float:
    """Compute corrected duration using per-segment highway penalties.

    Builds per-point lookup tables from road_class, road_environment, and
    average_speed detail intervals, then iterates over coordinate pairs to
    apply the appropriate penalty × time_mult for each segment.

    Ferry segments get _FERRY_SPEED_KPH, no time_mult (fixed schedule).
    All other segments: free_flow_time × _HIGHWAY_PENALTY[road_class] × time_mult.
    """
    from benchmark_engines.valhalla_engine import _HIGHWAY_PENALTY

    coords  = path["points"]["coordinates"]   # [[lon, lat], ...]
    details = path.get("details", {})
    n       = len(coords)

    # Build per-point speed map (km/h)
    speed_at: dict[int, float] = {}
    for f, t, spd in details.get("average_speed", []):
        for i in range(f, t):
            speed_at[i] = float(spd)

    # Build per-point road_class map (lowercase for penalty lookup)
    rc_at: dict[int, str] = {}
    for f, t, rc in details.get("road_class", []):
        for i in range(f, t):
            rc_at[i] = rc.lower()

    # Build per-point road_environment map (lowercase)
    env_at: dict[int, str] = {}
    for f, t, env in details.get("road_environment", []):
        for i in range(f, t):
            env_at[i] = env.lower()

    duration_s = 0.0
    _default_penalty = _HIGHWAY_PENALTY["unclassified"]

    for i in range(n - 1):
        a, b = coords[i], coords[i + 1]
        mid_lat = math.radians((a[1] + b[1]) / 2)
        dx      = (b[0] - a[0]) * math.cos(mid_lat) * 111_320
        dy      = (b[1] - a[1]) * 110_540
        seg_km  = math.sqrt(dx * dx + dy * dy) / 1000.0

        env   = env_at.get(i,   "road")
        speed = speed_at.get(i, 30.0)

        if env == "ferry":
            # Ferry: fixed speed, no traffic factor
            duration_s += seg_km / _FERRY_SPEED_KPH * 3600
        elif speed > 0:
            rc          = rc_at.get(i, "unclassified")
            penalty     = _HIGHWAY_PENALTY.get(rc, _default_penalty)
            free_flow_s = seg_km / speed * 3600
            duration_s += free_flow_s * penalty * time_mult

    return duration_s


class GraphHopperEngine(BaseEngine):
    NAME  = "GraphHopper"
    COLOR = "#EA580C"   # orange

    def __init__(self, url: str = "http://localhost:8989"):
        self.url = url
        self.traffic_multiplier: float | None = None   # None = auto (time of day)

    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self.url}/health", timeout=2.0)
            return r.status_code == 200 and r.text.strip() == "OK"
        except Exception:
            return False

    # ── Internal route call ────────────────────────────────────────────
    def _raw_route(self, start_lat, start_lon, end_lat, end_lon,
                   profile: str,
                   via_points: list[tuple[float, float]] | None = None,
                   blocked_points: list[dict] | None = None) -> dict:
        """POST /route and return the first path dict."""
        points = [[start_lon, start_lat]]
        for vp in (via_points or []):
            points.append([vp[1], vp[0]])   # (lat,lon) → [lon,lat]
        points.append([end_lon, end_lat])

        body: dict = {
            "points":          points,
            "profile":         profile,
            "points_encoded":  False,
            # road_class       → highway-type penalty lookup
            # road_environment → detect ferry segments
            # average_speed    → free-flow speed per segment for time calc
            "details":         ["osm_way_id", "road_class",
                                "road_environment", "average_speed"],
        }

        # Restriction enforcement via blocked_points (hard avoidance)
        if blocked_points:
            body["blocked_points"] = blocked_points   # [{lon, lat}, ...]

        r = httpx.post(f"{self.url}/route", json=body, timeout=15.0)
        r.raise_for_status()
        data = r.json()
        if "paths" not in data or not data["paths"]:
            raise ValueError("No path returned by GraphHopper")
        return data["paths"][0]

    # ── Public route method (matches BaseEngine signature) ─────────────
    def route(self, start_lat, start_lon, end_lat, end_lon, mode="car",
              restrictions: list[dict] | None = None,
              buffer_deg: float = 0.00009,
              via_points: list[tuple[float, float]] | None = None) -> RouteResult:

        profile = _PROFILE.get(mode, "car")
        factor  = _get_factor(mode, self.traffic_multiplier)

        try:
            # ── Separate 1-way and 2-way restrictions ──────────────────
            two_way: list[dict] = []
            one_way: list[dict] = []
            if restrictions:
                for rec in restrictions:
                    d = rec.get("direction", "2way") if rec.get("type") == "RC" else "2way"
                    if d in ("1way", "1way_reverse"):
                        one_way.append(rec)
                    else:
                        two_way.append(rec)

            # ── Pass 1: route with only 2-way restrictions ─────────────
            if one_way:
                blocked_2way = _restrictions_to_blocked_points(two_way)
                p1 = self._raw_route(start_lat, start_lon, end_lat, end_lon,
                                     profile, via_points, blocked_2way or None)
                p1_coords = [[c[0], c[1]] for c in p1["points"]["coordinates"]]

                from benchmark_engines.valhalla_engine import _directional_violations
                violated = _directional_violations(p1_coords, one_way)

                if violated:
                    print(f"[GraphHopper] 1-way check: {len(violated)}/{len(one_way)} "
                          f"restriction(s) violated — re-routing")
                else:
                    print(f"[GraphHopper] 1-way check: route already in allowed direction")

                active = two_way + violated
            else:
                active = two_way

            blocked = _restrictions_to_blocked_points(active)
            path = self._raw_route(start_lat, start_lon, end_lat, end_lon,
                                   profile, via_points, blocked or None)

            # ── Decode geometry ────────────────────────────────────────
            # GraphHopper returns GeoJSON [lon, lat] when points_encoded=False
            coords = [[c[0], c[1]] for c in path["points"]["coordinates"]]

            # ── Extract OSM way IDs from details ──────────────────────
            osm_way_ids = _extract_way_ids(path)

            # ── Leak check (RC ways still in result) ───────────────────
            if restrictions and osm_way_ids:
                rc_way_ids = [
                    wid for rec in restrictions if rec.get("type") == "RC"
                    for wid in rec.get("way_ids", [])
                ]
                leaked = [w for w in rc_way_ids if w in osm_way_ids]
                if leaked:
                    print(f"[GraphHopper] ⚠ Route still uses restricted way(s): {leaked}")

            # ── Per-segment duration with highway penalties ────────────
            # Replaces the single _BASE_FACTOR global multiplier.
            # road_class + average_speed + road_environment details give us
            # the free-flow speed per segment so we can apply the calibrated
            # highway penalty for each road type.
            # Ferry segments: _FERRY_SPEED_KPH, no traffic factor.
            time_mult  = (self.traffic_multiplier
                          if self.traffic_multiplier is not None
                          else _time_of_day_multiplier())
            duration_s = _compute_gh_duration(path, time_mult)

            return RouteResult(
                engine      = self.NAME,
                color       = self.COLOR,
                distance_km = path["distance"] / 1000.0,
                duration_s  = duration_s,
                coordinates = coords,
                osm_way_ids = osm_way_ids,
            )

        except Exception as e:
            traceback.print_exc()
            return RouteResult(self.NAME, self.COLOR, 0, 0, [], error=str(e))


# ── Restriction helpers ────────────────────────────────────────────────────

def _restrictions_to_blocked_points(restrictions: list[dict]) -> list[dict]:
    """Convert RC restrictions to GraphHopper blocked_points (interior nodes only).

    GraphHopper's blocked_points works like Valhalla's exclude_locations:
    hard avoidance at specific coordinates, bidirectional.

    Same interior-node-only rule: skip first and last node of each way so
    we don't block the shared junction/intersection nodes.
    """
    from benchmark_engines.valhalla_engine import fetch_way_geometry_overpass

    raw: list[tuple[float, float]] = []

    for rec in restrictions:
        if rec.get("type") != "RC":
            continue   # TR: no blocked_points (junction nodes)

        geom: dict = rec.get("geometry", {})
        for way_id in rec.get("way_ids", []):
            nodes_ll: list = geom.get(str(way_id), [])
            if not nodes_ll:
                fetched = fetch_way_geometry_overpass(int(way_id))
                nodes_ll = [[lat, lon] for lat, lon in fetched]

            interior = nodes_ll[1:-1]   # skip first and last (junction endpoints)
            if not interior:
                # 2-node way — no interior nodes; soft fallback only
                # (GH has no polygon exclusion API; accept that 2-node ways may leak)
                continue
            for pt in interior:
                raw.append((float(pt[0]), float(pt[1])))

    # Deduplicate
    seen: set[tuple[float, float]] = set()
    result: list[dict] = []
    for lat, lon in raw:
        key = (round(lat, 7), round(lon, 7))
        if key not in seen:
            seen.add(key)
            result.append({"lon": lon, "lat": lat})
    return result


def _extract_way_ids(path: dict) -> list[int]:
    """Extract deduplicated ordered OSM way IDs from GraphHopper route details."""
    intervals = path.get("details", {}).get("osm_way_id", [])
    seen: set[int] = set()
    result: list[int] = []
    for interval in intervals:
        # interval = [from_idx, to_idx, way_id]
        wid = int(interval[2])
        if wid not in seen:
            seen.add(wid)
            result.append(wid)
    return result
