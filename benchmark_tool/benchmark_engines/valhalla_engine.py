from __future__ import annotations
import traceback
import httpx

from benchmark_engines.base import BaseEngine, RouteResult

# Use Valhalla's native "motorcycle" costing for the motorcycle mode so that
# OSM tags like oneway:motorcycle=no (contra-flow allowed) are respected.
# We cap top_speed and penalise highways to simulate Vietnamese urban conditions
# (~20-30 km/h effective speed) rather than the default highway-touring profile.
_MODE = {
    "car":        "auto",
    "motorcycle": "motorcycle",   # native — respects oneway:motorcycle tag
    "bike":       "bicycle",
    "walk":       "pedestrian",
}

_COSTING_OPTIONS = {
    "auto": {
        "use_highways":       1.0,
        "use_living_streets": 0.1,
        "use_tracks":         0.0,
        "service_penalty":    30,
        "service_factor":     0.3,
    },
    # Tuned for Vietnamese urban motorcycles:
    #   top_speed 50 km/h → keeps Valhalla from modelling highway speeds
    #   use_highways 0.2  → strongly prefer surface streets over expressways
    #   use_living_streets 0.4 → allow alleyways (hẻm) that bikes use
    "motorcycle": {
        "top_speed":          50,
        "use_highways":       0.2,
        "use_living_streets": 0.4,
        "use_trails":         0.0,
    },
    "bicycle":    {"use_roads": 0.5, "avoid_bad_surfaces": 0.5},
    "pedestrian": {"walking_speed": 4.0},
}

# Duration correction: Valhalla uses free-flow speeds; multiply to match
# real Vietnamese urban traffic conditions.
#   car:        0.50 utilization → ×2.00  (validated vs Google Maps)
#   motorcycle: 0.55 utilization → ×1.82  (slightly faster than car — weaving)
import datetime

# Base duration factor — Valhalla uses free-flow OSM maxspeed.
# Real HCMC urban speed ≈ 30-45% of posted limit depending on time of day.
_BASE_FACTOR = {
    "car":        2.50,
    "motorcycle": 2.25,
    "bike":       1.50,
    "walk":       1.00,
}

# Time-of-day multiplier for HCMC traffic patterns
#   Peak rush:  7-8am, 5-7pm  → very heavy, 1.35×
#   Shoulder:   8-9am, 4-5pm, 7-8pm → moderate, 1.15×
#   Daytime:    9am-4pm → normal, 1.0×
#   Evening:    8-10pm → light, 0.85×
#   Night:      10pm-6am → free-flow, 0.70×
def _time_of_day_multiplier() -> float:
    h = datetime.datetime.now().hour
    if h in (7, 17, 18):       return 1.35   # peak rush
    if h in (8, 16, 19):       return 1.15   # shoulder
    if 9 <= h <= 15:           return 1.00   # daytime
    if h in (20, 21):          return 0.85   # evening
    return 0.70                              # night / early morning

def _get_factor(mode: str, multiplier_override: float | None = None) -> float:
    mult = multiplier_override if multiplier_override is not None else _time_of_day_multiplier()
    return _BASE_FACTOR.get(mode, 2.50) * mult

# Keep for backward compat
_MODE_DURATION_FACTOR = _BASE_FACTOR

# Traffic profile presets — displayed in the UI dropdown
TRAFFIC_PROFILES: list[tuple[str, float | None]] = [
    ("🕐 Auto (current time)",   None),
    ("🔴 Peak rush  ×1.35",      1.35),
    ("🟠 Shoulder   ×1.15",      1.15),
    ("🟡 Daytime    ×1.00",      1.00),
    ("🟢 Evening    ×0.85",      0.85),
    ("⚫ Night       ×0.70",     0.70),
]


def _decode_polyline6(encoded: str) -> list[list[float]]:
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
        coords.append([lng / factor, lat / factor])
    return coords


class ValhallaEngine(BaseEngine):
    NAME  = "Valhalla"
    COLOR = "#16A34A"   # green

    def __init__(self, url: str = "http://localhost:8002"):
        self.url = url
        self.traffic_multiplier: float | None = None   # None = auto (time of day)

    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self.url}/status", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def _raw_route(self, start_lat, start_lon, end_lat, end_lon,
                   costing: str, active: list[dict],
                   buffer_deg: float,
                   via_points: list[tuple[float, float]] | None = None) -> tuple[list, dict] | None:
        """One Valhalla /route call. Returns (coords, summary) or None on error."""
        locs = [{"lat": start_lat, "lon": start_lon, "type": "break"}]
        for vp in (via_points or []):
            locs.append({"lat": vp[0], "lon": vp[1], "type": "through"})
        locs.append({"lat": end_lat, "lon": end_lon, "type": "break"})
        body: dict = {
            "locations": locs,
            "costing": costing,
            "alternates": 0,
            "directions_options": {"units": "kilometers"},
            "costing_options": {costing: _COSTING_OPTIONS.get(costing, {})},
        }
        if active:
            locs  = _restrictions_to_exclude_locations(active)
            polys = _restrictions_to_exclude_polygons(active, buffer_deg)
            if locs:  body["exclude_locations"] = locs
            if polys: body["exclude_polygons"]  = polys
        r = httpx.post(f"{self.url}/route", json=body, timeout=15.0)
        r.raise_for_status()
        data    = r.json()
        trip    = data["trip"]
        coords  = _decode_polyline6(trip["legs"][0]["shape"])
        return coords, trip["summary"]

    def route(self, start_lat, start_lon, end_lat, end_lon, mode="car",
              restrictions: list[dict] | None = None,
              buffer_deg: float = 0.00045,
              via_points: list[tuple[float, float]] | None = None) -> RouteResult:
        costing = _MODE.get(mode, "auto")
        factor  = _get_factor(mode, self.traffic_multiplier)

        try:
            # ── Separate restrictions by direction type ────────────────
            two_way: list[dict] = []
            one_way: list[dict] = []
            if restrictions:
                for rec in restrictions:
                    d = rec.get("direction", "2way") if rec.get("type") == "RC" else "2way"
                    if d in ("1way", "1way_reverse"):
                        one_way.append(rec)
                    else:
                        two_way.append(rec)

            if one_way:
                # ── Pass 1: route ignoring 1-way restrictions ──────────
                # so we can detect which ones the natural route violates.
                p1_coords, _ = self._raw_route(
                    start_lat, start_lon, end_lat, end_lon,
                    costing, two_way, buffer_deg, via_points)

                violated = _directional_violations(p1_coords, one_way)
                active = two_way + violated
                if violated:
                    print(f"[Valhalla] 1-way check: {len(violated)}/{len(one_way)} "
                          f"restriction(s) violated — re-routing with them active")
                else:
                    print(f"[Valhalla] 1-way check: route already travels in the "
                          f"allowed direction — no extra restriction applied")
            else:
                active = two_way

            coords, summary = self._raw_route(
                start_lat, start_lon, end_lat, end_lon,
                costing, active, buffer_deg, via_points)

            osm_way_ids = _fetch_osm_way_ids(self.url, coords, costing)

            # Warn if any RC way IDs still appear in the result
            if restrictions and osm_way_ids:
                rc_way_ids = [
                    wid for rec in restrictions if rec.get("type") == "RC"
                    for wid in rec.get("way_ids", [])
                ]
                leaked = [w for w in rc_way_ids if w in osm_way_ids]
                if leaked:
                    print(f"[Valhalla] ⚠ Route still uses restricted way(s): {leaked}"
                          f" — try increasing the buffer size.")

            return RouteResult(
                engine      = self.NAME,
                color       = self.COLOR,
                distance_km = summary["length"],
                duration_s  = summary["time"] * factor,
                coordinates = coords,
                osm_way_ids = osm_way_ids,
            )
        except Exception as e:
            traceback.print_exc()
            return RouteResult(self.NAME, self.COLOR, 0, 0, [], error=str(e))


OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_WAY_POLYGON_CACHE: dict[int, dict | None] = {}    # cache per-session
_WAY_GEOMETRY_CACHE: dict[int, list[tuple[float, float]]] = {}  # raw nodes cache


def fetch_way_geometry_overpass(way_id: int) -> list[tuple[float, float]]:
    """Fetch way node coordinates from Overpass. Returns [(lat, lon), ...]. Cached."""
    if way_id in _WAY_GEOMETRY_CACHE:
        return _WAY_GEOMETRY_CACHE[way_id]
    query = f"[out:json];way({way_id});out geom;"
    try:
        r = httpx.post(OVERPASS_URL, data=query, timeout=10.0)
        r.raise_for_status()
        elements = r.json().get("elements", [])
        if not elements:
            _WAY_GEOMETRY_CACHE[way_id] = []
            return []
        geometry = elements[0].get("geometry", [])
        nodes = [(n["lat"], n["lon"]) for n in geometry]
        _WAY_GEOMETRY_CACHE[way_id] = nodes
        return nodes
    except Exception:
        return []


def fetch_node_coords_overpass(node_ids: list[int]) -> dict[int, tuple[float, float]]:
    """Fetch lat/lon for OSM node IDs. Returns {node_id: (lat, lon)}."""
    if not node_ids:
        return {}
    ids_str = ",".join(str(n) for n in node_ids)
    query = f"[out:json];node(id:{ids_str});out;"
    try:
        r = httpx.post(OVERPASS_URL, data=query, timeout=10.0)
        r.raise_for_status()
        elements = r.json().get("elements", [])
        return {int(e["id"]): (e["lat"], e["lon"]) for e in elements}
    except Exception:
        return {}


def _closest_route_idx(route_coords: list, lat: float, lon: float) -> tuple[int, float]:
    """Return (index, sq_distance) of the route point closest to (lat, lon).
    route_coords elements are [lon, lat] (Valhalla order).
    """
    best_i, best_d = 0, float("inf")
    for i, c in enumerate(route_coords):
        d = (c[1] - lat) ** 2 + (c[0] - lon) ** 2
        if d < best_d:
            best_d = d
            best_i = i
    return best_i, best_d


def _directional_violations(route_coords: list, one_way_recs: list[dict]) -> list[dict]:
    """Return the subset of one_way_recs whose blocked direction is used by the route.

    For each RC restriction with direction="1way" or "1way_reverse":
      - Fetch/use stored geometry to get the OSM node order (first_node … last_node).
      - Find the closest route points to first_node and last_node.
      - If the route passes near BOTH endpoints (within ~80 m), determine whether
        it traverses the way in the forward (OSM) or reverse direction by comparing
        the route indices of the two endpoints.
      - "1way"         blocks forward  (first_idx < last_idx in route).
      - "1way_reverse" blocks reverse  (first_idx > last_idx in route).

    The ~80 m threshold (~0.00072°) filters out ways the route doesn't use at all.
    """
    THRESHOLD_SQ = 0.00072 ** 2   # ≈ 80 m threshold squared
    violated: list[dict] = []

    for rec in one_way_recs:
        direction = rec.get("direction")   # "1way" or "1way_reverse"
        geom = rec.get("geometry", {})
        rec_violated = False

        for way_id in rec.get("way_ids", []):
            nodes = geom.get(str(way_id))
            if not nodes or len(nodes) < 2:
                # Geometry not loaded yet — treat as violated (safe fallback)
                rec_violated = True
                break

            first = nodes[0]   # [lat, lon]
            last  = nodes[-1]  # [lat, lon]

            fi, fd = _closest_route_idx(route_coords, first[0], first[1])
            li, ld = _closest_route_idx(route_coords, last[0],  last[1])

            # Route must pass near both endpoints to count
            if fd > THRESHOLD_SQ or ld > THRESHOLD_SQ:
                continue   # this way not used by the route

            route_forward = fi < li   # True = route goes in OSM direction
            blocked_fwd   = direction == "1way"

            if route_forward == blocked_fwd:
                rec_violated = True
                break   # at least one way is violated — enough

        if rec_violated:
            violated.append(rec)

    return violated


def _restrictions_to_exclude_locations(restrictions: list[dict]) -> list[dict]:
    """Convert restriction records to Valhalla exclude_locations (hard exclusions).

    IMPORTANT: Only interior nodes are excluded — never the first/last endpoint
    nodes of a way.  Endpoint nodes sit at intersections and are SHARED with
    cross-streets; excluding them would block ALL roads at that junction and
    force absurdly long detours.  Interior nodes belong exclusively to this
    way segment, so excluding them is a precise, road-only block.

    RC (2way):  interior nodes of each listed way (skips first + last).
                Falls back to exclude_polygon only when the way has ≤2 nodes.
    RC (1way/1way_reverse):  same interior-node exclusion as 2way.
                exclude_locations is non-directional but it is the ONLY hard
                blocker Valhalla supports via HTTP — exclude_polygons is soft
                (ignored when there is no alternative).  The polygon is NOT added
                for 1-way restrictions so that nearby side streets remain accessible
                and Valhalla can find the short loop-around detour.
    TR:         not excluded here (via node is an intersection node); rely on polygon.
    """
    raw: list[tuple[float, float]] = []

    for rec in restrictions:
        if rec.get("type") != "RC":
            continue   # TR handled only via polygon

        geom: dict = rec.get("geometry", {})
        for way_id in rec.get("way_ids", []):
            nodes_ll: list = geom.get(str(way_id), [])
            if not nodes_ll:
                fetched = fetch_way_geometry_overpass(int(way_id))
                nodes_ll = [[lat, lon] for lat, lon in fetched]

            # Skip first and last nodes (intersection endpoints shared with other roads)
            interior = nodes_ll[1:-1]
            if not interior:
                # Only 2-node way — no interior nodes; polygon exclusion handles it
                continue
            for pt in interior:
                raw.append((float(pt[0]), float(pt[1])))

    # Deduplicate while preserving order
    seen: set[tuple[float, float]] = set()
    result: list[dict] = []
    for lat, lon in raw:
        key = (round(lat, 7), round(lon, 7))
        if key not in seen:
            seen.add(key)
            result.append({"lat": lat, "lon": lon})
    return result


def _restrictions_to_exclude_polygons(restrictions: list[dict],
                                       buffer_deg: float = 0.00045) -> list[dict]:
    """Build line-buffer polygons for each restriction (exclude_polygons).

    RC (2way): polygon from INTERIOR nodes only (nodes[1:-1]).
               Endpoint/junction nodes are skipped so the buffer doesn't bleed into
               roundabouts or cross-streets.  2-node ways (no interior) are skipped;
               exclude_locations covers them.

    RC (1way/1way_reverse): NO polygon.
               Valhalla's exclude_polygons is soft (ignored when there is no short
               alternative).  A wide polygon would also cover nearby side streets
               that provide the short loop-around, causing huge detours.
               exclude_locations at interior nodes (hard) handles 1-way blocking.

    TR: built from via-node coordinates.
    """
    polys = []
    for rec in restrictions:
        if rec.get("type") == "RC":
            # 1-way: no polygon — see docstring above
            if rec.get("direction", "2way") in ("1way", "1way_reverse"):
                continue

            # 2-way: interior nodes only
            geom: dict = rec.get("geometry", {})
            for way_id, latlngs in geom.items():
                interior = latlngs[1:-1]
                if len(interior) < 2:
                    continue
                nodes = [(pt[0], pt[1]) for pt in interior]
                poly = _line_buffer_polygon(nodes, buffer_deg)
                if poly:
                    polys.append(poly)
        elif rec.get("type") == "TR":
            nodes: list[tuple[float, float]] = []
            for coords in rec.get("node_coords", {}).values():
                nodes.append((coords[0], coords[1]))
            if len(nodes) >= 2:
                poly = _line_buffer_polygon(nodes, buffer_deg)
                if poly:
                    polys.append(poly)
    return polys


def _way_id_to_exclude_polygon(way_id: int, width_deg: float = 0.00100) -> dict | None:
    """Return a Valhalla exclude_polygon (GeoJSON Polygon) for an OSM way.

    Builds a proper line-buffer polygon along the way's actual geometry
    (~33 m per side) rather than a bounding box, so narrow roads are
    excluded precisely without blocking adjacent parallel streets.

    GeoJSON exterior ring is counter-clockwise as required.
    Results are cached for the session.
    """
    if way_id in _WAY_POLYGON_CACHE:
        return _WAY_POLYGON_CACHE[way_id]

    nodes = fetch_way_geometry_overpass(way_id)
    poly = _line_buffer_polygon(nodes, width_deg)
    _WAY_POLYGON_CACHE[way_id] = poly
    if poly is None:
        print(f"[Valhalla] ⚠ way {way_id}: Overpass returned no geometry — exclusion skipped")
    else:
        print(f"[Valhalla] way {way_id}: exclusion polygon ready ({len(nodes)} nodes, ±{width_deg:.5f}°)")
    return poly


def _line_buffer_polygon(nodes: list[tuple[float, float]],
                         width_deg: float) -> dict | None:
    """Build a CCW GeoJSON Polygon buffer around a list of (lat, lon) nodes."""
    import math

    if len(nodes) < 2:
        return None

    def _offset(p1, p2, side: float):
        """Return (lat, lon) offset perpendicular to p1→p2 by `side` degrees.
        Positive side = left of travel direction.
        """
        dlat = p2[0] - p1[0]
        dlon = p2[1] - p1[1]
        length = math.sqrt(dlat * dlat + dlon * dlon)
        if length == 0:
            return (0.0, 0.0)
        # Perpendicular (left = rotate +90°): (-dlon, dlat) normalised
        return (-dlon / length * side, dlat / length * side)

    left_ring:  list[list[float]] = []
    right_ring: list[list[float]] = []

    for i, (lat, lon) in enumerate(nodes):
        # Use the segment direction at this node
        if i < len(nodes) - 1:
            seg = (nodes[i], nodes[i + 1])
        else:
            seg = (nodes[i - 1], nodes[i])

        dlat_l, dlon_l = _offset(seg[0], seg[1], width_deg)
        left_ring.append( [lon + dlon_l, lat + dlat_l])   # GeoJSON [lon, lat]
        right_ring.append([lon - dlon_l, lat - dlat_l])

    # CCW exterior ring: go forward along left side, backward along right side
    ring = left_ring + list(reversed(right_ring)) + [left_ring[0]]
    return {"type": "Polygon", "coordinates": [ring]}


def _fetch_osm_way_ids(url: str, coords: list[list[float]], costing: str) -> list[int]:
    """Call /trace_attributes to get deduplicated ordered OSM way IDs."""
    try:
        shape = [{"lat": c[1], "lon": c[0]} for c in coords]
        body = {
            "shape": shape,
            "costing": costing,
            "shape_match": "map_snap",
            "filters": {"attributes": ["edge.way_id"], "action": "include"},
        }
        r = httpx.post(f"{url}/trace_attributes", json=body, timeout=10.0)
        r.raise_for_status()
        edges = r.json().get("edges", [])
        seen, result = set(), []
        for e in edges:
            wid = e.get("way_id")
            if wid is not None and wid not in seen:
                seen.add(wid)
                result.append(int(wid))
        return result
    except Exception:
        return []
