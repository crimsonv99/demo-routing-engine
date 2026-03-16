from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import math
import time
import threading
import numpy as np
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree

# ---------------------------------------------------------------------------
# Speed tables (kph)
# ---------------------------------------------------------------------------

# Speed utilization ratios — fraction of maxspeed actually achievable
# in urban HCMC conditions (signals, congestion, turning).
# Applied to both OSM maxspeed tags AND fallback assumed speeds.
# Car travels slower relative to limit due to larger vehicle and lane discipline.
SPEED_UTILIZATION_CAR = 0.50         # 50% of maxspeed on average
SPEED_UTILIZATION_MOTORCYCLE = 0.65  # bikes weave more, closer to limit

# Fallback assumed maxspeed (kph) when OSM maxspeed tag is missing.
# These are legal/design speeds, NOT travel speeds —
# utilization ratio is applied on top.
HIGHWAY_FALLBACK_SPEED_CAR = {
    "motorway":      80,
    "trunk":         60,
    "primary":       50,
    "secondary":     40,
    "tertiary":      30,
    "unclassified":  30,
    "residential":   20,
    "service":       15,
    "living_street": 10,
}

HIGHWAY_FALLBACK_SPEED_MOTORCYCLE = {
    "motorway":      80,
    "trunk":         60,
    "primary":       50,
    "secondary":     40,
    "tertiary":      30,
    "unclassified":  30,
    "residential":   20,
    "service":       15,
    "living_street": 10,
}

# Road class penalty multipliers for CAR routing.
# Applied to travel-time weight so the router strongly prefers
# main roads over alleys/residential streets.
# 1.0 = no penalty, higher = more expensive to use.
CAR_ROAD_PENALTY = {
    "motorway":      0.5,   # expressway — very fast, strongly preferred
    "trunk":         0.6,
    "primary":       0.8,
    "secondary":     1.0,   # baseline
    "tertiary":      1.3,
    "unclassified":  2.0,
    "residential":   3.0,   # discourage residential for cars
    "service":       5.0,   # strongly discourage service roads
    "living_street": 8.0,   # near-block for cars
    "track":        15.0,
    "path":         20.0,
    "footway":      50.0,
    "cycleway":     50.0,
}

# Motorcycle penalty — much lighter, alleys are acceptable
MOTORCYCLE_ROAD_PENALTY = {
    "motorway":      0.8,
    "trunk":         0.9,
    "primary":       1.0,
    "secondary":     1.0,
    "tertiary":      1.1,
    "unclassified":  1.2,
    "residential":   1.3,
    "service":       1.5,
    "living_street": 1.8,
    "track":         3.0,
    "path":          4.0,
    "footway":      50.0,
    "cycleway":     50.0,
}

# Minimum fraction of edges that must differ between two routes.
# e.g. 0.20 means routes must share at most 80% of their edge-sets.
DIVERSITY_THRESHOLD = 0.20


# ---------------------------------------------------------------------------
# OSM attribute parsers
# ---------------------------------------------------------------------------

def _parse_bool_osm(v) -> Optional[bool]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y"):
        return True
    if s in ("0", "false", "no", "n"):
        return False
    return None


def _parse_oneway_value(v) -> Optional[str]:
    """Return 'forward', 'reverse', 'no', or None from OSM-like oneway values."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y"):
        return "forward"
    if s in ("-1", "reverse"):
        return "reverse"
    if s in ("0", "false", "no", "n"):
        return "no"
    return None


def _parse_access(v) -> Optional[bool]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("no", "0", "false", "private"):
        return False
    if s in ("yes", "1", "true", "designated", "permissive"):
        return True
    return None


def _parse_maxspeed_kph(v) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip().lower()
    m = "".join(ch for ch in s if (ch.isdigit() or ch == "."))
    if not m:
        return None
    try:
        return float(m)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class RouteResult:
    rank: int
    distance_m: float
    duration_s: float
    geometry_3857: LineString
    edge_set: frozenset  # internal – used for diversity filtering, not serialised


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RouteEngine:
    def __init__(self, roads_gdf_3857: gpd.GeoDataFrame):
        self.roads = roads_gdf_3857
        self.G = nx.DiGraph()
        self._node_id: Dict[Tuple[float, float], int] = {}
        self._node_xy: Dict[int, Tuple[float, float]] = {}
        self._project_cache: Dict[Tuple[float, float], Point] = {}
        self._snap_cache: Dict[Tuple[float, float], int] = {}
        # snap_node_id -> list of (u, v) connector edges belonging to that snap
        self._connector_edges: Dict[int, List[Tuple[int, int]]] = {}

        self._build_graph()

        # Spatial index over *permanent* edges only (built once, never mutated).
        self._edge_geoms: List[LineString] = []
        self._edge_meta: List[Tuple[int, int, dict]] = []
        for u, v, data in self.G.edges(data=True):
            self._edge_geoms.append(data["geometry"])
            self._edge_meta.append((u, v, data))
        self._tree = STRtree(self._edge_geoms)

    # ------------------------------------------------------------------
    # Node registry
    # ------------------------------------------------------------------

    def _nid(self, x: float, y: float) -> int:
        key = (round(x, 3), round(y, 3))
        if key in self._node_id:
            return self._node_id[key]
        nid = len(self._node_id) + 1
        self._node_id[key] = nid
        self._node_xy[nid] = (x, y)
        self.G.add_node(nid, x=x, y=y)
        return nid

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------

    def _project_point_3857(self, lon: float, lat: float) -> Point:
        key = (round(lon, 7), round(lat, 7))
        cached = self._project_cache.get(key)
        if cached is not None:
            return cached
        p = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
        self._project_cache[key] = p
        return p

    # ------------------------------------------------------------------
    # Connector edge lifecycle
    # ------------------------------------------------------------------

    def _clear_connector_edges(self, sn: int) -> None:
        """Remove all temporary connector edges that were added for snap-node *sn*."""
        edges = self._connector_edges.pop(sn, [])
        for u, v in edges:
            if self.G.has_edge(u, v):
                self.G.remove_edge(u, v)

    def _clear_all_connector_edges(self, snap_nodes: List[int]) -> None:
        """Remove connectors for a list of snap nodes (call after routing is done)."""
        for sn in snap_nodes:
            self._clear_connector_edges(sn)
            # Also invalidate the snap cache so the node is re-injected fresh next time.
            self._snap_cache = {k: v for k, v in self._snap_cache.items() if v != sn}

    # ------------------------------------------------------------------
    # Speed / access / oneway helpers
    # ------------------------------------------------------------------

    def _edge_speed_kph(self, props: Dict[str, Any], mode: str, default: float = 50.0) -> float:
        """
        Returns realistic travel speed (kph) = maxspeed × utilization_ratio.
        If OSM maxspeed tag is present, use it as the design speed.
        Otherwise fall back to the road-class assumed maxspeed.
        Utilization ratio accounts for signals, congestion, and turning —
        no one actually travels at the posted limit in HCMC.
        """
        # Get the design/legal maxspeed for this segment
        ms = _parse_maxspeed_kph(props.get("maxspeed"))
        if not ms or ms <= 0:
            hw = props.get("highway")
            if mode == "motorcycle":
                ms = float(HIGHWAY_FALLBACK_SPEED_MOTORCYCLE.get(hw, default))
            else:
                ms = float(HIGHWAY_FALLBACK_SPEED_CAR.get(hw, default))

        # Apply utilization ratio to get realistic travel speed
        ratio = SPEED_UTILIZATION_MOTORCYCLE if mode == "motorcycle" else SPEED_UTILIZATION_CAR
        return float(ms) * ratio

    def _allowed(self, props: Dict[str, Any], mode: str) -> bool:
        if mode == "motorcycle":
            if _parse_access(props.get("motorcycle")) is False:
                return False
            if _parse_access(props.get("motor_vehicle")) is False:
                return False
            return True
        # car
        if _parse_access(props.get("motorcar")) is False:
            return False
        if _parse_access(props.get("car")) is False:
            return False
        if _parse_access(props.get("motor_vehicle")) is False:
            return False
        return True

    def _oneway(self, props: Dict[str, Any], mode: str) -> str:
        """Return 'forward', 'reverse', or 'no'."""
        if mode == "motorcycle":
            ov = _parse_oneway_value(props.get("oneway:motorcycle"))
            if ov is not None:
                return ov
        return _parse_oneway_value(props.get("oneway")) or "no"

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        for _, row in self.roads.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty or geom.geom_type != "LineString":
                continue

            props = dict(row.drop(labels=["geometry"]))
            x1, y1 = geom.coords[0]
            x2, y2 = geom.coords[-1]
            u = self._nid(x1, y1)
            v = self._nid(x2, y2)

            length_m = float(geom.length)
            props["_length_m"] = length_m

            speed_kph = self._edge_speed_kph(props, mode="car")
            duration_s = length_m / (speed_kph * 1000.0 / 3600.0)

            # Precompute routing weights for both modes at build time.
            # _edge_weight is called thousands of times per query — doing it
            # once here eliminates repeated dict lookups and string parsing.
            hw = props.get("highway", "")
            oneway = _parse_oneway_value(props.get("oneway")) or "no"

            car_spd   = self._edge_speed_kph(props, mode="car")
            moto_spd  = self._edge_speed_kph(props, mode="motorcycle")
            car_base  = length_m / (car_spd  * 1000.0 / 3600.0) if car_spd  > 0 else float("inf")
            moto_base = length_m / (moto_spd * 1000.0 / 3600.0) if moto_spd > 0 else float("inf")

            car_pen  = CAR_ROAD_PENALTY.get(hw, 2.5)
            moto_pen = MOTORCYCLE_ROAD_PENALTY.get(hw, 1.3)

            car_allowed  = self._allowed(props, "car")
            moto_allowed = self._allowed(props, "motorcycle")

            # Forward edge weights
            car_w_fwd  = (car_base  * car_pen)  if car_allowed  and not math.isinf(car_pen)  else float("inf")
            moto_w_fwd = (moto_base * moto_pen) if moto_allowed and not math.isinf(moto_pen) else float("inf")
            # Oneway=forward blocks reverse; oneway=reverse blocks forward
            car_w_rev  = float("inf") if oneway == "forward"  else car_w_fwd
            moto_w_rev = float("inf") if oneway == "forward"  else moto_w_fwd
            car_w_fwd  = float("inf") if oneway == "reverse"  else car_w_fwd
            moto_w_fwd = float("inf") if oneway == "reverse"  else moto_w_fwd

            self.G.add_edge(
                u, v,
                geometry=geom,
                props=props,
                base_speed_kph=car_spd,
                length_m=length_m,
                base_duration_s=duration_s,
                is_forward_edge=True,
                is_connector=False,
                car_weight=car_w_fwd,
                moto_weight=moto_w_fwd,
            )
            self.G.add_edge(
                v, u,
                geometry=LineString(list(geom.coords)[::-1]),
                props=props,
                base_speed_kph=car_spd,
                length_m=length_m,
                base_duration_s=duration_s,
                is_forward_edge=False,
                is_connector=False,
                car_weight=car_w_rev,
                moto_weight=moto_w_rev,
            )

    # ------------------------------------------------------------------
    # Snapping
    # ------------------------------------------------------------------

    def _snap_node(self, lon: float, lat: float) -> int:
        """
        Project (lon, lat) → EPSG:3857, snap to the nearest road edge,
        inject a temporary node, and wire it up with oneway-aware connectors.

        FIX vs original:
        - Connector edges now respect the oneway direction of the host edge.
          A snap onto a oneway=forward edge only gets forward connectors
          (approach from u, depart toward v) so the router can't exit the
          wrong way through the snap point.
        - The snap cache is intentionally NOT used here for routing correctness;
          connectors are always rebuilt fresh and torn down after routing.
        """
        p = self._project_point_3857(lon, lat)
        nearest = self._tree.nearest(p)
        if nearest is None:
            raise ValueError("No edges to snap to.")

        if isinstance(nearest, (int, np.integer)):
            idx = int(nearest)
            nearest_geom = self._edge_geoms[idx]
            u, v, data = self._edge_meta[idx]
        else:
            nearest_geom = nearest
            idx = self._edge_geoms.index(nearest_geom)
            u, v, data = self._edge_meta[idx]

        snapped = nearest_geom.interpolate(nearest_geom.project(p))
        sn = self._nid(snapped.x, snapped.y)

        # Always rebuild connectors cleanly.
        self._clear_connector_edges(sn)
        self._connector_edges[sn] = []

        props = data.get("props", {})
        speed_kph = float(data.get("base_speed_kph", 35.0))

        def link(a: int, b: int, seg: LineString, fwd: bool):
            length_m = float(seg.length)
            if length_m < 1e-9:
                return
            dur_s = length_m / (speed_kph * 1000.0 / 3600.0)
            self.G.add_edge(
                a, b,
                geometry=seg,
                props=props,
                base_speed_kph=speed_kph,
                length_m=length_m,
                base_duration_s=dur_s,
                is_connector=True,
                is_forward_edge=fwd,
            )
            self._connector_edges[sn].append((a, b))

        ps = Point(snapped.x, snapped.y)
        pu = Point(self.G.nodes[u]["x"], self.G.nodes[u]["y"])
        pv = Point(self.G.nodes[v]["x"], self.G.nodes[v]["y"])

        # Determine the oneway direction of the host edge (mode-agnostic at snap time;
        # actual access enforcement happens in _edge_weight during routing).
        oneway = _parse_oneway_value(props.get("oneway")) or "no"

        if oneway == "forward":
            # Traffic flows u → v only.
            # Connectors: approach snap from u, depart snap toward v.
            link(u, sn, LineString([pu, ps]), fwd=True)   # u → sn  (enter)
            link(sn, v, LineString([ps, pv]), fwd=True)   # sn → v  (exit)
        elif oneway == "reverse":
            # Traffic flows v → u only.
            link(v, sn, LineString([pv, ps]), fwd=False)  # v → sn  (enter)
            link(sn, u, LineString([ps, pu]), fwd=False)  # sn → u  (exit)
        else:
            # Bidirectional – full four connectors as before.
            link(sn, u, LineString([ps, pu]), fwd=False)
            link(u, sn, LineString([pu, ps]), fwd=True)
            link(sn, v, LineString([ps, pv]), fwd=True)
            link(v, sn, LineString([pv, ps]), fwd=False)

        return sn

    # ------------------------------------------------------------------
    # Edge weight
    # ------------------------------------------------------------------

    def _edge_weight(self, u: int, v: int, mode: str) -> float:
        data = self.G.edges[u, v]
        # Connector edges (snap nodes) don't have precomputed weights — compute on the fly
        if data.get("is_connector", False):
            length_m = float(data.get("length_m", 0.0))
            if length_m <= 0:
                return float("inf")
            speed_kph = float(data.get("base_speed_kph", 35.0))
            return length_m / (speed_kph * 1000.0 / 3600.0)
        # Permanent edges — use precomputed weight (set at build time, O(1))
        key = "car_weight" if mode == "car" else "moto_weight"
        return float(data.get(key, float("inf")))

    # ------------------------------------------------------------------
    # Path utilities
    # ------------------------------------------------------------------

    def _path_to_linestring(self, path: List[int]) -> LineString:
        coords = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            seg = list(self.G.edges[u, v]["geometry"].coords)
            if i > 0:
                seg = seg[1:]
            coords.extend(seg)
        return LineString(coords)

    def _path_cost(self, path: List[int], mode: str) -> Tuple[float, float]:
        """
        Returns (distance_m, duration_s) using RAW travel time (no road penalties).
        Penalties are only used during pathfinding to choose the route —
        they must NOT inflate the reported travel time shown to the user.
        Uses precomputed base_duration_s where available for speed.
        """
        dist = 0.0
        dur = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            data = self.G.edges[u, v]
            # Check passability — inf penalized weight means blocked
            if math.isinf(self._edge_weight(u, v, mode)):
                return float("inf"), float("inf")
            length_m = float(data.get("length_m", 0.0))
            dist += length_m
            # Use precomputed raw duration where available (permanent edges)
            if not data.get("is_connector", False):
                dur += float(data.get("base_duration_s", 0.0))
            else:
                speed_kph = float(data.get("base_speed_kph", 35.0))
                dur += length_m / (speed_kph * 1000.0 / 3600.0) if speed_kph > 0 else 0.0
        return dist, dur

    @staticmethod
    def _path_edge_set(path: List[int]) -> frozenset:
        """Return a frozenset of (u, v) pairs representing the path's edges."""
        return frozenset((path[i], path[i + 1]) for i in range(len(path) - 1))

    @staticmethod
    def _jaccard_overlap(a: frozenset, b: frozenset) -> float:
        """Jaccard similarity between two edge-sets (0 = disjoint, 1 = identical)."""
        if not a and not b:
            return 1.0
        return len(a & b) / len(a | b)

    def _is_diverse_enough(self, new_edges: frozenset, accepted: List[RouteResult]) -> bool:
        """
        Return True if *new_edges* differs from every already-accepted route by
        at least DIVERSITY_THRESHOLD (fraction of non-overlapping edges).
        """
        for r in accepted:
            overlap = self._jaccard_overlap(new_edges, r.edge_set)
            if overlap > (1.0 - DIVERSITY_THRESHOLD):
                return False
        return True

    # ------------------------------------------------------------------
    # Public routing entry point
    # ------------------------------------------------------------------

    def route_top3(
        self,
        start_lon: float,
        start_lat: float,
        end_lon: float,
        end_lat: float,
        mode: str = "car",
        k: int = 3,
    ) -> List[RouteResult]:
        t0 = time.perf_counter()
        print(f"[route_engine] route_top3 start | mode={mode} k={k}")

        print("[route_engine] snapping start...")
        s = self._snap_node(start_lon, start_lat)
        print(f"[route_engine] snapping start done | node={s} | dt={time.perf_counter()-t0:.3f}s")

        t1 = time.perf_counter()
        print("[route_engine] snapping end...")
        t = self._snap_node(end_lon, end_lat)
        print(f"[route_engine] snapping end done | node={t} | dt={time.perf_counter()-t1:.3f}s")

        # Use precomputed weights directly — avoids method call overhead
        # on every edge relaxation during Dijkstra/Yen's.
        wkey = "car_weight" if mode == "car" else "moto_weight"
        def weight(u, v, d):
            if d.get("is_connector", False):
                length_m = d.get("length_m", 0.0)
                spd = d.get("base_speed_kph", 35.0)
                return length_m / (spd * 1000.0 / 3600.0) if spd > 0 and length_m > 0 else float("inf")
            return d.get(wkey, float("inf"))

        results: List[RouteResult] = []

        try:
            # ── Fast path: k == 1 ──────────────────────────────────────────
            if k <= 1:
                t2 = time.perf_counter()
                print("[route_engine] shortest_path...")
                try:
                    path = nx.shortest_path(self.G, s, t, weight=weight, method="dijkstra")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    print(f"[route_engine] no path | dt={time.perf_counter()-t0:.3f}s")
                    return []
                print(f"[route_engine] shortest_path done | hops={len(path)} | dt={time.perf_counter()-t2:.3f}s")

                dist, dur = self._path_cost(path, mode)
                if math.isinf(dur):
                    print("[route_engine] path blocked by access rules")
                    return []
                geom = self._path_to_linestring(path)
                edge_set = self._path_edge_set(path)
                results.append(RouteResult(rank=1, distance_m=dist, duration_s=dur,
                                           geometry_3857=geom, edge_set=edge_set))
                print(f"[route_engine] done | routes=1 | total_dt={time.perf_counter()-t0:.3f}s")
                return results

            # ── k > 1: Edge-penalty alternative routing ────────────────────
            # Yen's algorithm (shortest_simple_paths) is O(kN³) — too slow
            # on large graphs (137k+ nodes). Instead we use a much faster
            # approach: run Dijkstra for the best route, then heavily penalize
            # those edges and re-run to force the next path onto a different
            # corridor. Each iteration is just one Dijkstra = O(E log N).

            MAX_ALTERNATIVES     = 2
            ALT_MAX_DIST_GAP_M   = 1000.0
            ALT_MAX_DUR_GAP_S    = 60.0
            ALT_MAX_GEOM_OVERLAP = 0.50
            # Penalty multiplier applied to edges used by an accepted route.
            # High enough to force a detour, but not so high that the graph
            # becomes disconnected and returns no path.
            ALT_PENALTY_MULT     = 8.0
            wkey = "car_weight" if mode == "car" else "moto_weight"

            def _dijkstra(penalized_edges: dict) -> Optional[List[int]]:
                """Run Dijkstra with optional per-edge penalty overrides."""
                def w(u, v, d):
                    if d.get("is_connector", False):
                        length_m = d.get("length_m", 0.0)
                        spd = d.get("base_speed_kph", 35.0)
                        return length_m / (spd * 1000.0 / 3600.0) if spd > 0 and length_m > 0 else float("inf")
                    base = d.get(wkey, float("inf"))
                    if math.isinf(base):
                        return base
                    return base * penalized_edges.get((u, v), 1.0)
                try:
                    return nx.shortest_path(self.G, s, t, weight=w, method="dijkstra")
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    return None

            t2 = time.perf_counter()

            # ── Step 1: best route ─────────────────────────────────────────
            path = _dijkstra({})
            if path is None:
                print(f"[route_engine] no path | dt={time.perf_counter()-t0:.3f}s")
                return []

            dist, dur = self._path_cost(path, mode)
            if math.isinf(dur):
                return []

            best_edges = self._path_edge_set(path)
            results.append(RouteResult(rank=1, distance_m=dist, duration_s=dur,
                                       geometry_3857=self._path_to_linestring(path),
                                       edge_set=best_edges))
            print(f"[route_engine] best route | dist={dist:.0f}m dur={dur:.0f}s "
                  f"| dt={time.perf_counter()-t2:.3f}s")

            # ── Step 2: alternatives via edge penalization ─────────────────
            penalized: dict = {}
            for iteration in range(MAX_ALTERNATIVES):
                # Penalize ALL edges used by every accepted route so far
                penalized = {}
                for r in results:
                    for u2, v2 in r.edge_set:
                        penalized[(u2, v2)] = ALT_PENALTY_MULT

                t_alt = time.perf_counter()
                alt_path = _dijkstra(penalized)
                if alt_path is None:
                    print(f"[route_engine] no alt path at iteration {iteration+1}")
                    break

                alt_dist, alt_dur = self._path_cost(alt_path, mode)
                if math.isinf(alt_dur):
                    break

                alt_edges = self._path_edge_set(alt_path)

                dist_gap = alt_dist - dist
                dur_gap  = alt_dur  - dur

                if dist_gap >= ALT_MAX_DIST_GAP_M:
                    print(f"[route_engine] alt {iteration+1} rejected: dist_gap={dist_gap:.0f}m")
                    break

                if dur_gap >= ALT_MAX_DUR_GAP_S:
                    print(f"[route_engine] alt {iteration+1} rejected: dur_gap={dur_gap:.0f}s")
                    continue

                ov = self._jaccard_overlap(alt_edges, best_edges)
                if ov >= ALT_MAX_GEOM_OVERLAP:
                    print(f"[route_engine] alt {iteration+1} rejected: overlap={ov:.2f} (not different enough)")
                    continue

                results.append(RouteResult(rank=len(results)+1,
                                           distance_m=alt_dist, duration_s=alt_dur,
                                           geometry_3857=self._path_to_linestring(alt_path),
                                           edge_set=alt_edges))
                print(f"[route_engine] alt {len(results)-1} accepted | "
                      f"dist_gap={dist_gap:.0f}m dur_gap={dur_gap:.0f}s ov={ov:.2f} "
                      f"| dt={time.perf_counter()-t_alt:.3f}s")

            print(f"[route_engine] done | routes={len(results)} "
                  f"| total_dt={time.perf_counter()-t0:.3f}s")
            return results

        finally:
            # ── CRITICAL: always clean up connector edges after routing ────
            # This prevents ghost edges from accumulating across requests.
            snap_nodes = [sn for sn in [s, t] if sn in self._connector_edges]
            self._clear_all_connector_edges(snap_nodes)
            print(f"[route_engine] connector edges cleaned up for nodes {snap_nodes}")
