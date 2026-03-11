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

HIGHWAY_FALLBACK_SPEED_CAR = {
    "motorway": 80,
    "trunk": 70,
    "primary": 60,
    "secondary": 50,
    "tertiary": 40,
    "unclassified": 40,
    "residential": 30,
    "service": 20,
    "living_street": 15,
}

HIGHWAY_FALLBACK_SPEED_MOTORCYCLE = {
    "motorway": 60,
    "trunk": 55,
    "primary": 45,
    "secondary": 40,
    "tertiary": 35,
    "unclassified": 35,
    "residential": 28,
    "service": 18,
    "living_street": 12,
}

# Minimum fraction of edges that must differ between two routes.
# e.g. 0.20 means routes must share at most 80% of their edge-sets.
DIVERSITY_THRESHOLD = 0.20

# Timeout in seconds for the k-shortest-paths generator loop.
ROUTE_TIMEOUT_S = 8.0


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

    def _edge_speed_kph(self, props: Dict[str, Any], mode: str, default: float = 35.0) -> float:
        ms = _parse_maxspeed_kph(props.get("maxspeed"))
        if ms and ms > 0:
            if mode == "motorcycle":
                return float(min(ms, 60.0))
            return float(ms)

        hw = props.get("highway")
        if mode == "motorcycle":
            return float(HIGHWAY_FALLBACK_SPEED_MOTORCYCLE.get(hw, min(default, 30.0)))
        return float(HIGHWAY_FALLBACK_SPEED_CAR.get(hw, default))

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

            self.G.add_edge(
                u, v,
                geometry=geom,
                props=props,
                base_speed_kph=speed_kph,
                length_m=length_m,
                base_duration_s=duration_s,
                is_forward_edge=True,
                is_connector=False,
            )
            self.G.add_edge(
                v, u,
                geometry=LineString(list(geom.coords)[::-1]),
                props=props,
                base_speed_kph=speed_kph,
                length_m=length_m,
                base_duration_s=duration_s,
                is_forward_edge=False,
                is_connector=False,
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
        props = data.get("props", {})

        if not self._allowed(props, mode):
            return float("inf")

        if not data.get("is_connector", False):
            oneway = self._oneway(props, mode)
            is_forward = bool(data.get("is_forward_edge", True))
            if oneway == "forward" and not is_forward:
                return float("inf")
            if oneway == "reverse" and is_forward:
                return float("inf")

        length_m = float(data.get("length_m", 0.0))
        if length_m <= 0:
            return float("inf")

        speed_kph = self._edge_speed_kph(props, mode=mode)
        if speed_kph <= 0:
            return float("inf")

        return length_m / (speed_kph * 1000.0 / 3600.0)

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
        dist = 0.0
        dur = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            data = self.G.edges[u, v]
            w = self._edge_weight(u, v, mode)
            if math.isinf(w):
                return float("inf"), float("inf")
            dist += float(data.get("length_m", 0.0))
            dur += float(w)
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

        def weight(u, v, d):
            return self._edge_weight(u, v, mode)

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

            # ── k > 1: Yen's k-shortest with timeout + diversity filter ────
            t2 = time.perf_counter()
            print("[route_engine] shortest_simple_paths...")
            try:
                gen = nx.shortest_simple_paths(self.G, s, t, weight=weight)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                print(f"[route_engine] no path | dt={time.perf_counter()-t0:.3f}s")
                return []
            print(f"[route_engine] generator ready | dt={time.perf_counter()-t2:.3f}s")

            deadline = time.perf_counter() + ROUTE_TIMEOUT_S
            candidates_checked = 0

            # We over-fetch candidates to allow diversity filtering to select
            # the best k diverse routes (cap at 5×k to bound runtime).
            max_candidates = k * 5

            while len(results) < k and candidates_checked < max_candidates:
                if time.perf_counter() > deadline:
                    print(f"[route_engine] timeout after {ROUTE_TIMEOUT_S}s — returning {len(results)} route(s)")
                    break
                try:
                    path = next(gen)
                except (StopIteration, nx.NetworkXNoPath):
                    break

                candidates_checked += 1
                dist, dur = self._path_cost(path, mode)
                if math.isinf(dur):
                    continue

                edge_set = self._path_edge_set(path)
                if not self._is_diverse_enough(edge_set, results):
                    continue

                geom = self._path_to_linestring(path)
                results.append(RouteResult(
                    rank=len(results) + 1,
                    distance_m=dist,
                    duration_s=dur,
                    geometry_3857=geom,
                    edge_set=edge_set,
                ))

            print(f"[route_engine] done | routes={len(results)} candidates_checked={candidates_checked} "
                  f"| total_dt={time.perf_counter()-t0:.3f}s")
            return results

        finally:
            # ── CRITICAL: always clean up connector edges after routing ────
            # This prevents ghost edges from accumulating across requests.
            snap_nodes = [sn for sn in [s, t] if sn in self._connector_edges]
            self._clear_all_connector_edges(snap_nodes)
            print(f"[route_engine] connector edges cleaned up for nodes {snap_nodes}")
