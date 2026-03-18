from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import json
import math
import os
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

SPEED_UTILIZATION_CAR = 0.50
SPEED_UTILIZATION_MOTORCYCLE = 0.65

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

CAR_ROAD_PENALTY = {
    "motorway":      0.5,
    "trunk":         0.6,
    "primary":       0.8,
    "secondary":     1.0,
    "tertiary":      1.3,
    "unclassified":  2.0,
    "residential":   3.0,
    "service":       5.0,
    "living_street": 8.0,
    "track":        15.0,
    "path":         20.0,
    "footway":      50.0,
    "cycleway":     50.0,
}

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
    edge_set: frozenset


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RouteEngine:
    def __init__(self, roads_gdf_3857: gpd.GeoDataFrame, restrictions_path: str = None):
        self.roads = roads_gdf_3857
        self.G = nx.DiGraph()
        self._node_id: Dict[Tuple[float, float], int] = {}
        self._node_xy: Dict[int, Tuple[float, float]] = {}
        self._project_cache: Dict[Tuple[float, float], Point] = {}
        self._snap_cache: Dict[Tuple[float, float], int] = {}
        self._connector_edges: Dict[int, List[Tuple[int, int]]] = {}

        # OSM way ID → list of graph edge (u, v) pairs belonging to that way
        self._way_edges: Dict[int, List[Tuple[int, int]]] = {}
        # (via_node, from_u, from_v) → set of blocked (to_u, to_v)
        self._blocked_turns: Dict[Tuple, set] = {}
        # (via_node, from_u, from_v) → set of ONLY-allowed (to_u, to_v)
        self._only_turns: Dict[Tuple, set] = {}

        self._build_graph()
        self._load_restrictions(restrictions_path)

        # Spatial index — forward edges only (same bbox as reverse, half the size)
        t1 = time.time()
        self._edge_geoms: List[LineString] = []
        self._edge_meta: List[Tuple[int, int, dict]] = []
        for u, v, data in self.G.edges(data=True):
            if data.get("is_forward_edge", True) and not data.get("is_connector", False):
                self._edge_geoms.append(data["geometry"])
                self._edge_meta.append((u, v, data))
        self._tree = STRtree(self._edge_geoms)
        print(f"[route_engine] STRtree built | {len(self._edge_geoms):,} edges | {time.time()-t1:.1f}s")

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
        edges = self._connector_edges.pop(sn, [])
        for u, v in edges:
            if self.G.has_edge(u, v):
                self.G.remove_edge(u, v)

    def _clear_all_connector_edges(self, snap_nodes: List[int]) -> None:
        for sn in snap_nodes:
            self._clear_connector_edges(sn)
            self._snap_cache = {k: v for k, v in self._snap_cache.items() if v != sn}

    # ------------------------------------------------------------------
    # Speed / access / oneway helpers
    # ------------------------------------------------------------------

    def _edge_speed_kph(self, props: Dict[str, Any], mode: str, default: float = 50.0) -> float:
        ms = _parse_maxspeed_kph(props.get("maxspeed"))
        if not ms or ms <= 0:
            hw = props.get("highway")
            if mode == "motorcycle":
                ms = float(HIGHWAY_FALLBACK_SPEED_MOTORCYCLE.get(hw, default))
            else:
                ms = float(HIGHWAY_FALLBACK_SPEED_CAR.get(hw, default))
        ratio = SPEED_UTILIZATION_MOTORCYCLE if mode == "motorcycle" else SPEED_UTILIZATION_CAR
        return float(ms) * ratio

    def _allowed(self, props: Dict[str, Any], mode: str) -> bool:
        hw = props.get("highway", "")
        if hw in ("footway", "steps", "path", "cycleway", "pedestrian"):
            return False
        if mode == "motorcycle":
            if _parse_access(props.get("motorcycle")) is False:
                return False
            mv = props.get("motor_vehicle", "")
            if str(mv).strip().lower() in ("no", "private"):
                return False
            return True
        if _parse_access(props.get("motorcar")) is False:
            return False
        mv = props.get("motor_vehicle", "")
        if str(mv).strip().lower() in ("no", "private"):
            return False
        return True

    def _oneway(self, props: Dict[str, Any], mode: str) -> str:
        if mode == "motorcycle":
            ov = _parse_oneway_value(props.get("oneway:motorcycle"))
            if ov is not None:
                return ov
        return _parse_oneway_value(props.get("oneway")) or "no"

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        # to_dict("records") is 10-20x faster than iterrows() for large DataFrames.
        # Each iterrows() call allocates a new Series object; records() gives plain dicts.
        t0 = time.time()
        records = self.roads.to_dict("records")
        print(f"[route_engine] _build_graph: {len(records):,} rows | to_dict done in {time.time()-t0:.1f}s")

        fwd_edges: list = []
        rev_edges: list = []
        n_total   = len(records)
        report_every = max(1, n_total // 10)   # log progress 10 times

        for idx, row in enumerate(records):
            if idx % report_every == 0:
                pct = idx / n_total * 100
                print(f"[route_engine] _build_graph {idx:,}/{n_total:,} "
                      f"({pct:.0f}%) | {time.time()-t0:.0f}s elapsed")

            geom = row.get("geometry")
            if geom is None or geom.is_empty or geom.geom_type != "LineString":
                continue

            props = {k: v for k, v in row.items() if k != "geometry"}

            # Extract OSM way ID for turn restriction lookup
            osm_id = None
            for id_key in ("osm_id", "@id", "id", "@osm_id"):
                val = props.get(id_key)
                if val is not None:
                    try:
                        osm_id = int(str(val).replace("way/", "").strip())
                        break
                    except (ValueError, TypeError):
                        pass

            x1, y1 = geom.coords[0]
            x2, y2 = geom.coords[-1]
            u = self._nid(x1, y1)
            v = self._nid(x2, y2)

            length_m = float(geom.length)
            props["_length_m"] = length_m

            hw     = props.get("highway", "")
            oneway = _parse_oneway_value(props.get("oneway")) or "no"

            car_spd   = self._edge_speed_kph(props, mode="car")
            moto_spd  = self._edge_speed_kph(props, mode="motorcycle")
            car_base  = length_m / (car_spd  * 1000.0 / 3600.0) if car_spd  > 0 else float("inf")
            moto_base = length_m / (moto_spd * 1000.0 / 3600.0) if moto_spd > 0 else float("inf")

            car_pen  = CAR_ROAD_PENALTY.get(hw, 2.5)
            moto_pen = MOTORCYCLE_ROAD_PENALTY.get(hw, 1.3)

            car_allowed  = self._allowed(props, "car")
            moto_allowed = self._allowed(props, "motorcycle")

            car_w_fwd  = (car_base  * car_pen)  if car_allowed  else float("inf")
            moto_w_fwd = (moto_base * moto_pen) if moto_allowed else float("inf")
            car_w_rev  = float("inf") if oneway == "forward" else car_w_fwd
            moto_w_rev = float("inf") if oneway == "forward" else moto_w_fwd
            car_w_fwd  = float("inf") if oneway == "reverse" else car_w_fwd
            moto_w_fwd = float("inf") if oneway == "reverse" else moto_w_fwd

            # Reverse geometry is stored as coords tuple — resolved to LineString
            # only when actually needed, saving 6.5M eager LineString allocations.
            fwd_edges.append((u, v, dict(
                geometry=geom,
                props=props,
                osm_id=osm_id,
                base_speed_kph=car_spd,
                length_m=length_m,
                base_duration_s=car_base,
                is_forward_edge=True,
                is_connector=False,
                car_weight=car_w_fwd,
                moto_weight=moto_w_fwd,
            )))
            rev_edges.append((v, u, dict(
                geometry=geom,          # same object — reversed on demand via is_forward_edge=False
                props=props,
                osm_id=osm_id,
                base_speed_kph=car_spd,
                length_m=length_m,
                base_duration_s=car_base,
                is_forward_edge=False,
                is_connector=False,
                car_weight=car_w_rev,
                moto_weight=moto_w_rev,
            )))

            if osm_id is not None:
                we = self._way_edges.setdefault(osm_id, [])
                we.append((u, v))
                we.append((v, u))

        t1 = time.time()
        print(f"[route_engine] _build_graph loop done in {t1-t0:.1f}s — adding edges to graph…")

        # Batch add — much faster than 13M individual G.add_edge() calls
        self.G.add_edges_from(fwd_edges)
        self.G.add_edges_from(rev_edges)
        del fwd_edges, rev_edges   # free ~2 GB before STRtree

        print(f"[route_engine] _build_graph done | "
              f"nodes={self.G.number_of_nodes():,} "
              f"edges={self.G.number_of_edges():,} | "
              f"total={time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Turn restriction loader
    # ------------------------------------------------------------------

    def _load_restrictions(self, restrictions_path: str = None) -> None:
        """
        Load turn restrictions from JSON and build two lookup tables:
          _blocked_turns:  (via_node, from_u, from_v) → {(to_u, to_v), ...}
          _only_turns:     (via_node, from_u, from_v) → {(to_u, to_v), ...}

        Supports:
          - via_nodes: simple intersection (most common)
          - via_ways:  complex intersection via a road segment
          - no_* restrictions: block specific turns
          - only_* restrictions: block all turns except specified
        """
        if restrictions_path is None or not os.path.exists(restrictions_path):
            if restrictions_path:
                print(f"[route_engine] restrictions file not found: {restrictions_path}")
            else:
                print("[route_engine] no restrictions file — turn restrictions disabled")
            return

        with open(restrictions_path, encoding="utf-8") as f:
            restrictions = json.load(f)

        applied = skipped = 0

        for r in restrictions:
            rtype        = r.get("restriction", "")
            from_way_ids = r.get("from_ways", [])
            via_node_ids = r.get("via_nodes", [])
            via_way_ids  = r.get("via_ways", [])
            to_way_ids   = r.get("to_ways", [])

            if not rtype or not from_way_ids or not to_way_ids:
                skipped += 1
                continue

            # Collect graph edges for from/to ways
            from_edges = []
            for wid in from_way_ids:
                from_edges.extend(self._way_edges.get(wid, []))

            to_edges = []
            for wid in to_way_ids:
                to_edges.extend(self._way_edges.get(wid, []))

            if not from_edges or not to_edges:
                skipped += 1
                continue

            # --- Resolve via nodes in graph coordinates ---
            via_graph_nodes = set()

            if via_node_ids:
                # via=node: the shared endpoint between from_edges and to_edges
                # that is closest to the intersection
                from_nodes = {n for e in from_edges for n in e}
                to_nodes   = {n for e in to_edges   for n in e}
                shared = from_nodes & to_nodes
                via_graph_nodes.update(shared)

            if via_way_ids:
                # via=way: find nodes that connect from_edges → via_way → to_edges
                for via_wid in via_way_ids:
                    via_edges = self._way_edges.get(via_wid, [])
                    if not via_edges:
                        continue
                    via_nodes_set = {n for e in via_edges for n in e}
                    from_nodes = {n for e in from_edges for n in e}
                    to_nodes   = {n for e in to_edges   for n in e}
                    # Entry node: shared between from_edges and via_way
                    entry = from_nodes & via_nodes_set
                    # Exit node: shared between via_way and to_edges
                    exit_ = via_nodes_set & to_nodes
                    # The restriction applies at the exit node
                    via_graph_nodes.update(exit_)

            if not via_graph_nodes:
                skipped += 1
                continue

            is_no   = rtype.startswith("no_")
            is_only = rtype.startswith("only_")

            for via_node in via_graph_nodes:
                arriving  = [(u, v) for u, v in from_edges if v == via_node]
                departing = [(u, v) for u, v in to_edges   if u == via_node]

                if not arriving or not departing:
                    continue

                for from_edge in arriving:
                    key = (via_node, from_edge[0], from_edge[1])
                    if is_no:
                        if key not in self._blocked_turns:
                            self._blocked_turns[key] = set()
                        for to_edge in departing:
                            self._blocked_turns[key].add(to_edge)
                    elif is_only:
                        if key not in self._only_turns:
                            self._only_turns[key] = set()
                        for to_edge in departing:
                            self._only_turns[key].add(to_edge)

            applied += 1

        print(f"[route_engine] restrictions loaded | "
              f"applied={applied} skipped={skipped} "
              f"blocked_keys={len(self._blocked_turns)} "
              f"only_keys={len(self._only_turns)}")

    # ------------------------------------------------------------------
    # Turn restriction check
    # ------------------------------------------------------------------

    def _turn_allowed(self, prev_u: int, prev_v: int, next_u: int, next_v: int) -> bool:
        """
        Return False if the turn from edge (prev_u→prev_v) to (next_u→next_v)
        is blocked by a restriction at via_node = prev_v = next_u.
        """
        via_node = prev_v
        if via_node != next_u:
            return True

        key = (via_node, prev_u, prev_v)

        blocked = self._blocked_turns.get(key)
        if blocked and (next_u, next_v) in blocked:
            return False

        only = self._only_turns.get(key)
        if only and (next_u, next_v) not in only:
            return False

        return True

    # ------------------------------------------------------------------
    # Snapping
    # ------------------------------------------------------------------

    def _snap_node(self, lon: float, lat: float) -> int:
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

        oneway = _parse_oneway_value(props.get("oneway")) or "no"

        if oneway == "forward":
            link(u, sn, LineString([pu, ps]), fwd=True)
            link(sn, v, LineString([ps, pv]), fwd=True)
        elif oneway == "reverse":
            link(v, sn, LineString([pv, ps]), fwd=False)
            link(sn, u, LineString([ps, pu]), fwd=False)
        else:
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
        if data.get("is_connector", False):
            length_m = float(data.get("length_m", 0.0))
            if length_m <= 0:
                return float("inf")
            speed_kph = float(data.get("base_speed_kph", 35.0))
            return length_m / (speed_kph * 1000.0 / 3600.0)
        key = "car_weight" if mode == "car" else "moto_weight"
        return float(data.get(key, float("inf")))

    # ------------------------------------------------------------------
    # Path utilities
    # ------------------------------------------------------------------

    def _path_to_linestring(self, path: List[int]) -> LineString:
        coords = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            data = self.G.edges[u, v]
            seg  = list(data["geometry"].coords)
            # Reverse edges share the forward geometry object — flip coords here
            if not data.get("is_forward_edge", True):
                seg = seg[::-1]
            if i > 0:
                seg = seg[1:]   # drop first coord to avoid duplicates at joins
            coords.extend(seg)
        return LineString(coords)

    def _path_cost(self, path: List[int], mode: str) -> Tuple[float, float]:
        dist = 0.0
        dur = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            data = self.G.edges[u, v]
            if math.isinf(self._edge_weight(u, v, mode)):
                return float("inf"), float("inf")
            length_m = float(data.get("length_m", 0.0))
            dist += length_m
            if not data.get("is_connector", False):
                dur += float(data.get("base_duration_s", 0.0))
            else:
                speed_kph = float(data.get("base_speed_kph", 35.0))
                dur += length_m / (speed_kph * 1000.0 / 3600.0) if speed_kph > 0 else 0.0
        return dist, dur

    @staticmethod
    def _path_edge_set(path: List[int]) -> frozenset:
        return frozenset((path[i], path[i + 1]) for i in range(len(path) - 1))

    @staticmethod
    def _jaccard_overlap(a: frozenset, b: frozenset) -> float:
        if not a and not b:
            return 1.0
        return len(a & b) / len(a | b)

    def _is_diverse_enough(self, new_edges: frozenset, accepted: List[RouteResult]) -> bool:
        for r in accepted:
            overlap = self._jaccard_overlap(new_edges, r.edge_set)
            if overlap > (1.0 - DIVERSITY_THRESHOLD):
                return False
        return True

    # ------------------------------------------------------------------
    # Turn-aware A* (used when restrictions are loaded)
    # ------------------------------------------------------------------

    def _astar_with_turns(
        self,
        source: int,
        target: int,
        wkey: str,
        heuristic,
        penalized_edges: dict = None,
    ) -> Optional[List[int]]:
        """
        Custom A* that enforces turn restrictions by tracking previous node.

        Standard nx.astar_path cannot enforce turn restrictions because its
        weight function is stateless (only sees current edge, not arrival direction).
        This tracks (node, prev_node) state so we can call _turn_allowed at each step.

        State in heap: (f_score, g_score, node, prev_node, path)
        """
        import heapq

        INF = float("inf")
        penalized = penalized_edges or {}

        def edge_cost(u, v, d):
            if d.get("is_connector", False):
                length_m = d.get("length_m", 0.0)
                spd = d.get("base_speed_kph", 35.0)
                return length_m / (spd * 1000.0 / 3600.0) if spd > 0 and length_m > 0 else INF
            base = d.get(wkey, INF)
            if math.isinf(base):
                return base
            return base * penalized.get((u, v), 1.0)

        tx, ty = self._node_xy.get(target, (0.0, 0.0))
        max_speed_ms = 80.0 * 1000.0 / 3600.0

        def h(node):
            nx_, ny = self._node_xy.get(node, (0.0, 0.0))
            return math.hypot(nx_ - tx, ny - ty) / max_speed_ms

        # heap: (f, g, node, prev_node, path)
        heap = [(h(source), 0.0, source, None, [source])]
        # visited: (node, prev_node) → best g seen
        visited: Dict[Tuple, float] = {}

        while heap:
            f, g, node, prev, path = heapq.heappop(heap)

            state = (node, prev)
            if state in visited and visited[state] <= g:
                continue
            visited[state] = g

            if node == target:
                return path

            for nbr, edge_data in self.G[node].items():
                # Enforce turn restriction: prev → node → nbr
                if prev is not None and not self._turn_allowed(prev, node, node, nbr):
                    continue

                cost = edge_cost(node, nbr, edge_data)
                if math.isinf(cost):
                    continue

                new_g = g + cost
                new_state = (nbr, node)
                if new_state in visited and visited[new_state] <= new_g:
                    continue

                new_f = new_g + h(nbr)
                heapq.heappush(heap, (new_f, new_g, nbr, node, path + [nbr]))

        return None

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

        wkey = "car_weight" if mode == "car" else "moto_weight"
        has_restrictions = bool(self._blocked_turns or self._only_turns)

        # A* heuristic — admissible lower bound on travel time
        tx, ty = self._node_xy.get(t, (0.0, 0.0))
        max_speed_ms = 80.0 * 1000.0 / 3600.0

        def heuristic(u, _t):
            ux, uy = self._node_xy.get(u, (0.0, 0.0))
            return math.hypot(ux - tx, uy - ty) / max_speed_ms

        def weight(u, v, d):
            if d.get("is_connector", False):
                length_m = d.get("length_m", 0.0)
                spd = d.get("base_speed_kph", 35.0)
                return length_m / (spd * 1000.0 / 3600.0) if spd > 0 and length_m > 0 else float("inf")
            return d.get(wkey, float("inf"))

        def _astar(penalized_edges: dict) -> Optional[List[int]]:
            """
            A* routing — uses turn-aware implementation when restrictions are loaded,
            falls back to fast nx.astar_path when no restrictions exist.
            """
            if has_restrictions:
                return self._astar_with_turns(
                    source=s, target=t,
                    wkey=wkey,
                    heuristic=heuristic,
                    penalized_edges=penalized_edges,
                )
            # No restrictions — use fast NetworkX built-in
            if not penalized_edges:
                try:
                    return nx.astar_path(self.G, s, t, heuristic=heuristic, weight=weight)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    return None
            else:
                def w_pen(u, v, d):
                    if d.get("is_connector", False):
                        length_m = d.get("length_m", 0.0)
                        spd = d.get("base_speed_kph", 35.0)
                        return length_m / (spd * 1000.0 / 3600.0) if spd > 0 and length_m > 0 else float("inf")
                    base = d.get(wkey, float("inf"))
                    if math.isinf(base):
                        return base
                    return base * penalized_edges.get((u, v), 1.0)
                try:
                    return nx.astar_path(self.G, s, t, heuristic=heuristic, weight=w_pen)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    return None

        results: List[RouteResult] = []

        try:
            # ── Fast path: k == 1 ─────────────────────────────────────────
            if k <= 1:
                t2 = time.perf_counter()
                print("[route_engine] astar k=1...")
                path = _astar({})
                if path is None:
                    print(f"[route_engine] no path | dt={time.perf_counter()-t0:.3f}s")
                    return []
                print(f"[route_engine] astar done | hops={len(path)} | dt={time.perf_counter()-t2:.3f}s")
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

            # ── k > 1: A* + edge-penalty alternative routing ──────────────
            MAX_ALTERNATIVES     = 2
            ALT_MAX_DIST_GAP_M   = 5000.0
            ALT_MAX_DUR_GAP_S    = 600.0
            ALT_MAX_GEOM_OVERLAP = 0.50
            ALT_PENALTY_MULT     = 8.0

            t2 = time.perf_counter()

            # Step 1: best route
            path = _astar({})
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

            # Step 2: alternatives via edge penalization
            for iteration in range(MAX_ALTERNATIVES):
                penalized = {}
                for r in results:
                    for u2, v2 in r.edge_set:
                        penalized[(u2, v2)] = ALT_PENALTY_MULT

                t_alt = time.perf_counter()
                alt_path = _astar(penalized)
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
                    break  # penalization unchanged → next iter finds identical path, skip it

                ov = self._jaccard_overlap(alt_edges, best_edges)
                if ov >= ALT_MAX_GEOM_OVERLAP:
                    print(f"[route_engine] alt {iteration+1} rejected: overlap={ov:.2f}")
                    break  # same reason: penalization unchanged next iter

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
            snap_nodes = [sn for sn in [s, t] if sn in self._connector_edges]
            self._clear_all_connector_edges(snap_nodes)
            print(f"[route_engine] connector edges cleaned up for nodes {snap_nodes}")
