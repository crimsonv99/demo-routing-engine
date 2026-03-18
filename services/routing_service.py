from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import Point, mapping
from shapely.strtree import STRtree

from config import Settings
from routing_engine import RouteEngine, RouteResult
from preprocess import node_roads_preserve_attrs
from poi_loader import load_pois_csv
from schemas import (
    LatLon, PoiInfo, RouteFeature, RouteProperties,
    RouteSummary, GeoJSONGeometry, RouteResponse, SnapInfo, TripRecord,
)

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _json_safe(v: Any) -> Any:
    if v is None:
        return None
    if hasattr(v, "item") and callable(getattr(v, "item")):
        try:
            v = v.item()
        except Exception:
            pass
    if isinstance(v, (float, np.floating)):
        if math.isnan(float(v)) or math.isinf(float(v)):
            return None
        return float(v)
    return v


def _bearing_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    import math as _math
    dx, dy = x2 - x1, y2 - y1
    return _math.degrees(_math.atan2(dx, dy)) % 360


def _bearing_to_cardinal(deg: float) -> str:
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[round(deg / 45) % 8]


def _turn_instruction(prev_b: float, next_b: float) -> str:
    diff = (next_b - prev_b + 360) % 360
    if diff < 20 or diff > 340:    return "Continue straight"
    elif diff < 80:                return "Turn slight right"
    elif diff < 150:               return "Turn right"
    elif diff < 200:               return "Make a U-turn"
    elif diff < 260:               return "Turn left"
    else:                          return "Turn slight left"


def _build_instructions(coords_ll: list, road_names: list) -> List[str]:
    if len(coords_ll) < 2:
        return []
    instructions: List[str] = []
    first_b = _bearing_deg(coords_ll[0][0], coords_ll[0][1],
                           coords_ll[1][0], coords_ll[1][1])
    name_str = f" on {road_names[0]}" if road_names and road_names[0] else ""
    instructions.append(f"Head {_bearing_to_cardinal(first_b)}{name_str}")

    for i in range(1, len(coords_ll) - 1):
        b_in  = _bearing_deg(*coords_ll[i-1], *coords_ll[i])
        b_out = _bearing_deg(*coords_ll[i],   *coords_ll[i+1])
        turn  = _turn_instruction(b_in, b_out)
        if turn == "Continue straight":
            continue
        road_after = road_names[i] if i < len(road_names) and road_names[i] else ""
        onto = f" onto {road_after}" if road_after else ""
        instructions.append(f"{turn}{onto} ({coords_ll[i][1]:.5f}, {coords_ll[i][0]:.5f})")

    end = coords_ll[-1]
    instructions.append(f"Arrive at destination ({end[1]:.5f}, {end[0]:.5f})")
    return instructions


def _hits_to_indices(hits: Any, id_to_idx: Dict[int, int]) -> List[int]:
    out = []
    for h in hits:
        if isinstance(h, (int, np.integer)):
            out.append(int(h))
        else:
            idx = id_to_idx.get(id(h))
            if idx is not None:
                out.append(int(idx))
    return out


# ── RoutingService ────────────────────────────────────────────────────────────

class RoutingService:
    """
    Owns all stateful geo objects (engine, POI trees) and exposes
    clean methods consumed by the API routers.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._trips: Dict[str, TripRecord] = {}
        self._load_roads()
        self._load_pois()
        logger.info(
            "RoutingService ready | roads=%d pois=%d",
            len(self.engine.roads), len(self.pois),
        )

    # ── Initialisation ────────────────────────────────────────────────────

    def _load_roads(self) -> None:
        import os
        logger.info("Loading roads from %s", self.settings.roads_path)
        roads = gpd.read_file(self.settings.roads_path).to_crs(epsg=3857)
        # already_split=True when loading pre-noded file (from split_lines_standalone.py)
        # skips the 20-min intersection pass — does vectorised cleanup only (~seconds)
        already_split = "split" in self.settings.roads_path.lower()
        roads_noded = node_roads_preserve_attrs(roads, already_split=already_split)

        # Use RESTRICTIONS_PATH from .env if set, otherwise disabled
        restrictions_path = self.settings.restrictions_path or None
        if restrictions_path and os.path.exists(restrictions_path):
            logger.info("Found restrictions: %s", restrictions_path)
        elif restrictions_path:
            logger.warning("Restrictions file not found: %s — turn restrictions disabled", restrictions_path)
            restrictions_path = None
        else:
            logger.info("RESTRICTIONS_PATH not set — turn restrictions disabled")

        self.engine = RouteEngine(roads_noded, restrictions_path=restrictions_path)
        logger.info("Road graph ready | nodes=%d edges=%d",
                    self.engine.G.number_of_nodes(),
                    self.engine.G.number_of_edges())

    def _load_pois(self) -> None:
        logger.info("Loading POIs from %s", self.settings.pois_path)
        pois = load_pois_csv(self.settings.pois_path)
        if str(getattr(pois, "crs", "")) != "EPSG:4326":
            pois = pois.to_crs("EPSG:4326")
        self.pois = pois

        self._poi_geoms = list(pois.geometry)
        self._poi_tree  = STRtree(self._poi_geoms)

        pois_3857 = pois.to_crs("EPSG:3857")
        self._poi_geoms_3857    = list(pois_3857.geometry)
        self._poi_tree_3857     = STRtree(self._poi_geoms_3857)
        self._poi_id_to_idx     = {id(g): i for i, g in enumerate(self._poi_geoms_3857)}
        logger.info("POI trees ready | count=%d", len(self._poi_geoms))

    # ── POI helpers ───────────────────────────────────────────────────────

    def nearest_poi_index(self, lat: float, lon: float) -> int:
        q = Point(lon, lat)
        nearest = self._poi_tree.nearest(q)
        if nearest is None:
            raise ValueError("No POI found")
        return int(nearest) if isinstance(nearest, (int, np.integer)) \
               else self._poi_geoms.index(nearest)

    def poi_payload(self, idx: int) -> Dict[str, Any]:
        row   = self.pois.iloc[idx]
        props = {k: _json_safe(v) for k, v in row.drop(labels=["geometry"]).to_dict().items()}
        props["poi_latitude"]  = _json_safe(float(row.geometry.y))
        props["poi_longitude"] = _json_safe(float(row.geometry.x))
        return props

    def candidates_within_radius(
        self,
        lat: float,
        lon: float,
        radius_m: Optional[float] = None,
        limit: int = 10,
    ) -> List[Tuple[float, int]]:
        """Return (dist_m, index) pairs within radius, sorted ascending."""
        radius_m = radius_m or self.settings.default_snap_radius_m
        limit    = max(1, min(self.settings.max_poi_candidates, int(limit)))

        q_3857 = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326") \
                    .to_crs("EPSG:3857").iloc[0]
        hits   = self._poi_tree_3857.query(q_3857.buffer(radius_m))
        idxs   = _hits_to_indices(hits, self._poi_id_to_idx)
        if not idxs:
            return []

        dists = [(float(q_3857.distance(self._poi_geoms_3857[i])), int(i))
                 for i in set(idxs)]
        dists.sort()
        return dists[:limit]

    # ── Routing ───────────────────────────────────────────────────────────

    def compute_routes(
        self,
        start_lat: float, start_lon: float,
        end_lat: float,   end_lon: float,
        mode: str,
        k: int,
        snap_to_poi: bool,
        fallback_to_raw: bool,
        poi_candidates: int,
        snap_radius_m: float,
    ) -> RouteResponse:

        def _route(sl, so, el, eo, k_override=None) -> List[RouteResult]:
            return self.engine.route_top3(
                so, sl, eo, el,
                mode=mode,
                k=k_override or k,
            )

        used       = "raw"
        snap_info  = None
        start_lat  = float(start_lat)
        start_lon  = float(start_lon)
        end_lat    = float(end_lat)
        end_lon    = float(end_lon)
        final_slat = start_lat
        final_slon = start_lon
        final_elat = end_lat
        final_elon = end_lon

        if snap_to_poi:
            s_dists = self.candidates_within_radius(start_lat, start_lon, snap_radius_m, poi_candidates)
            e_dists = self.candidates_within_radius(end_lat,   end_lon,   snap_radius_m, poi_candidates)

            if not s_dists or not e_dists:
                logger.info("No POI within %.0fm — using raw coords", snap_radius_m)
                routes    = _route(start_lat, start_lon, end_lat, end_lon)
                snap_info = SnapInfo(
                    rule=f"{snap_radius_m}m", status="skipped",
                    reason="no_poi_within_radius",
                    start_has_poi=bool(s_dists), end_has_poi=bool(e_dists),
                )
            else:
                s_infos = [PoiInfo(idx=i, dist_m=round(d,2), poi=self.poi_payload(i))
                           for d, i in s_dists]
                e_infos = [PoiInfo(idx=i, dist_m=round(d,2), poi=self.poi_payload(i))
                           for d, i in e_dists]

                snap_info = SnapInfo(
                    rule=f"{snap_radius_m}m", status="attempted",
                    start={"input": {"lat": start_lat, "lon": start_lon},
                           "candidates": [i.model_dump() for i in s_infos]},
                    end={"input": {"lat": end_lat, "lon": end_lon},
                         "candidates": [i.model_dump() for i in e_infos]},
                )

                routes  = []
                chosen  = None
                pairs   = 0

                for si in s_infos:
                    if chosen: break
                    for ei in e_infos:
                        pairs += 1
                        if pairs > 4: break
                        sl = float(si.poi["poi_latitude"])
                        so = float(si.poi["poi_longitude"])
                        el = float(ei.poi["poi_latitude"])
                        eo = float(ei.poi["poi_longitude"])
                        if _route(sl, so, el, eo, k_override=1):
                            chosen = {"start": si.model_dump(), "end": ei.model_dump(),
                                      "pairs_tried": pairs}
                            final_slat, final_slon = sl, so
                            final_elat, final_elon = el, eo
                            used = "poi"
                            break
                    if chosen or pairs > 4: break

                if chosen:
                    snap_info.status = "used_poi"
                    snap_info.chosen = chosen
                    routes = _route(final_slat, final_slon, final_elat, final_elon)
                else:
                    snap_info.status   = "failed_poi"
                    snap_info.pairs_tried = pairs
                    if fallback_to_raw:
                        logger.info("All POI pairs failed — falling back to raw")
                        routes = _route(start_lat, start_lon, end_lat, end_lon)
                    else:
                        routes = []
        else:
            routes = _route(start_lat, start_lon, end_lat, end_lon)

        if not routes:
            raise ValueError("No route found between the given points.")

        features = self._build_features(routes, mode, used)

        return RouteResponse(
            type="FeatureCollection",
            features=features,
            routing_endpoints={
                "start": LatLon(lat=final_slat, lon=final_slon),
                "end":   LatLon(lat=final_elat, lon=final_elon),
            },
            poi_snap=snap_info,
            used_routing=used,
        )

    def _build_features(
        self,
        routes: List[RouteResult],
        mode: str,
        used: str,
    ) -> List[RouteFeature]:
        features = []
        for r in routes:
            geom_ll = gpd.GeoSeries([r.geometry_3857], crs="EPSG:3857") \
                         .to_crs("EPSG:4326").iloc[0]
            coords  = list(getattr(geom_ll, "coords", []))

            road_names: List[str] = []
            if hasattr(r, "edge_set"):
                for u, v in sorted(r.edge_set):
                    if self.engine.G.has_edge(u, v):
                        props = self.engine.G.edges[u, v].get("props", {})
                        name  = props.get("name") or props.get("ref") or ""
                        road_names.append(str(name) if name else "")

            instructions = _build_instructions(coords, road_names)
            start_txt    = f"{coords[0][1]:.6f}, {coords[0][0]:.6f}" if coords else "-"
            end_txt      = f"{coords[-1][1]:.6f}, {coords[-1][0]:.6f}" if coords else "-"
            geo          = mapping(geom_ll)

            features.append(RouteFeature(
                properties=RouteProperties(
                    rank=r.rank,
                    distance_m=round(r.distance_m, 2),
                    duration_s=round(r.duration_s, 2),
                    duration_min=math.ceil(r.duration_s / 60.0),
                    mode=mode,
                    summary=RouteSummary(start=start_txt, end=end_txt, used_routing=used),
                    instructions=instructions,
                ),
                geometry=GeoJSONGeometry(type=geo["type"], coordinates=geo["coordinates"]),
            ))
        return features

    # ── Trip store ────────────────────────────────────────────────────────

    def store_trip(
        self,
        trip_id: str,
        req_dict: Dict[str, Any],
        response: RouteResponse,
    ) -> TripRecord:
        """Persist a planned trip keyed by trip_id."""
        planned_coords: List[List[float]] = []
        if response.features:
            planned_coords = list(response.features[0].geometry.coordinates)

        record = TripRecord(
            trip_id=trip_id,
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            request=req_dict,
            planned_coords=planned_coords,
            response=response,
        )
        self._trips[trip_id] = record
        logger.info("trip stored | id=%s routes=%d", trip_id, len(response.features))
        return record

    def get_trip(self, trip_id: str) -> Optional[TripRecord]:
        """Return a stored TripRecord, or None if not found."""
        return self._trips.get(trip_id)

    # ── Graph stats ───────────────────────────────────────────────────────

    def graph_stats(self) -> Dict[str, Any]:
        G    = self.engine.G
        Gu   = G.to_undirected(as_view=True)
        comp = list(nx.connected_components(Gu))
        sizes = sorted([len(c) for c in comp], reverse=True)
        return {
            "ok": True,
            "nodes": int(G.number_of_nodes()),
            "edges": int(G.number_of_edges()),
            "components": int(len(comp)),
            "largest_component_nodes": int(sizes[0]) if sizes else 0,
            "top5_component_sizes": sizes[:5],
        }

    def roads_bounds(self) -> Dict[str, Any]:
        minx, miny, maxx, maxy = self.engine.roads.total_bounds
        pts  = gpd.GeoSeries([Point(minx, miny), Point(maxx, maxy)], crs="EPSG:3857") \
                  .to_crs("EPSG:4326")
        pmin, pmax = pts.iloc[0], pts.iloc[1]
        return dict(crs="EPSG:4326",
                    min_lon=float(pmin.x), min_lat=float(pmin.y),
                    max_lon=float(pmax.x), max_lat=float(pmax.y))
