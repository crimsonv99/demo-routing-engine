from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from shapely.strtree import STRtree

from config import Settings
from services.valhalla_client import get_routes, health_check, RouteResult
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
    Routing via self-hosted Valhalla.
    POI snapping, trip recording, and all API response shapes are unchanged.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._trips: Dict[str, TripRecord] = {}
        self._load_pois()
        logger.info("RoutingService ready | pois=%d | backend=valhalla", len(self.pois))

    # ── Initialisation ────────────────────────────────────────────────────

    def _load_pois(self) -> None:
        logger.info("Loading POIs from %s", self.settings.pois_path)
        pois = load_pois_csv(self.settings.pois_path)
        if str(getattr(pois, "crs", "")) != "EPSG:4326":
            pois = pois.to_crs("EPSG:4326")
        self.pois = pois

        self._poi_geoms      = list(pois.geometry)
        self._poi_tree       = STRtree(self._poi_geoms)

        pois_3857                = pois.to_crs("EPSG:3857")
        self._poi_geoms_3857     = list(pois_3857.geometry)
        self._poi_tree_3857      = STRtree(self._poi_geoms_3857)
        self._poi_id_to_idx      = {id(g): i for i, g in enumerate(self._poi_geoms_3857)}
        logger.info("POI trees ready | count=%d", len(self._poi_geoms))

    # ── POI helpers ───────────────────────────────────────────────────────

    def nearest_poi_index(self, lat: float, lon: float) -> int:
        q       = Point(lon, lat)
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

    async def compute_routes(
        self,
        start_lat: float, start_lon: float,
        end_lat:   float, end_lon:   float,
        mode:      str,
        k:         int,
        snap_to_poi:     bool,
        fallback_to_raw: bool,
        poi_candidates:  int,
        snap_radius_m:   float,
    ) -> RouteResponse:

        async def _route(sl, so, el, eo, k_override=None) -> List[RouteResult]:
            return await get_routes(
                sl, so, el, eo,
                mode=mode,
                alternatives=max(0, (k_override or k) - 1),
                timeout=self.settings.route_timeout_s,
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
                routes    = await _route(start_lat, start_lon, end_lat, end_lon)
                snap_info = SnapInfo(
                    rule=f"{snap_radius_m}m", status="skipped",
                    reason="no_poi_within_radius",
                    start_has_poi=bool(s_dists), end_has_poi=bool(e_dists),
                )
            else:
                s_infos = [PoiInfo(idx=i, dist_m=round(d, 2), poi=self.poi_payload(i))
                           for d, i in s_dists]
                e_infos = [PoiInfo(idx=i, dist_m=round(d, 2), poi=self.poi_payload(i))
                           for d, i in e_dists]

                snap_info = SnapInfo(
                    rule=f"{snap_radius_m}m", status="attempted",
                    start={"input": {"lat": start_lat, "lon": start_lon},
                           "candidates": [i.model_dump() for i in s_infos]},
                    end={"input": {"lat": end_lat, "lon": end_lon},
                         "candidates": [i.model_dump() for i in e_infos]},
                )

                routes = []
                chosen = None
                pairs  = 0

                for si in s_infos:
                    if chosen:
                        break
                    for ei in e_infos:
                        pairs += 1
                        if pairs > 4:
                            break
                        sl = float(si.poi["poi_latitude"])
                        so = float(si.poi["poi_longitude"])
                        el = float(ei.poi["poi_latitude"])
                        eo = float(ei.poi["poi_longitude"])
                        attempt = await _route(sl, so, el, eo, k_override=1)
                        if attempt:
                            chosen = {"start": si.model_dump(), "end": ei.model_dump(),
                                      "pairs_tried": pairs}
                            final_slat, final_slon = sl, so
                            final_elat, final_elon = el, eo
                            used = "poi"
                            break
                    if chosen or pairs > 4:
                        break

                if chosen:
                    snap_info.status = "used_poi"
                    snap_info.chosen = chosen
                    routes = await _route(final_slat, final_slon, final_elat, final_elon)
                else:
                    snap_info.status      = "failed_poi"
                    snap_info.pairs_tried = pairs
                    if fallback_to_raw:
                        logger.info("All POI pairs failed — falling back to raw")
                        routes = await _route(start_lat, start_lon, end_lat, end_lon)
                    else:
                        routes = []
        else:
            routes = await _route(start_lat, start_lon, end_lat, end_lon)

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
        for rank, r in enumerate(routes, start=1):
            coords = r.geometry.get("coordinates", [])
            start_txt = f"{coords[0][1]:.6f}, {coords[0][0]:.6f}"  if coords else "-"
            end_txt   = f"{coords[-1][1]:.6f}, {coords[-1][0]:.6f}" if coords else "-"

            features.append(RouteFeature(
                properties=RouteProperties(
                    rank=rank,
                    distance_m=round(r.distance_km * 1000, 2),
                    duration_s=round(r.duration_s, 2),
                    duration_min=math.ceil(r.duration_s / 60.0),
                    mode=mode,
                    summary=RouteSummary(start=start_txt, end=end_txt, used_routing=used),
                    instructions=r.instructions,
                ),
                geometry=GeoJSONGeometry(
                    type=r.geometry["type"],
                    coordinates=r.geometry["coordinates"],
                ),
            ))
        return features

    # ── Trip store ────────────────────────────────────────────────────────

    def store_trip(
        self,
        trip_id:  str,
        req_dict: Dict[str, Any],
        response: RouteResponse,
    ) -> TripRecord:
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
        return self._trips.get(trip_id)

    # ── Health / stats ────────────────────────────────────────────────────

    async def graph_stats(self) -> Dict[str, Any]:
        """Return Valhalla status instead of NetworkX graph stats."""
        ok = await health_check()
        return {
            "ok": ok,
            "backend": "valhalla",
            "valhalla_url": "http://localhost:8002",
            "status": "reachable" if ok else "unreachable",
        }
