from __future__ import annotations
import sys, os, time, traceback

# routing_engine.py lives in the parent (prod/) directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pyproj import Transformer
import geopandas as gpd

from benchmark_engines.base import BaseEngine, RouteResult

_T = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

# Travel mode → routing_engine mode string
_MODE = {
    "car":        "car",
    "motorcycle": "motorcycle",
    "bike":       "car",   # fallback — NetworkX has no bike profile
    "walk":       "car",   # fallback
}


class NetworkXEngine(BaseEngine):
    NAME  = "NetworkX"
    COLOR = "#2563EB"   # blue

    def __init__(self):
        self._engine = None
        self._ready  = False

    # ------------------------------------------------------------------
    def load(self, network_path: str = None, restrictions_path: str = None) -> None:
        if not network_path:
            raise ValueError("NetworkX engine requires a road network file.")

        from preprocess import node_roads_preserve_attrs
        from routing_engine import RouteEngine

        already_split = "split" in os.path.basename(network_path).lower()

        print(f"[NetworkX] loading {network_path}  already_split={already_split}")
        t0 = time.time()
        roads = gpd.read_file(network_path)
        roads_3857 = node_roads_preserve_attrs(roads, already_split=already_split)
        print(f"[NetworkX] preprocess done  {time.time()-t0:.1f}s")

        print("[NetworkX] building graph …")
        t1 = time.time()
        self._engine = RouteEngine(roads_3857, restrictions_path=restrictions_path)
        print(f"[NetworkX] graph ready  {time.time()-t1:.1f}s")
        self._ready = True

    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        return self._ready

    # ------------------------------------------------------------------
    def route(self, start_lat, start_lon, end_lat, end_lon, mode="car",
              restrictions: list[dict] | None = None) -> RouteResult:
        if not self._ready:
            return RouteResult(self.NAME, self.COLOR, 0, 0, [], error="Engine not loaded")

        try:
            eng_mode = _MODE.get(mode, "car")
            results  = self._engine.route_top3(start_lon, start_lat, end_lon, end_lat,
                                                mode=eng_mode)
            if not results:
                return RouteResult(self.NAME, self.COLOR, 0, 0, [],
                                   error="No route found")

            best = results[0]

            # Convert EPSG:3857 LineString → [[lon, lat], ...]
            xs, ys  = zip(*best.geometry_3857.coords)
            lons, lats = _T.transform(xs, ys)
            coords  = [[lon, lat] for lon, lat in zip(lons, lats)]

            dist_km = best.distance_m / 1000.0

            return RouteResult(
                engine      = self.NAME,
                color       = self.COLOR,
                distance_km = dist_km,
                duration_s  = best.duration_s,
                coordinates = coords,
            )

        except Exception as e:
            traceback.print_exc()
            return RouteResult(self.NAME, self.COLOR, 0, 0, [], error=str(e))
