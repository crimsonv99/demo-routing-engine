from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RouteResult:
    engine:       str
    color:        str
    distance_km:  float
    duration_s:   float
    coordinates:  list[list[float]]   # [[lon, lat], ...]  GeoJSON order
    instructions: list[str] = field(default_factory=list)
    osm_way_ids:  list[int] = field(default_factory=list)
    error:        Optional[str] = None
    elapsed_ms:   float = 0.0         # wall-clock engine response time

    @property
    def duration_min(self) -> float:
        return self.duration_s / 60.0

    @property
    def ok(self) -> bool:
        return self.error is None


class BaseEngine:
    NAME:  str = "base"
    COLOR: str = "#000000"

    def load(self, network_path: str = None, restrictions_path: str = None) -> None:
        """Build / warm-up the engine.  May be slow (NetworkX graph build)."""
        pass

    def is_available(self) -> bool:
        raise NotImplementedError

    def route(
        self,
        start_lat: float, start_lon: float,
        end_lat:   float, end_lon:   float,
        mode: str = "car",
        restrictions: list[dict] | None = None,
        buffer_deg: float = 0.00009,
        via_points: list[tuple[float, float]] | None = None,
    ) -> RouteResult:
        raise NotImplementedError
