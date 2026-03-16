from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ── Request ───────────────────────────────────────────────────────────────────

class RouteRequest(BaseModel):
    start_lat: float = Field(..., ge=-90, le=90, description="Start latitude")
    start_lon: float = Field(..., ge=-180, le=180, description="Start longitude")
    end_lat: float = Field(..., ge=-90, le=90, description="End latitude")
    end_lon: float = Field(..., ge=-180, le=180, description="End longitude")
    mode: str = Field("car", description="Travel mode: 'car' or 'motorcycle'")
    k: int = Field(3, ge=1, le=5, description="Number of routes to return")
    snap_to_poi: bool = Field(True, description="Snap endpoints to nearest POI")
    fallback_to_raw: bool = Field(True, description="Fall back to raw coords if POI snap fails")
    poi_candidates: int = Field(3, ge=1, le=10, description="Max POI candidates to try")
    snap_radius_m: float = Field(20.0, ge=1.0, le=200.0, description="POI snap radius in metres")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("car", "motorcycle"):
            raise ValueError("mode must be 'car' or 'motorcycle'")
        return v


# ── Shared sub-models ─────────────────────────────────────────────────────────

class LatLon(BaseModel):
    lat: float
    lon: float


class PoiInfo(BaseModel):
    idx: int
    dist_m: float
    poi: Dict[str, Any]


class SnapInfo(BaseModel):
    rule: str
    status: str
    reason: Optional[str] = None
    start_has_poi: Optional[bool] = None
    end_has_poi: Optional[bool] = None
    start: Optional[Dict[str, Any]] = None
    end: Optional[Dict[str, Any]] = None
    chosen: Optional[Dict[str, Any]] = None
    pairs_tried: Optional[int] = None


class RouteSummary(BaseModel):
    start: str
    end: str
    used_routing: str


class RouteProperties(BaseModel):
    rank: int
    distance_m: float
    duration_s: float
    duration_min: float
    mode: str
    summary: RouteSummary
    instructions: List[str]


class GeoJSONGeometry(BaseModel):
    type: str
    coordinates: List[Any]


class RouteFeature(BaseModel):
    type: str = "Feature"
    properties: RouteProperties
    geometry: GeoJSONGeometry


class RouteResponse(BaseModel):
    type: str = "FeatureCollection"
    features: List[RouteFeature]
    routing_endpoints: Dict[str, LatLon]
    poi_snap: Optional[SnapInfo] = None
    used_routing: str


# ── POI ───────────────────────────────────────────────────────────────────────

class NearestPoiResponse(BaseModel):
    query: LatLon
    nearest_poi: Dict[str, Any]


# ── Health / Stats ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    ok: bool
    roads_segments: int
    pois: int
    api_version: str


class GraphStatsResponse(BaseModel):
    ok: bool
    nodes: int
    edges: int
    components: int
    largest_component_nodes: int
    top5_component_sizes: List[int]


class BoundsResponse(BaseModel):
    crs: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


# ── Standard error wrapper ────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    code: str
    message: str
    detail: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
