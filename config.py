from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Data paths ────────────────────────────────────────────────────────
    roads_path: str = "data/test_road.geojson"
    pois_path: str = "data/VN Sample Data.csv"
    restrictions_path: str = ""  # optional — leave empty to disable turn restrictions

    # ── Routing defaults ─────────────────────────────────────────────────
    default_k: int = 3
    max_k: int = 5
    route_timeout_s: float = 8.0
    diversity_threshold: float = 0.20
    default_snap_radius_m: float = 20.0
    max_snap_radius_m: float = 200.0
    max_poi_candidates: int = 10

    # ── API ───────────────────────────────────────────────────────────────
    api_title: str = "VN Routing API"
    api_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    debug: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
