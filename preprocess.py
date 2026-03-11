from __future__ import annotations
from typing import List, Dict, Any
import geopandas as gpd
from shapely.geometry import LineString, Point, MultiPoint
from shapely.ops import split


def _collect_split_points(line: LineString, candidates) -> List[Point]:
    pts: List[Point] = []
    for geom in candidates:
        if geom is None or geom.is_empty:
            continue
        inter = line.intersection(geom)
        if inter.is_empty:
            continue
        gtype = inter.geom_type
        if gtype == "Point":
            pts.append(inter)
        elif gtype == "MultiPoint":
            pts.extend(list(inter.geoms))
        # LineString overlaps are ignored (rare in clean road data)
    return pts


def _dedup_points(points: List[Point], tol: float = 0.01) -> MultiPoint:
    """Bucket-deduplicate points at *tol* metre resolution."""
    seen = set()
    uniq = []
    for p in points:
        key = (round(p.x / tol) * tol, round(p.y / tol) * tol)
        if key not in seen:
            seen.add(key)
            uniq.append(Point(key[0], key[1]))
    return MultiPoint(uniq)


def node_roads_preserve_attrs(roads_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Split each road feature at intersections with all other roads, preserving
    all non-geometry attributes on every resulting sub-segment.

    Assumes roads_gdf is already in a METRIC CRS (EPSG:3857).

    Improvements vs original:
    - Progress logging every 500 rows so long preprocessing isn't silent.
    - Spatial index is built *after* explode/reset so iloc lookups are safe.
    - Tiny segments (< 0.1 m) are dropped to avoid near-zero-length graph edges.
    """
    roads = roads_gdf.copy()
    roads = roads[roads.geometry.notnull()].copy()
    roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    roads = roads.explode(index_parts=False).reset_index(drop=True)

    n_total = len(roads)
    print(f"[preprocess] node_roads: {n_total} segments to process")

    # Build sindex after explode/reset so positional indices are contiguous.
    sindex = roads.sindex

    out_rows: List[Dict[str, Any]] = []

    for i in range(n_total):
        if i > 0 and i % 500 == 0:
            print(f"[preprocess] {i}/{n_total} segments processed, {len(out_rows)} output rows so far")

        row = roads.iloc[i]
        line: LineString = row.geometry
        if line is None or line.is_empty or line.geom_type != "LineString":
            continue

        # Candidate neighbours by bounding-box overlap
        cand_idx = list(sindex.intersection(line.bounds))
        cand_geoms = [roads.iloc[j].geometry for j in cand_idx if j != i]

        pts = _collect_split_points(line, cand_geoms)

        # Drop split points that coincide with the line's own endpoints
        start = Point(line.coords[0])
        end = Point(line.coords[-1])
        filtered = [
            p for p in pts
            if p.distance(start) >= 0.05 and p.distance(end) >= 0.05
        ]

        base_attrs = row.drop(labels=["geometry"]).to_dict()

        if not filtered:
            out = dict(base_attrs)
            out["geometry"] = line
            out_rows.append(out)
            continue

        mp = _dedup_points(filtered, tol=0.05)  # 5 cm bucketing
        try:
            pieces = split(line, mp)
        except Exception as exc:
            print(f"[preprocess] split failed for segment {i}: {exc} — keeping original")
            out = dict(base_attrs)
            out["geometry"] = line
            out_rows.append(out)
            continue

        for seg in pieces.geoms:
            if seg.length < 0.1:   # drop sub-10cm fragments
                continue
            out = dict(base_attrs)
            out["geometry"] = seg
            out_rows.append(out)

    print(f"[preprocess] done: {n_total} input segments → {len(out_rows)} output segments")
    return gpd.GeoDataFrame(out_rows, crs=roads_gdf.crs)
