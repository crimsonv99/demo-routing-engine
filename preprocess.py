from __future__ import annotations

import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely import wkb as shapely_wkb

_SHAPELY_2 = int(shapely.__version__.split(".")[0]) >= 2


def node_roads_preserve_attrs(
    roads_gdf: gpd.GeoDataFrame,
    already_split: bool = False,
    min_length_m: float = 0.1,
) -> gpd.GeoDataFrame:
    """
    Prepare road GeoDataFrame for the routing graph.

    already_split=True  (default when loading pre-split file)
        → vectorised cleanup only: explode, filter, drop tiny segments.
          Completes in seconds regardless of feature count.

    already_split=False  (raw OSM download, not yet noded)
        → unary_union to node entire network in GEOS (C),
          then vectorised attribute lookup via STRtree.nearest.
          Much faster than the old per-feature Python loop.
    """

    # ── 1. basic cleanup (always) ─────────────────────────────────────────────
    roads = roads_gdf.copy()
    roads = roads[roads.geometry.notnull()].copy()
    roads = roads.explode(index_parts=False).reset_index(drop=True)
    roads = roads[roads.geometry.geom_type == "LineString"].copy()
    roads = roads[roads.geometry.apply(lambda g: len(g.coords) >= 2)].copy()

    # Drop sub-threshold fragments (vectorised)
    roads = roads[roads.geometry.length >= min_length_m].copy()
    roads = roads.reset_index(drop=True)

    n = len(roads)
    print(f"[preprocess] {n:,} segments after basic cleanup")

    if already_split or n == 0:
        # Data is already noded — nothing more to do
        print(f"[preprocess] already_split=True → skipping intersection pass")
        print(f"[preprocess] done: {n:,} segments ready")
        return roads

    # ── 2. node via GEOS unary_union (raw data path) ──────────────────────────
    print("[preprocess] Noding via GEOS unary_union …")
    original_lines = roads.geometry.tolist()
    noded = unary_union(original_lines)

    if noded is None or noded.is_empty:
        print("[preprocess] warning: unary_union empty — returning cleaned input")
        return roads

    if noded.geom_type == "MultiLineString":
        valid_segs = [g for g in noded.geoms
                      if isinstance(g, LineString) and len(g.coords) >= 2
                      and g.length >= min_length_m]
    elif noded.geom_type == "LineString":
        valid_segs = [noded] if noded.length >= min_length_m else []
    else:
        valid_segs = [g for g in getattr(noded, "geoms", [])
                      if isinstance(g, LineString)
                      and len(g.coords) >= 2
                      and g.length >= min_length_m]

    print(f"[preprocess] {len(valid_segs):,} segments after noding")

    # ── 3. attribute lookup ───────────────────────────────────────────────────
    tree = STRtree(original_lines)
    attr_cols = [c for c in roads.columns if c != "geometry"]
    records = roads[attr_cols].to_dict("records")

    if _SHAPELY_2:
        import numpy as np
        wkb_bytes  = [s.wkb for s in valid_segs]
        segs_arr   = shapely.from_wkb(wkb_bytes)
        mids_arr   = shapely.line_interpolate_point(segs_arr, 0.5, normalized=True)
        parent_idx = tree.nearest(mids_arr).astype(int)
    else:
        id_to_idx  = {id(ln): i for i, ln in enumerate(original_lines)}
        from shapely.geometry import Point
        wkb_bytes  = []
        parent_idx = []
        for seg in valid_segs:
            mid = Point(seg.coords[len(seg.coords) // 2])
            near = tree.nearest(mid)
            parent_idx.append(id_to_idx.get(id(near), 0))
            wkb_bytes.append(seg.wkb)
        parent_idx = parent_idx  # keep as list, handled below

    # ── 4. assemble output ────────────────────────────────────────────────────
    rows = []
    for k, seg in enumerate(valid_segs):
        pidx = int(parent_idx[k])
        row = dict(records[pidx])
        row["geometry"] = seg
        rows.append(row)

    out = gpd.GeoDataFrame(rows, crs=roads_gdf.crs)
    print(f"[preprocess] done: {n:,} input → {len(out):,} output segments")
    return out
