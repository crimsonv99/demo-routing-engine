"""
Split road lines at intersections and assign segment_id
Standalone — no QGIS required.

Install
    pip install geopandas shapely numpy

Usage
    python split_lines_standalone.py road.geojson          # → road_split.geojson
    python split_lines_standalone.py road.gpkg  out.gpkg
    python split_lines_standalone.py road.shp   out.shp  --layer roads
"""

from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree

_SHAPELY_2 = int(shapely.__version__.split(".")[0]) >= 2

# ── logging ───────────────────────────────────────────────────────────────────

def _log(msg: str, indent: int = 0) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}]{'  ' * indent}{msg}", flush=True)


def _progress(done: int, total: int, width: int = 30) -> str:
    pct  = done / total if total else 0
    fill = int(pct * width)
    bar  = "█" * fill + "░" * (width - fill)
    return f"  [{bar}] {pct:5.1%}  {done:,}/{total:,}"


# ── step 1: load ──────────────────────────────────────────────────────────────

def load(path: str, layer: str | None) -> tuple[gpd.GeoDataFrame, list[LineString], list[int]]:
    """
    Read file → explode multi-geometries → deduplicate.
    Returns (gdf, line_list, gdf_index_per_line).
    gdf_index_per_line[i]  is the row index in gdf that line_list[i] came from.
    """
    _log(f"Loading  {path}" + (f"  (layer={layer})" if layer else ""))
    kw = {"layer": layer} if layer else {}
    gdf = gpd.read_file(path, **kw)
    _log(f"  {len(gdf):,} features read,  CRS = {gdf.crs}", indent=1)

    # Normalise to LineString
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.geom_type == "LineString"].copy()
    gdf = gdf[gdf.geometry.apply(lambda g: len(g.coords) >= 2)].copy()
    gdf = gdf.reset_index(drop=True)

    # Deduplicate by WKB
    wkb = gdf.geometry.apply(lambda g: g.wkb)
    mask = ~wkb.duplicated()
    gdf  = gdf[mask].reset_index(drop=True)

    lines   = gdf.geometry.tolist()
    row_idx = list(range(len(gdf)))           # identity: line i → gdf row i

    _log(f"  {len(lines):,} unique LineStrings after dedup", indent=1)
    return gdf, lines, row_idx


# ── step 2: node ──────────────────────────────────────────────────────────────

def node_network(lines: list[LineString]) -> list[LineString]:
    """
    Run GEOS unary_union to insert nodes at every intersection.
    Returns list of noded LineStrings.
    """
    _log("Noding   GEOS unary_union …  (may take a few minutes for large networks)")
    t0 = time.time()
    noded = unary_union(lines)
    elapsed = time.time() - t0
    _log(f"  unary_union finished in {elapsed:.1f}s", indent=1)

    if noded is None or noded.is_empty:
        raise RuntimeError("unary_union returned an empty geometry — check input.")

    if noded.geom_type == "LineString":
        segs = [noded]
    elif noded.geom_type == "MultiLineString":
        segs = list(noded.geoms)
    else:
        # GeometryCollection edge case
        segs = [g for g in getattr(noded, "geoms", [])
                if isinstance(g, LineString)]

    valid = [s for s in segs if isinstance(s, LineString) and len(s.coords) >= 2]
    _log(f"  {len(valid):,} segments after noding", indent=1)
    return valid


# ── step 3: attribute lookup ───────────────────────────────────────────────────

def find_parents(
    valid_segs: list[LineString],
    original_lines: list[LineString],
) -> np.ndarray:
    """
    For each noded segment return the index of the original line it belongs to.
    Uses vectorised C operations in Shapely 2.x; coord-midpoint fallback for 1.x.
    """
    _log("Lookup   finding parent line for each segment …")
    tree = STRtree(original_lines)

    if _SHAPELY_2:
        _log("  Shapely 2.x — vectorised midpoints + nearest (all in C)", indent=1)
        wkb_bytes  = [s.wkb for s in valid_segs]
        segs_arr   = shapely.from_wkb(wkb_bytes)
        mids_arr   = shapely.line_interpolate_point(segs_arr, 0.5, normalized=True)
        parent_idx = tree.nearest(mids_arr).astype(int)      # numpy array
        return parent_idx, wkb_bytes

    else:
        _log("  Shapely 1.x — coordinate-midpoint + nearest loop", indent=1)
        id_to_idx = {id(ln): i for i, ln in enumerate(original_lines)}
        parent_idx: list[int] = []
        wkb_bytes:  list[bytes] = []
        n = len(valid_segs)

        for k, seg in enumerate(valid_segs):
            mid_pt = Point(seg.coords[len(seg.coords) // 2])
            near   = tree.nearest(mid_pt)
            parent_idx.append(id_to_idx.get(id(near), 0))
            wkb_bytes.append(seg.wkb)

            if k % 200_000 == 0:
                print(_progress(k, n), end="\r", flush=True)

        print(_progress(n, n))
        return np.array(parent_idx, dtype=int), wkb_bytes


# ── step 4: build output ───────────────────────────────────────────────────────

def build_output(
    valid_segs:     list[LineString],
    wkb_bytes:      list[bytes],
    parent_idx:     np.ndarray,
    gdf:            gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Assemble output GeoDataFrame: original attributes + segment_id + geometry.
    """
    _log(f"Building output for {len(valid_segs):,} segments …")

    osm_col   = "osm_id" if "osm_id" in gdf.columns else None
    attr_cols = [c for c in gdf.columns if c != "geometry"]
    records   = gdf[attr_cols].to_dict("records")   # list of dicts (fast row access)

    rows:        list[dict] = []
    seg_counter: dict[str, int] = {}
    n = len(valid_segs)

    for k in range(n):
        pidx   = int(parent_idx[k])
        row    = dict(records[pidx])             # copy parent attributes

        base   = str(row[osm_col]) if osm_col else f"line{pidx}"
        seg_counter[base] = seg_counter.get(base, 0) + 1
        row["segment_id"] = f"{base}_{seg_counter[base]}"
        row["geometry"]   = valid_segs[k]
        rows.append(row)

        if k % 200_000 == 0:
            print(_progress(k, n), end="\r", flush=True)

    print(_progress(n, n))

    out = gpd.GeoDataFrame(rows, crs=gdf.crs)
    _log(f"  {len(out):,} features assembled", indent=1)
    return out


# ── step 5: save ──────────────────────────────────────────────────────────────

def save(out_gdf: gpd.GeoDataFrame, path: str) -> None:
    _log(f"Saving   {path} …")
    ext = Path(path).suffix.lower()
    if ext in (".gpkg",):
        out_gdf.to_file(path, driver="GPKG")
    elif ext in (".geojson", ".json"):
        out_gdf.to_file(path, driver="GeoJSON")
    elif ext in (".shp",):
        out_gdf.to_file(path)
    else:
        out_gdf.to_file(path)   # let Fiona detect
    _log(f"  Done — {len(out_gdf):,} segments written.", indent=1)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split road lines at intersections and assign segment_id"
    )
    parser.add_argument("input",         help="Input file  (.geojson / .gpkg / .shp)")
    parser.add_argument("output", nargs="?",
                        help="Output file (default: <input>_split.<ext>)")
    parser.add_argument("--layer", default=None,
                        help="Layer name (for .gpkg with multiple layers)")
    args = parser.parse_args()

    # Default output path
    if not args.output:
        p = Path(args.input)
        args.output = str(p.with_stem(p.stem + "_split"))

    t_start = time.time()
    _log(f"=== split_lines_standalone  (Shapely {shapely.__version__}) ===")
    _log(f"Input  : {args.input}")
    _log(f"Output : {args.output}")
    print()

    gdf, original_lines, _ = load(args.input, args.layer)
    print()

    valid_segs = node_network(original_lines)
    print()

    parent_idx, wkb_bytes = find_parents(valid_segs, original_lines)
    print()

    out_gdf = build_output(valid_segs, wkb_bytes, parent_idx, gdf)
    print()

    save(out_gdf, args.output)
    print()
    _log(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
