"""
Split lines at intersections and assign segment_id
────────────────────────────────────────────────────────────────
Architecture
  Step 1  Load + deduplicate input                 (Python)
  Step 2  unary_union → full noded network          (GEOS / C)
  Step 3  Vectorised midpoint + nearest lookup      (Shapely 2.x C / numpy)
          — or coordinate-midpoint fallback          (Shapely 1.x)
  Step 4  Write to sink                             (Python, unavoidable)

Why Step 3 was slow before
  Calling seg.interpolate() and orig_tree.nearest() 6.5 M times
  from Python adds ~1 ms Python-call overhead per feature = 1.8 h.
  Shapely 2.x exposes vectorised versions of both that run entirely
  inside C on a numpy array — 6.5 M lookups finish in seconds.
"""

import shapely
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union
from shapely import wkb as shapely_wkb
from shapely.strtree import STRtree
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFeatureSink,
    QgsProcessing,
    QgsProcessingException,
    QgsFeature,
    QgsGeometry,
    QgsWkbTypes,
    QgsField,
    QgsFeatureSink,
)
from PyQt5.QtCore import QVariant

_SHAPELY_2 = int(shapely.__version__.split(".")[0]) >= 2


class SplitLinesWithSegmentID(QgsProcessingAlgorithm):
    INPUT  = "INPUT"
    OUTPUT = "OUTPUT"

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INPUT, "Input line layer", [QgsProcessing.TypeVectorLine]))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, "Output split lines with segment_id"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer")

        osm_idx = (source.fields().indexOf("osm_id")
                   if "osm_id" in source.fields().names() else -1)

        # ── Step 1: load + deduplicate ─────────────────────────────────────────
        feedback.pushInfo("Step 1/4  Loading features…")
        original_lines: list = []
        attributes:     list = []
        seen: set = set()

        for feat in source.getFeatures():
            if feedback.isCanceled():
                return {self.OUTPUT: None}
            geom = feat.geometry()
            if not geom or geom.isEmpty():
                continue
            try:
                shp = shapely_wkb.loads(bytes(geom.asWkb()))
            except Exception:
                continue

            parts = []
            if isinstance(shp, MultiLineString):
                parts = [p for p in shp.geoms
                         if isinstance(p, LineString) and len(p.coords) >= 2]
            elif isinstance(shp, LineString) and len(shp.coords) >= 2:
                parts = [shp]

            for part in parts:
                key = part.wkb
                if key not in seen:
                    seen.add(key)
                    original_lines.append(part)
                    attributes.append(feat.attributes())

        del seen
        if not original_lines:
            raise QgsProcessingException("No valid line geometries found.")

        feedback.pushInfo(f"  {len(original_lines):,} unique segments loaded.")
        feedback.setProgress(10)

        # ── Step 2: node the whole network in GEOS (C, not Python) ────────────
        feedback.pushInfo("Step 2/4  Noding via GEOS unary_union…")
        noded = unary_union(original_lines)

        if noded is None or noded.is_empty:
            raise QgsProcessingException("unary_union returned empty geometry.")

        if noded.geom_type == "MultiLineString":
            all_segs = list(noded.geoms)
        elif noded.geom_type == "LineString":
            all_segs = [noded]
        else:
            all_segs = [g for g in getattr(noded, "geoms", [])
                        if isinstance(g, LineString)]

        feedback.pushInfo(f"  {len(all_segs):,} segments after noding.")
        feedback.setProgress(55)

        # ── Step 3: vectorised attribute lookup ───────────────────────────────
        # Filter to valid LineStrings first
        valid_segs = [s for s in all_segs
                      if isinstance(s, LineString) and len(s.coords) >= 2]
        del all_segs
        n_valid = len(valid_segs)

        feedback.pushInfo(f"Step 3/4  Attribute lookup for {n_valid:,} segments…")
        orig_tree = STRtree(original_lines)

        if _SHAPELY_2:
            # ── Shapely 2.x: everything in C / numpy ──────────────────────────
            import shapely as shplib
            import numpy as np

            # Pre-extract WKB bytes once (reused for QgsGeometry later)
            feedback.pushInfo("  Extracting WKB…")
            wkb_list = [s.wkb for s in valid_segs]

            # Build geometry array for vectorised ops
            feedback.pushInfo("  Computing midpoints (vectorised)…")
            segs_arr = shplib.from_wkb(wkb_list)
            mids_arr = shplib.line_interpolate_point(segs_arr, 0.5, normalized=True)

            # All nearest lookups in one C call → numpy int array
            feedback.pushInfo("  Running nearest lookups (vectorised)…")
            parent_idxs = orig_tree.nearest(mids_arr)   # shape (n_valid,)

        else:
            # ── Shapely 1.x fallback: avoid .interpolate(), use coord midpoint ─
            # seg.coords[n//2] is a pure Python list access — no Shapely overhead.
            id_to_idx = {id(ln): i for i, ln in enumerate(original_lines)}

            feedback.pushInfo("  Running nearest lookups (Shapely 1.x)…")
            parent_idxs = []
            wkb_list    = []
            step = max(1, n_valid // 200)

            for k, seg in enumerate(valid_segs):
                if k % step == 0:
                    if feedback.isCanceled():
                        return {self.OUTPUT: None}
                    feedback.setProgress(55 + int(k / n_valid * 10))

                # Midpoint via coordinate list — no Shapely call
                mid_coord = seg.coords[len(seg.coords) // 2]
                mid_pt    = Point(mid_coord)

                nearest_geom = orig_tree.nearest(mid_pt)
                parent_idxs.append(id_to_idx.get(id(nearest_geom), 0))
                wkb_list.append(seg.wkb)

        feedback.setProgress(65)

        # ── Step 4: write output ───────────────────────────────────────────────
        feedback.pushInfo("Step 4/4  Writing output…")

        fields = source.fields()
        fields.append(QgsField("segment_id", QVariant.String))
        sink, dest_id = self.parameterAsSink(
            parameters, self.OUTPUT, context, fields,
            QgsWkbTypes.LineString, source.sourceCrs()
        )

        seg_counter: dict = {}
        total_written = 0
        step = max(1, n_valid // 1000)

        for k in range(n_valid):
            if k % step == 0:
                if feedback.isCanceled():
                    break
                feedback.setProgress(65 + int(k / n_valid * 35))

            parent_idx = int(parent_idxs[k])
            attr       = attributes[parent_idx]
            base_id    = (str(attr[osm_idx]) if osm_idx != -1
                          else f"line{parent_idx}")
            seg_counter[base_id] = seg_counter.get(base_id, 0) + 1

            out_feat = QgsFeature()
            out_feat.setFields(fields, True)
            _geom = QgsGeometry()
            _geom.fromWkb(wkb_list[k])        # reuse pre-extracted WKB
            out_feat.setGeometry(_geom)
            out_feat.setAttributes(attr + [f"{base_id}_{seg_counter[base_id]}"])
            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)
            total_written += 1

        feedback.setProgress(100)
        feedback.pushInfo(f"Done — {total_written:,} segments written.")
        return {self.OUTPUT: dest_id}

    def name(self):           return "split_lines_with_segment_id"
    def displayName(self):    return "Split lines at intersections and assign segment_id"
    def group(self):          return "Custom Scripts"
    def groupId(self):        return "customscripts"
    def createInstance(self): return SplitLinesWithSegmentID()
