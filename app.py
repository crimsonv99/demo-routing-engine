from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from pydantic import BaseModel
import geopandas as gpd
from shapely.geometry import mapping, Point
from shapely.strtree import STRtree
import numpy as np
import math
import networkx as nx

from preprocess import node_roads_preserve_attrs
from routing_engine import RouteEngine
from poi_loader import load_pois_csv

import os
import traceback

print(f"[route_tool] Loaded app.py from: {__file__}")
print(f"[route_tool] CWD: {os.getcwd()}")

ROADS_PATH = "data/test_road.geojson"
POIS_PATH = "data/VN Sample Data.csv"

app = FastAPI(title="Top-3 Routing Tool (Road GeoJSON + POI CSV)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_safe(v):
    """Convert values to JSON-safe types (replace NaN/Inf with None)."""
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
    """Compass bearing (degrees, 0=North) from point 1 → point 2 in projected coords."""
    dx, dy = x2 - x1, y2 - y1
    angle = math.degrees(math.atan2(dx, dy)) % 360
    return angle


def _bearing_to_cardinal(deg: float) -> str:
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = round(deg / 45) % 8
    return dirs[idx]


def _turn_instruction(prev_bearing: float, next_bearing: float) -> str:
    """Return a human-readable turn instruction based on bearing change."""
    diff = (next_bearing - prev_bearing + 360) % 360
    if diff < 20 or diff > 340:
        return "Continue straight"
    elif diff < 80:
        return "Turn slight right"
    elif diff < 150:
        return "Turn right"
    elif diff < 200:
        return "Make a U-turn"
    elif diff < 260:
        return "Turn left"
    else:
        return "Turn slight left"


def _build_instructions(coords_ll: list, road_names: list) -> list:
    """
    Generate turn-by-turn instructions from a coordinate list and road-name list.
    coords_ll: list of (lon, lat) tuples in EPSG:4326
    road_names: list of road name strings, one per segment between coords (len = len(coords)-1)
    """
    if len(coords_ll) < 2:
        return []

    instructions = []
    start = coords_ll[0]
    first_name = road_names[0] if road_names else ""
    first_bearing = _bearing_deg(start[0], start[1], coords_ll[1][0], coords_ll[1][1])
    dir_str = _bearing_to_cardinal(first_bearing)
    name_str = f" on {first_name}" if first_name else ""
    instructions.append(f"Head {dir_str}{name_str}")

    for i in range(1, len(coords_ll) - 1):
        prev = coords_ll[i - 1]
        curr = coords_ll[i]
        nxt = coords_ll[i + 1]

        b_in = _bearing_deg(prev[0], prev[1], curr[0], curr[1])
        b_out = _bearing_deg(curr[0], curr[1], nxt[0], nxt[1])

        turn = _turn_instruction(b_in, b_out)
        if turn == "Continue straight":
            continue  # suppress noise

        road_after = road_names[i] if i < len(road_names) else ""
        onto = f" onto {road_after}" if road_after else ""
        lat_str = f"{curr[1]:.5f}"
        lon_str = f"{curr[0]:.5f}"
        instructions.append(f"{turn}{onto} ({lat_str}, {lon_str})")

    end = coords_ll[-1]
    instructions.append(f"Arrive at destination ({end[1]:.5f}, {end[0]:.5f})")
    return instructions


# ---------------------------------------------------------------------------
# Map UI
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/map", response_class=HTMLResponse)
def map_ui():
    return HTMLResponse(
        """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Route Tool Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body { height: 100%; margin: 0; }
    #wrap { display: flex; height: 100%; }
    #map { flex: 2; }
    #panel {
      flex: 1;
      padding: 12px;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial;
      overflow: auto;
      border-left: 1px solid #ddd;
      background: #fafafa;
      min-width: 280px;
      max-width: 380px;
    }
    .hint { color: #666; font-size: 13px; margin-bottom: 10px; }
    .section { margin-bottom: 16px; }
    .row { margin-bottom: 8px; }
    .label { font-size: 12px; color: #555; margin-bottom: 4px; }
    .value { font-weight: 600; word-break: break-word; }
    button {
      padding: 8px 10px;
      border: 1px solid #ccc;
      background: white;
      cursor: pointer;
      border-radius: 6px;
      margin-right: 8px;
      margin-bottom: 8px;
    }
    button.active { border-color: #333; font-weight: 700; }
    pre {
      white-space: pre-wrap;
      background: white;
      border: 1px solid #ddd;
      padding: 8px;
      border-radius: 6px;
      font-size: 12px;
    }
    .route-card {
      background: white;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 10px;
      margin-bottom: 10px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .route-title { font-weight: 700; margin-bottom: 6px; }
    .route-badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 10px;
      font-size: 11px;
      font-weight: 600;
      margin-left: 6px;
      vertical-align: middle;
    }
    .badge-poi { background: #d1fae5; color: #065f46; }
    .badge-raw { background: #fee2e2; color: #991b1b; }

    /* Error banner */
    #error-banner {
      display: none;
      background: #fef2f2;
      border: 1px solid #fca5a5;
      border-radius: 8px;
      padding: 10px 12px;
      margin-bottom: 12px;
      color: #991b1b;
      font-size: 13px;
    }
    #error-banner .err-title { font-weight: 700; margin-bottom: 4px; }
    #error-banner .err-hint { color: #b91c1c; }

    /* Spinner */
    .spinner {
      display: inline-block;
      width: 14px; height: 14px;
      border: 2px solid #ccc;
      border-top-color: #333;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      vertical-align: middle;
      margin-right: 6px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
<div id="wrap">
  <div id="map"></div>
  <div id="panel">

    <div id="error-banner">
      <div class="err-title" id="err-title">Route not found</div>
      <div class="err-hint" id="err-hint"></div>
    </div>

    <div class="section">
      <div class="hint">Click map to set Start and End, then click Route.</div>
      <button id="mode-start" class="active">📍 Set Start</button>
      <button id="mode-end">🏁 Set End</button>
      <button id="btn-route">🗺 Route</button>
      <button id="btn-clear">✕ Clear</button>
    </div>

    <div class="section">
      <div class="row"><div class="label">Start</div><div id="start-out" class="value">Not set</div></div>
      <div class="row"><div class="label">End</div><div id="end-out" class="value">Not set</div></div>
      <div class="row"><div class="label">Routing used</div><div id="used-out" class="value">-</div></div>
      <div class="row"><div class="label">POI coverage boundary</div><div id="boundary-out" class="value">Loading...</div></div>
    </div>

    <div class="section">
      <div class="row"><label><input type="checkbox" id="snap-to-poi" checked> snap_to_poi</label></div>
      <div class="row"><label><input type="checkbox" id="fallback-to-raw" checked> fallback_to_raw</label></div>
      <div class="row">
        <div class="label">Mode</div>
        <select id="mode-select">
          <option value="car">🚗 car</option>
          <option value="motorcycle">🏍 motorcycle</option>
        </select>
      </div>
      <div class="row"><div class="label">k (max routes)</div><input id="k-input" type="number" value="3" min="1" max="3" style="width:50px"/></div>
      <div class="row"><div class="label">poi_candidates</div><input id="poi-candidates" type="number" value="3" min="1" max="10" style="width:50px"/></div>
    </div>

    <div class="section">
      <div class="label">Nearest POI — Start</div>
      <pre id="poi-start-out">Start not set.</pre>
    </div>

    <div class="section">
      <div class="label">Nearest POI — End</div>
      <pre id="poi-end-out">End not set.</pre>
    </div>

    <div class="section">
      <div class="label">Routes</div>
      <div id="routes-out">No routes yet.</div>
    </div>
  </div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
  const map = L.map('map').setView([10.78, 106.65], 14);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map);

  let clickMode = 'start';
  let startPoint = null;
  let endPoint = null;
  let startMarker = null;
  let endMarker = null;
  let routeLayers = [];
  let routeMeta = [];
  let selectedRouteIndex = -1;
  let boundaryLayer = null;
  let boundaryBounds = null;
  let lastRoutedMode = null;  // track mode used for current displayed routes

  const startOut = document.getElementById('start-out');
  const endOut = document.getElementById('end-out');
  const usedOut = document.getElementById('used-out');
  const boundaryOut = document.getElementById('boundary-out');
  const poiStartOut = document.getElementById('poi-start-out');
  const poiEndOut = document.getElementById('poi-end-out');
  const routesOut = document.getElementById('routes-out');
  const modeStartBtn = document.getElementById('mode-start');
  const modeEndBtn = document.getElementById('mode-end');
  const errorBanner = document.getElementById('error-banner');
  const errTitle = document.getElementById('err-title');
  const errHint = document.getElementById('err-hint');

  // ── Error banner helpers ────────────────────────────────────────────────
  function showError(title, hint) {
    errTitle.textContent = title;
    errHint.textContent = hint || '';
    errorBanner.style.display = 'block';
  }
  function hideError() {
    errorBanner.style.display = 'none';
  }

  // ── Boundary ────────────────────────────────────────────────────────────
  async function loadBoundary() {
    try {
      const res = await fetch('/roads_bounds');
      if (!res.ok) { boundaryOut.textContent = 'Failed to load'; return; }
      const b = await res.json();
      boundaryBounds = b;
      boundaryOut.textContent =
        `${b.min_lat.toFixed(5)}, ${b.min_lon.toFixed(5)} → ${b.max_lat.toFixed(5)}, ${b.max_lon.toFixed(5)}`;
      const bounds = [[b.min_lat, b.min_lon], [b.max_lat, b.max_lon]];
      if (boundaryLayer) map.removeLayer(boundaryLayer);
      boundaryLayer = L.rectangle(bounds, { color: '#6b7280', weight: 2, opacity: 0.9, fillOpacity: 0.04 }).addTo(map);
      map.fitBounds(bounds, { padding: [20, 20] });
    } catch (err) {
      boundaryOut.textContent = 'Failed to load';
    }
  }

  function pointInsideBoundary(p) {
    if (!boundaryBounds || !p) return true;
    return (
      p.lat >= boundaryBounds.min_lat && p.lat <= boundaryBounds.max_lat &&
      p.lon >= boundaryBounds.min_lon && p.lon <= boundaryBounds.max_lon
    );
  }

  // ── Mode buttons ────────────────────────────────────────────────────────
  function setMode(mode) {
    clickMode = mode;
    modeStartBtn.classList.toggle('active', mode === 'start');
    modeEndBtn.classList.toggle('active', mode === 'end');
  }
  modeStartBtn.onclick = () => setMode('start');
  modeEndBtn.onclick = () => setMode('end');

  // Clear routes when travel mode changes so stale routes aren't shown
  document.getElementById('mode-select').onchange = () => {
    const newMode = document.getElementById('mode-select').value;
    if (lastRoutedMode && newMode !== lastRoutedMode) {
      clearRoutes();
      routesOut.innerHTML =
        `<div style="color:#92400e; background:#fffbeb; border:1px solid #fde68a;
         border-radius:6px; padding:8px; font-size:13px;">
          ⚠ Mode changed to <strong>${newMode}</strong>. Click Route to recalculate.
        </div>`;
    }
  };

  // ── Markers & POI ───────────────────────────────────────────────────────
  function formatPoint(p) {
    if (!p) return 'Not set';
    return `${p.lat.toFixed(6)}, ${p.lon.toFixed(6)}`;
  }

  function renderNearestPoi(target, obj) {
    if (!obj || !obj.nearest_poi) { target.textContent = 'No POI found.'; return; }
    target.textContent = JSON.stringify(obj.nearest_poi, null, 2);
  }

  async function fetchNearest(lat, lon, target) {
    try {
      const res = await fetch(`/nearest_poi?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}`);
      if (!res.ok) { renderNearestPoi(target, null); return; }
      renderNearestPoi(target, await res.json());
    } catch { renderNearestPoi(target, null); }
  }

  // ── Route management ────────────────────────────────────────────────────
  function clearRoutes() {
    for (const layer of routeLayers) map.removeLayer(layer);
    routeLayers = [];
    routeMeta = [];
    selectedRouteIndex = -1;
    routesOut.innerHTML = 'No routes yet.';
    usedOut.textContent = '-';
    hideError();
  }

  function clearAll() {
    startPoint = null; endPoint = null;
    if (startMarker) map.removeLayer(startMarker);
    if (endMarker) map.removeLayer(endMarker);
    startMarker = null; endMarker = null;
    startOut.textContent = 'Not set';
    endOut.textContent = 'Not set';
    poiStartOut.textContent = 'Start not set.';
    poiEndOut.textContent = 'End not set.';
    lastRoutedMode = null;
    clearRoutes();
  }
  document.getElementById('btn-clear').onclick = clearAll;

  // ── Map click ───────────────────────────────────────────────────────────
  map.on('click', async (e) => {
    hideError();
    const p = { lat: e.latlng.lat, lon: e.latlng.lng };
    if (!pointInsideBoundary(p)) {
      showError(
        'Point outside POI coverage',
        'Please click inside the grey rectangle that marks the area where POI data is available.'
      );
      return;
    }
    if (clickMode === 'start') {
      startPoint = p;
      if (startMarker) map.removeLayer(startMarker);
      startMarker = L.marker([p.lat, p.lon], {
        icon: L.divIcon({ className: '', html: '<div style="background:#2563eb;width:14px;height:14px;border-radius:50%;border:2px solid white;box-shadow:0 1px 4px rgba(0,0,0,.4)"></div>' })
      }).addTo(map).bindPopup('Start');
      startOut.textContent = formatPoint(p);
      await fetchNearest(p.lat, p.lon, poiStartOut);
      setMode('end');
    } else {
      endPoint = p;
      if (endMarker) map.removeLayer(endMarker);
      endMarker = L.marker([p.lat, p.lon], {
        icon: L.divIcon({ className: '', html: '<div style="background:#16a34a;width:14px;height:14px;border-radius:50%;border:2px solid white;box-shadow:0 1px 4px rgba(0,0,0,.4)"></div>' })
      }).addTo(map).bindPopup('End');
      endOut.textContent = formatPoint(p);
      await fetchNearest(p.lat, p.lon, poiEndOut);
      setMode('start');
    }
  });

  // ── Route rendering ─────────────────────────────────────────────────────
  const ROUTE_COLORS = ['#2563eb', '#ef4444', '#16a34a'];

  function baseRouteStyle(idx) {
    return { color: ROUTE_COLORS[idx] || '#7c3aed', weight: 6, opacity: 0.9 };
  }

  function applyRouteHighlight(idx) {
    selectedRouteIndex = idx;
    routeLayers.forEach((layer, i) => {
      const style = baseRouteStyle(i);
      const sel = i === idx;
      layer.setStyle({ color: style.color, weight: sel ? style.weight + 4 : style.weight, opacity: sel ? 1.0 : 0.35 });
      if (sel) { try { layer.bringToFront(); } catch (e) {} }
    });
    document.querySelectorAll('.route-card').forEach((card, i) => {
      card.style.border = i === idx ? `2px solid ${ROUTE_COLORS[i] || '#111827'}` : '1px solid #ddd';
      card.style.boxShadow = i === idx ? `0 0 0 3px ${ROUTE_COLORS[i] || '#111'}22` : 'none';
    });
  }

  function renderRoutes(data) {
    clearRoutes();
    const currentMode = document.getElementById('mode-select').value;
    lastRoutedMode = currentMode;
    usedOut.textContent = data.used_routing || '-';

    const features = data.features || [];
    const routeCards = [];

    for (let i = 0; i < features.length; i++) {
      const f = features[i];
      const layer = L.geoJSON(f, { style: baseRouteStyle(i) }).addTo(map);
      routeLayers.push(layer);
      routeMeta.push(f);

      const p = f.properties || {};
      const summary = p.summary || {};
      const instructions = Array.isArray(p.instructions) ? p.instructions : [];

      const usedTag = summary.used_routing === 'poi'
        ? '<span class="route-badge badge-poi">POI snap</span>'
        : '<span class="route-badge badge-raw">raw coords</span>';

      const colorDot = `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${ROUTE_COLORS[i] || '#888'};margin-right:6px;"></span>`;

      const instrHtml = instructions.length
        ? `<ol style="margin:6px 0 0 18px; padding:0; font-size:12px; color:#374151;">
            ${instructions.map(x => `<li style="margin-bottom:3px;">${x}</li>`).join('')}
           </ol>`
        : '<div style="margin-top:6px; color:#9ca3af; font-size:12px;">No instructions.</div>';

      routeCards.push(
        `<div class="route-card" data-route-index="${i}">` +
        `<div class="route-title">${colorDot}Route #${p.rank} ${usedTag}</div>` +
        `<div style="font-size:13px; margin-bottom:4px;">` +
        `  <strong>${(p.distance_m / 1000).toFixed(2)} km</strong> &nbsp;·&nbsp; ` +
        `  <strong>${p.duration_min} min</strong> &nbsp;·&nbsp; ${p.mode}` +
        `</div>` +
        `<details><summary style="font-size:12px; cursor:pointer; color:#6b7280;">Turn-by-turn</summary>${instrHtml}</details>` +
        `</div>`
      );
    }

    routesOut.innerHTML = routeCards.length
      ? routeCards.join('')
      : '<div style="color:#6b7280;">No routes returned.</div>';

    // Fit map
    try {
      const group = L.featureGroup([startMarker, endMarker, ...routeLayers].filter(Boolean));
      map.fitBounds(group.getBounds().pad(0.15));
    } catch (e) {}

    // Card click handlers
    document.querySelectorAll('.route-card').forEach(card => {
      card.addEventListener('click', () => {
        const idx = parseInt(card.getAttribute('data-route-index') || '-1', 10);
        if (idx >= 0 && idx < routeLayers.length) {
          applyRouteHighlight(idx);
          try {
            const b = routeLayers[idx].getBounds();
            if (b.isValid()) map.fitBounds(b.pad(0.15));
          } catch (e) {}
        }
      });
    });

    if (routeLayers.length > 0) applyRouteHighlight(0);
  }

  // ── Route button ─────────────────────────────────────────────────────────
  document.getElementById('btn-route').onclick = async () => {
    hideError();
    if (!startPoint || !endPoint) {
      showError('Missing points', 'Please set both a Start and End point on the map first.');
      return;
    }

    const payload = {
      start_lat: startPoint.lat,
      start_lon: startPoint.lon,
      end_lat: endPoint.lat,
      end_lon: endPoint.lon,
      mode: document.getElementById('mode-select').value,
      k: parseInt(document.getElementById('k-input').value || '3', 10),
      snap_to_poi: document.getElementById('snap-to-poi').checked,
      fallback_to_raw: document.getElementById('fallback-to-raw').checked,
      poi_candidates: parseInt(document.getElementById('poi-candidates').value || '3', 10)
    };

    clearRoutes();
    routesOut.innerHTML = '<span class="spinner"></span> Calculating route…';

    let res, data;
    try {
      res = await fetch('/route_top3', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      data = await res.json();
    } catch (err) {
      routesOut.innerHTML = '';
      showError('Network error', 'Could not reach the routing server. Is it running?');
      return;
    }

    if (!res.ok) {
      routesOut.innerHTML = '';
      const detail = data?.detail || JSON.stringify(data);
      const isNoRoute = res.status === 404;
      showError(
        isNoRoute ? 'No route found' : `Error ${res.status}`,
        isNoRoute
          ? 'No path exists between these points. Try moving them closer to a road, or enable fallback_to_raw.'
          : detail
      );
      return;
    }

    renderRoutes(data);
  };

  loadBoundary();
</script>
</body>
</html>"""
    )


# ---------------------------------------------------------------------------
# App startup: load data
# ---------------------------------------------------------------------------

@app.get("/_info")
def info():
    return {
        "file": __file__,
        "cwd": os.getcwd(),
        "routes": [
            {"path": r.path, "name": r.name, "methods": sorted(list(r.methods or []))}
            for r in app.router.routes
        ],
    }


roads = gpd.read_file(ROADS_PATH).to_crs(epsg=3857)
roads_noded = node_roads_preserve_attrs(roads)
engine = RouteEngine(roads_noded)

pois = load_pois_csv(POIS_PATH)
if str(getattr(pois, "crs", "")) != "EPSG:4326":
    pois = pois.to_crs("EPSG:4326")

# POI spatial indexes
_poi_geoms = list(pois.geometry)
_poi_tree = STRtree(_poi_geoms)

_pois_3857 = pois.to_crs("EPSG:3857")
_poi_geoms_3857 = list(_pois_3857.geometry)
_poi_tree_3857 = STRtree(_poi_geoms_3857)
_poi_geom_id_to_idx_3857 = {id(g): i for i, g in enumerate(_poi_geoms_3857)}


# ---------------------------------------------------------------------------
# POI helpers
# ---------------------------------------------------------------------------

def _nearest_poi_index(lat: float, lon: float) -> int:
    q = Point(lon, lat)
    nearest = _poi_tree.nearest(q)
    if nearest is None:
        raise HTTPException(404, "No POI found.")
    if isinstance(nearest, (int, np.integer)):
        return int(nearest)
    return _poi_geoms.index(nearest)


def _hits_to_indices(hits, id_to_idx):
    out = []
    for h in hits:
        if isinstance(h, (int, np.integer)):
            out.append(int(h))
        else:
            idx = id_to_idx.get(id(h))
            if idx is not None:
                out.append(int(idx))
    return out


def _poi_candidates_within_radius(lat: float, lon: float, radius_m: float = 20.0, limit: int = 10):
    """
    Return (dist_m, poi_index) pairs for POIs within *radius_m* metres,
    sorted by distance ascending.  Distances are computed directly from
    the EPSG:3857 tree — no second reprojection needed.
    """
    limit = max(1, min(50, int(limit)))
    q_3857 = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
    buf = q_3857.buffer(radius_m)
    hits = _poi_tree_3857.query(buf)
    idxs = _hits_to_indices(hits, _poi_geom_id_to_idx_3857)
    if not idxs:
        return []
    dists = [(float(q_3857.distance(_poi_geoms_3857[i])), int(i)) for i in set(idxs)]
    dists.sort()
    return dists[:limit]


def _poi_payload(idx: int):
    row = pois.iloc[idx]
    props = {k: _json_safe(v) for k, v in row.drop(labels=["geometry"]).to_dict().items()}
    props["poi_latitude"] = _json_safe(float(row.geometry.y))
    props["poi_longitude"] = _json_safe(float(row.geometry.x))
    return props


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class RouteReq(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    mode: str = "car"
    k: int = 3
    snap_to_poi: bool = True
    fallback_to_raw: bool = True
    poi_candidates: int = 3
    snap_radius_m: float = 20.0   # NEW: configurable snap radius (was hardcoded 20m)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True, "roads_segments": int(len(engine.roads)), "pois": int(len(pois))}


@app.get("/graph_stats")
def graph_stats():
    G = getattr(engine, "G", None)
    if G is None:
        return {"ok": False, "error": "RouteEngine has no attribute 'G'"}
    Gu = G.to_undirected(as_view=True)
    comps = list(nx.connected_components(Gu))
    comp_sizes = sorted([len(c) for c in comps], reverse=True)
    return {
        "ok": True,
        "nodes": int(G.number_of_nodes()),
        "edges": int(G.number_of_edges()),
        "components": int(len(comps)),
        "largest_component_nodes": int(comp_sizes[0]) if comp_sizes else 0,
        "top5_component_sizes": comp_sizes[:5],
    }


@app.get("/roads_bounds")
def roads_bounds():
    """
    Return a boundary suitable for UI point-picking.
    Priority: overlap of road bbox and POI bbox, so users stay inside the area
    where both routing roads and POI coverage exist.
    Fallback: POI bbox if overlap is invalid, then road bbox if POI is empty.
    """
    # Road bounds are stored in EPSG:3857, convert corner points to EPSG:4326
    r_minx, r_miny, r_maxx, r_maxy = engine.roads.total_bounds
    road_pts = gpd.GeoSeries([Point(r_minx, r_miny), Point(r_maxx, r_maxy)], crs="EPSG:3857").to_crs("EPSG:4326")
    road_min, road_max = road_pts.iloc[0], road_pts.iloc[1]
    road_bbox = {
        "min_lon": float(road_min.x),
        "min_lat": float(road_min.y),
        "max_lon": float(road_max.x),
        "max_lat": float(road_max.y),
    }

    # POI bounds are already EPSG:4326
    try:
        p_minx, p_miny, p_maxx, p_maxy = pois.total_bounds
        poi_bbox = {
            "min_lon": float(p_minx),
            "min_lat": float(p_miny),
            "max_lon": float(p_maxx),
            "max_lat": float(p_maxy),
        }
        has_poi = np.isfinite([p_minx, p_miny, p_maxx, p_maxy]).all()
    except Exception:
        poi_bbox = None
        has_poi = False

    # Prefer overlap between road bbox and POI bbox
    if has_poi and poi_bbox is not None:
        overlap = {
            "min_lon": max(road_bbox["min_lon"], poi_bbox["min_lon"]),
            "min_lat": max(road_bbox["min_lat"], poi_bbox["min_lat"]),
            "max_lon": min(road_bbox["max_lon"], poi_bbox["max_lon"]),
            "max_lat": min(road_bbox["max_lat"], poi_bbox["max_lat"]),
        }
        if overlap["min_lon"] < overlap["max_lon"] and overlap["min_lat"] < overlap["max_lat"]:
            return {
                "crs": "EPSG:4326",
                **overlap,
                "source": "road_poi_overlap",
                "road_bbox": road_bbox,
                "poi_bbox": poi_bbox,
            }

        # If overlap is invalid, fall back to POI bbox to keep UI constrained to POI area
        return {
            "crs": "EPSG:4326",
            **poi_bbox,
            "source": "poi_bbox",
            "road_bbox": road_bbox,
            "poi_bbox": poi_bbox,
        }

    # Final fallback if POI dataset is empty/unavailable
    return {
        "crs": "EPSG:4326",
        **road_bbox,
        "source": "road_bbox",
        "road_bbox": road_bbox,
        "poi_bbox": poi_bbox,
    }


@app.get("/suggest_points")
def suggest_points(n: int = 5):
    n = max(1, min(50, n))
    sampled = []
    roads_ll = engine.roads.to_crs("EPSG:4326")
    step = max(1, len(roads_ll) // n)
    for i in range(0, len(roads_ll), step):
        if len(sampled) >= n:
            break
        geom = roads_ll.geometry.iloc[i]
        if geom is None or geom.is_empty:
            continue
        try:
            p = geom.interpolate(0.5, normalized=True)
            sampled.append({"lon": float(p.x), "lat": float(p.y)})
        except Exception:
            continue
    return {"count": len(sampled), "points": sampled}


@app.get("/suggest_pair")
def suggest_pair():
    roads_ll = engine.roads.to_crs("EPSG:4326")
    for i in range(len(roads_ll)):
        geom = roads_ll.geometry.iloc[i]
        if geom is None or geom.is_empty:
            continue
        try:
            if geom.geom_type == "MultiLineString":
                geom = max(list(geom.geoms), key=lambda g: g.length)
            p1 = geom.interpolate(0.25, normalized=True)
            p2 = geom.interpolate(0.75, normalized=True)
        except Exception:
            continue
        start = {"lon": float(p1.x), "lat": float(p1.y)}
        end = {"lon": float(p2.x), "lat": float(p2.y)}
        try:
            s_idx = _nearest_poi_index(start["lat"], start["lon"])
            e_idx = _nearest_poi_index(end["lat"], end["lon"])
            start_poi = _poi_payload(s_idx)
            end_poi = _poi_payload(e_idx)
        except Exception:
            start_poi = None
            end_poi = None
        return {"start": start, "end": end, "nearest_poi": {"start": start_poi, "end": end_poi}}
    raise HTTPException(404, "No suitable road geometry found to suggest a pair.")


@app.get("/nearest_poi")
def nearest_poi(lat: float, lon: float):
    idx = _nearest_poi_index(lat, lon)
    return {"query": {"lat": float(lat), "lon": float(lon)}, "nearest_poi": _poi_payload(idx)}


# ---------------------------------------------------------------------------
# Main routing endpoint
# ---------------------------------------------------------------------------

@app.post("/route_top3")
def route_top3(req: RouteReq):
    mode = req.mode.strip().lower()
    if mode not in ("car", "motorcycle"):
        raise HTTPException(400, "mode must be 'car' or 'motorcycle'")

    snap_radius = max(1.0, min(200.0, float(req.snap_radius_m)))

    print(
        f"[route_tool] /route_top3 | mode={mode} k={req.k} snap_to_poi={req.snap_to_poi} "
        f"fallback={req.fallback_to_raw} radius={snap_radius}m candidates={req.poi_candidates}"
    )

    try:
        def _try_route(sl, so, el, eo, k=None):
            return engine.route_top3(so, sl, eo, el, mode=mode, k=k or req.k)

        used = "raw"
        snap_info = None

        if req.snap_to_poi:
            cand_limit = int(req.poi_candidates)

            # FIX: distances come directly from the 3857 tree — no extra reprojection
            s_dists = _poi_candidates_within_radius(req.start_lat, req.start_lon, snap_radius, cand_limit)
            e_dists = _poi_candidates_within_radius(req.end_lat, req.end_lon, snap_radius, cand_limit)

            if not s_dists or not e_dists:
                start_lat, start_lon = float(req.start_lat), float(req.start_lon)
                end_lat, end_lon = float(req.end_lat), float(req.end_lon)
                print("[route_tool] fallback to raw | no POI within radius")
                routes = _try_route(start_lat, start_lon, end_lat, end_lon)
                snap_info = {
                    "rule": f"{snap_radius}m",
                    "status": "skipped",
                    "reason": "no_poi_within_radius",
                    "start_has_poi": bool(s_dists),
                    "end_has_poi": bool(e_dists),
                }
            else:
                def _make_cand_info(lat, lon, dists):
                    return [
                        {"idx": idx, "dist_m": round(d, 2), "poi": _poi_payload(idx)}
                        for d, idx in dists
                    ]

                s_infos = _make_cand_info(req.start_lat, req.start_lon, s_dists)
                e_infos = _make_cand_info(req.end_lat, req.end_lon, e_dists)

                snap_info = {
                    "rule": f"{snap_radius}m",
                    "status": "attempted",
                    "start": {"input": {"lat": float(req.start_lat), "lon": float(req.start_lon)}, "candidates": s_infos},
                    "end": {"input": {"lat": float(req.end_lat), "lon": float(req.end_lon)}, "candidates": e_infos},
                }

                routes = []
                chosen = None
                pairs_tried = 0
                max_pairs = 4

                for si in s_infos:
                    if chosen:
                        break
                    for ei in e_infos:
                        pairs_tried += 1
                        if pairs_tried > max_pairs:
                            break
                        sl = float(si["poi"]["poi_latitude"])
                        so = float(si["poi"]["poi_longitude"])
                        el = float(ei["poi"]["poi_latitude"])
                        eo = float(ei["poi"]["poi_longitude"])
                        probe = _try_route(sl, so, el, eo, k=1)
                        if probe:
                            chosen = {"start": si, "end": ei, "pairs_tried": pairs_tried}
                            start_lat, start_lon = sl, so
                            end_lat, end_lon = el, eo
                            used = "poi"
                            break
                    if chosen or pairs_tried > max_pairs:
                        break

                if chosen:
                    snap_info["status"] = "used_poi"
                    snap_info["chosen"] = chosen
                    routes = _try_route(start_lat, start_lon, end_lat, end_lon)
                else:
                    snap_info["status"] = "failed_poi"
                    snap_info["pairs_tried"] = pairs_tried
                    if req.fallback_to_raw:
                        start_lat, start_lon = float(req.start_lat), float(req.start_lon)
                        end_lat, end_lon = float(req.end_lat), float(req.end_lon)
                        print("[route_tool] fallback to raw | all POI pairs failed")
                        routes = _try_route(start_lat, start_lon, end_lat, end_lon)
                    else:
                        routes = []
        else:
            start_lat, start_lon = float(req.start_lat), float(req.start_lon)
            end_lat, end_lon = float(req.end_lat), float(req.end_lon)
            routes = _try_route(start_lat, start_lon, end_lat, end_lon)

        if not routes:
            raise HTTPException(404, "No route found.")

    except HTTPException:
        raise
    except nx.NetworkXNoPath:
        raise HTTPException(404, "No route found (disconnected road network).")
    except Exception as e:
        print("[route_tool] ERROR in /route_top3")
        traceback.print_exc()
        raise HTTPException(400, str(e))

    # ── Build GeoJSON response ────────────────────────────────────────────
    features = []
    for r in routes:
        geom_ll = gpd.GeoSeries([r.geometry_3857], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
        coords = list(getattr(geom_ll, "coords", []))

        # Gather road names from the engine graph for turn instructions
        # (best-effort: names may be absent in the data)
        road_names: list = []
        if hasattr(r, "edge_set"):
            # edge_set gives us (u,v) pairs; look up road names
            for u, v in sorted(r.edge_set):
                if engine.G.has_edge(u, v):
                    props = engine.G.edges[u, v].get("props", {})
                    name = props.get("name") or props.get("ref") or ""
                    road_names.append(str(name) if name else "")

        instructions = _build_instructions(coords, road_names)

        start_txt = f"{coords[0][1]:.6f}, {coords[0][0]:.6f}" if coords else "-"
        end_txt = f"{coords[-1][1]:.6f}, {coords[-1][0]:.6f}" if coords else "-"

        features.append({
            "type": "Feature",
            "properties": {
                "rank": r.rank,
                "distance_m": round(r.distance_m, 2),
                "duration_s": round(r.duration_s, 2),
                "duration_min": round(r.duration_s / 60.0, 2),
                "mode": mode,
                "summary": {
                    "start": start_txt,
                    "end": end_txt,
                    "used_routing": used,
                },
                "instructions": instructions,
            },
            "geometry": mapping(geom_ll),
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "routing_endpoints": {
            "start": {"lat": float(start_lat), "lon": float(start_lon)},
            "end": {"lat": float(end_lat), "lon": float(end_lon)},
        },
        "poi_snap": snap_info,
        "used_routing": used,
    }
