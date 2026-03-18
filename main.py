from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import networkx as nx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse

from config import Settings, get_settings
from schemas import (
    BoundsResponse, ErrorDetail, ErrorResponse,
    GraphStatsResponse, HealthResponse, NearestPoiResponse,
    RouteRequest, RouteResponse, LatLon, TripRecord,
)
from services import RoutingService

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("vn_routing")


# ── App state ─────────────────────────────────────────────────────────────────

_service: RoutingService | None = None


def get_service() -> RoutingService:
    if _service is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Service not initialised yet")
    return _service


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    settings = get_settings()
    logger.info("=== VN Routing API starting up ===")
    t0 = time.perf_counter()
    _service = RoutingService(settings)
    logger.info("=== Ready in %.1fs ===", time.perf_counter() - t0)
    yield
    logger.info("=== VN Routing API shutting down ===")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()

    app = FastAPI(
        title=cfg.api_title,
        version=cfg.api_version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
        responses={
            422: {"model": ErrorResponse, "description": "Validation error"},
            500: {"model": ErrorResponse, "description": "Internal server error"},
        },
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request ID + timing middleware ────────────────────────────────────
    @app.middleware("http")
    async def request_middleware(request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        t0 = time.perf_counter()
        logger.info("→ %s %s [%s]", request.method, request.url.path, request_id)
        response = await call_next(request)
        dt = time.perf_counter() - t0
        logger.info("← %s %s [%s] %.0fms",
                    response.status_code, request.url.path, request_id, dt * 1000)
        response.headers["X-Request-ID"]      = request_id
        response.headers["X-Response-Time-Ms"] = str(round(dt * 1000))
        return response

    # ── Global error handlers ─────────────────────────────────────────────
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=ErrorDetail(
                    code=f"HTTP_{exc.status_code}",
                    message=str(exc.detail),
                )
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="INTERNAL_ERROR",
                    message="An unexpected error occurred.",
                    detail=str(exc) if cfg.debug else None,
                )
            ).model_dump(),
        )

    # ── Register routers ──────────────────────────────────────────────────
    prefix = cfg.api_prefix
    app.include_router(health_router,  prefix=prefix, tags=["Health"])
    app.include_router(routing_router, prefix=prefix, tags=["Routing"])
    app.include_router(trips_router,   prefix=prefix, tags=["Trips"])
    app.include_router(poi_router,     prefix=prefix, tags=["POI"])
    app.include_router(map_router,     tags=["Map UI"])

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")

    return app


# ── Health router ─────────────────────────────────────────────────────────────

from fastapi import APIRouter

health_router = APIRouter()


@health_router.get("/health", response_model=HealthResponse, summary="Health check")
def health(
    svc: RoutingService = Depends(get_service),
    cfg: Settings       = Depends(get_settings),
):
    return HealthResponse(
        ok=True,
        roads_segments=int(len(svc.engine.roads)),
        pois=int(len(svc.pois)),
        api_version=cfg.api_version,
    )


@health_router.get("/health/graph", response_model=GraphStatsResponse,
                   summary="Graph connectivity stats")
def graph_stats(svc: RoutingService = Depends(get_service)):
    stats = svc.graph_stats()
    return GraphStatsResponse(**stats)


# ── Routing router ────────────────────────────────────────────────────────────

routing_router = APIRouter()


@routing_router.post(
    "/routes",
    response_model=RouteResponse,
    summary="Compute top-K diverse routes",
    responses={404: {"model": ErrorResponse, "description": "No route found"}},
)
def compute_routes(
    req: RouteRequest,
    svc: RoutingService = Depends(get_service),
):
    logger.info(
        "route request | start=(%.5f,%.5f) end=(%.5f,%.5f) mode=%s k=%d",
        req.start_lat, req.start_lon, req.end_lat, req.end_lon, req.mode, req.k,
    )
    try:
        result = svc.compute_routes(
            start_lat=req.start_lat, start_lon=req.start_lon,
            end_lat=req.end_lat,     end_lon=req.end_lon,
            mode=req.mode,
            k=req.k,
            snap_to_poi=req.snap_to_poi,
            fallback_to_raw=req.fallback_to_raw,
            poi_candidates=req.poi_candidates,
            snap_radius_m=req.snap_radius_m,
        )
    except ValueError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(e))
    except nx.NetworkXNoPath:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            detail="No path exists between the snapped points.")

    trip_id = str(uuid.uuid4())
    result.trip_id = trip_id
    svc.store_trip(trip_id, req.model_dump(), result)
    return result


@routing_router.get("/routes/bounds", response_model=BoundsResponse,
                    summary="Bounding box of road data")
def roads_bounds(svc: RoutingService = Depends(get_service)):
    return BoundsResponse(**svc.roads_bounds())


# ── Trips router ──────────────────────────────────────────────────────────────

trips_router = APIRouter()


@trips_router.get(
    "/trips/{trip_id}",
    response_model=TripRecord,
    summary="Retrieve a recorded trip by ID",
    responses={404: {"model": ErrorResponse, "description": "Trip not found"}},
)
def get_trip(trip_id: str, svc: RoutingService = Depends(get_service)):
    record = svc.get_trip(trip_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            detail=f"Trip '{trip_id}' not found.")
    return record


# ── POI router ────────────────────────────────────────────────────────────────

poi_router = APIRouter()


@poi_router.get(
    "/pois/nearest",
    response_model=NearestPoiResponse,
    summary="Nearest POI to a coordinate",
)
def nearest_poi(
    lat: float,
    lon: float,
    snap_radius_m: float = 20.0,
    svc: RoutingService = Depends(get_service),
    cfg: Settings       = Depends(get_settings),
):
    try:
        idx = svc.nearest_poi_index(lat, lon)
    except ValueError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(e))

    # Compute actual distance to nearest POI
    candidates = svc.candidates_within_radius(lat, lon, radius_m=99999, limit=1)
    dist_m = candidates[0][0] if candidates else float("inf")
    within = dist_m <= snap_radius_m

    return NearestPoiResponse(
        query=LatLon(lat=lat, lon=lon),
        nearest_poi=svc.poi_payload(idx) if within else None,
        distance_m=round(dist_m, 1),
        within_snap_radius=within,
    )


# ── Map UI router ─────────────────────────────────────────────────────────────

map_router = APIRouter()


@map_router.get("/map", response_class=HTMLResponse, include_in_schema=False)
def map_ui(cfg: Settings = Depends(get_settings)):
    api_prefix = cfg.api_prefix
    return HTMLResponse(_map_html(api_prefix))


def _map_html(api_prefix: str) -> str:
    return _MAP_HTML_TEMPLATE.replace("__API_PREFIX__", api_prefix)


_MAP_HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>VN Routing Tool</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <style>
    html,body{height:100%;margin:0;}
    #wrap{display:flex;height:100%;position:relative;}
    #history-panel{
      position:absolute;top:10px;left:10px;z-index:1000;
      background:rgba(255,255,255,0.95);border:1px solid #ddd;border-radius:10px;
      padding:10px 12px;width:230px;max-height:340px;overflow-y:auto;
      font-family:system-ui,-apple-system,sans-serif;
      box-shadow:0 2px 12px rgba(0,0,0,0.18);
    }
    #history-panel h4{margin:0 0 8px;font-size:13px;color:#374151;font-weight:700;letter-spacing:.3px;}
    .history-item{
      cursor:pointer;padding:8px 10px;border-radius:8px;border:1px solid #e5e7eb;
      margin-bottom:6px;background:white;transition:background .15s;font-size:12px;
    }
    .history-item:hover{background:#eff6ff;border-color:#93c5fd;}
    .history-item.active{background:#dbeafe;border-color:#2563eb;}
    .hi-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:3px;}
    .hi-num{font-weight:700;color:#1d4ed8;font-size:12px;}
    .hi-mode{font-size:11px;color:#6b7280;}
    .hi-meta{color:#374151;font-weight:600;font-size:12px;}
    .hi-time{font-size:10px;color:#9ca3af;margin-top:2px;}
    #map{flex:2;}
    #panel{flex:1;padding:12px;font-family:system-ui,-apple-system,sans-serif;overflow:auto;border-left:1px solid #ddd;background:#fafafa;min-width:280px;max-width:380px;}
    .hint{color:#666;font-size:13px;margin-bottom:10px;}
    .section{margin-bottom:16px;}
    .row{margin-bottom:8px;}
    .label{font-size:12px;color:#555;margin-bottom:4px;}
    .value{font-weight:600;word-break:break-word;}
    button{padding:8px 10px;border:1px solid #ccc;background:white;cursor:pointer;border-radius:6px;margin-right:8px;margin-bottom:8px;}
    button.active{border-color:#333;font-weight:700;}
    pre{white-space:pre-wrap;background:white;border:1px solid #ddd;padding:8px;border-radius:6px;font-size:12px;}
    .route-card{cursor:pointer;transition:box-shadow .15s;}
    .route-title{font-weight:700;margin-bottom:6px;}
    .badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;margin-left:6px;}
    .badge-poi{background:#d1fae5;color:#065f46;}
    .badge-raw{background:#fee2e2;color:#991b1b;}
    #error-banner{display:none;background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;padding:10px 12px;margin-bottom:12px;color:#991b1b;font-size:13px;}
    .spinner{display:inline-block;width:14px;height:14px;border:2px solid #ccc;border-top-color:#333;border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:6px;}
    @keyframes spin{to{transform:rotate(360deg);}}
    .traffic-panel{margin-top:10px;border-top:1px solid #eee;padding-top:10px;}
    .traffic-row{display:flex;align-items:center;gap:8px;margin-bottom:7px;}
    .traffic-dot{width:13px;height:13px;border-radius:50%;flex-shrink:0;cursor:pointer;border:2px solid transparent;transition:border-color .15s;}
    .traffic-dot.active{border-color:#111;}
    .traffic-input{width:70px;padding:3px 6px;border:1px solid #ddd;border-radius:5px;font-size:12px;}
    .traffic-label{font-size:12px;color:#555;flex:1;}
    .traffic-result{font-size:12px;font-weight:700;color:#2563eb;margin-top:6px;}
    .traffic-hint{font-size:11px;color:#9ca3af;margin-top:3px;}
  </style>
</head>
<body>
<div id="wrap">
  <div id="map"></div>
  <div id="history-panel">
    <h4>&#128199; Trip history</h4>
    <div id="history-list"><div style="font-size:11px;color:#9ca3af;">No trips yet.</div></div>
  </div>
  <div id="panel">
    <div id="error-banner"><strong id="err-title"></strong><div id="err-hint" style="margin-top:4px;"></div></div>
    <div class="section">
      <div class="hint">Click map to set Start and End, then click Route.</div>
      <button id="mode-start" class="active">&#128205; Set Start</button>
      <button id="mode-end">&#127937; Set End</button>
      <button id="btn-route">&#128506; Route</button>
      <button id="btn-reverse" title="Swap start and end">&#8645; Reverse</button>
      <button id="btn-clear">&#10005; Clear</button>
    </div>
    <div class="section">
      <div class="row"><div class="label">Start</div><div id="start-out" class="value">Not set</div></div>
      <div class="row"><div class="label">End</div><div id="end-out" class="value">Not set</div></div>
      <div class="row"><div class="label">Routing used</div><div id="used-out" class="value">-</div></div>
      <div class="row"><div class="label">Road boundary</div><div id="boundary-out" class="value">Loading...</div></div>
    </div>
    <div class="section">
      <div class="row"><label><input type="checkbox" id="snap-to-poi" checked> snap_to_poi</label></div>
      <div class="row"><label><input type="checkbox" id="fallback-to-raw" checked> fallback_to_raw</label></div>
      <div class="row"><div class="label">Mode</div>
        <select id="mode-select"><option value="car">&#128663; car</option><option value="motorcycle">&#127949; motorcycle</option></select>
      </div>
      <div class="row"><div class="label">k (max routes)</div><input id="k-input" type="number" value="3" min="1" max="5" style="width:50px"/></div>
      <div class="row"><div class="label">snap radius (m)</div><input id="snap-radius" type="number" value="20" min="1" max="200" style="width:60px"/></div>
    </div>
    <div class="section"><div class="label">Nearest POI - Start</div><pre id="poi-start-out">Start not set.</pre></div>
    <div class="section"><div class="label">Nearest POI - End</div><pre id="poi-end-out">End not set.</pre></div>
    <div class="section"><div class="label">Routes</div><div id="routes-out">No routes yet.</div></div>
    <div class="section" id="traffic-section" style="display:none;">
      <div class="label">&#128678; Traffic Annotation <span style="font-size:11px;color:#9ca3af;font-weight:400">(selected route)</span></div>
      <div style="font-size:12px;color:#6b7280;margin-bottom:8px;">Click a color, enter length (m) of congestion along the route.</div>
      <div class="traffic-row">
        <div class="traffic-dot" id="td-red" style="background:#ef4444;" title="Heavy (0.1x speed)" onclick="selectTraffic('red')"></div>
        <div class="traffic-label">Heavy jam</div>
        <input class="traffic-input" id="ti-red" type="number" min="0" value="0" placeholder="0 m"/>
        <span style="font-size:11px;color:#9ca3af;">m</span>
      </div>
      <div class="traffic-row">
        <div class="traffic-dot" id="td-orange" style="background:#f97316;" title="Slow (0.3x speed)" onclick="selectTraffic('orange')"></div>
        <div class="traffic-label">Slow traffic</div>
        <input class="traffic-input" id="ti-orange" type="number" min="0" value="0" placeholder="0 m"/>
        <span style="font-size:11px;color:#9ca3af;">m</span>
      </div>
      <div class="traffic-row">
        <div class="traffic-dot" id="td-yellow" style="background:#eab308;" title="Moving slowly (0.5x speed)" onclick="selectTraffic('yellow')"></div>
        <div class="traffic-label">Moving slowly</div>
        <input class="traffic-input" id="ti-yellow" type="number" min="0" value="0" placeholder="0 m"/>
        <span style="font-size:11px;color:#9ca3af;">m</span>
      </div>
      <button onclick="applyTraffic()" style="margin-top:4px;background:#2563eb;color:white;border:none;font-weight:600;">&#9654; Recalculate</button>
      <button onclick="clearTraffic()" style="margin-top:4px;">&#10005; Clear</button>
      <div id="traffic-result" class="traffic-result"></div>
      <div class="traffic-hint">travel_speed = maxspeed × util_ratio × traffic_ratio</div>
    </div>
  </div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
var API = "__API_PREFIX__";
var map = L.map("map").setView([10.78, 106.65], 14);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {maxZoom: 19, attribution: "&copy; OpenStreetMap"}).addTo(map);

var clickMode = "start";
var startPoint = null, endPoint = null;
var startMarker = null, endMarker = null;
var routeLayers = [], routeLabels = [], boundaryBounds = null, lastMode = null;
var COLORS = ["#2563eb", "#ef4444", "#16a34a"];
var dragTimer = null;

// ── Trip history ─────────────────────────────────────────────────────────────
var tripHistory = [];   // [{trip_id, created_at, mode, duration_min, distance_km, data}]
var activeHistoryIdx = null;

function _nowStr() {
  var d = new Date();
  var pad = function(n){return n<10?"0"+n:n;};
  return d.getFullYear()+"-"+pad(d.getMonth()+1)+"-"+pad(d.getDate())+" "+
         pad(d.getHours())+":"+pad(d.getMinutes())+":"+pad(d.getSeconds());
}

function addTripToHistory(data) {
  var f0 = data.features && data.features[0];
  if (!f0) return;
  var p = f0.properties || {};
  var modeIcon = (p.mode === "motorcycle") ? "🏍" : "🚗";
  var durMin = Math.ceil(p.duration_min);
  var distKm = ((p.distance_m||0)/1000).toFixed(2);
  tripHistory.unshift({
    trip_id: data.trip_id || null,
    created_at: _nowStr(),
    mode: p.mode || "car",
    mode_icon: modeIcon,
    duration_min: durMin,
    distance_km: distKm,
    data: data,
  });
  activeHistoryIdx = 0;
  renderHistoryPanel();
}

function renderHistoryPanel() {
  var el = ge("history-list");
  if (!tripHistory.length) {
    el.innerHTML = "<div style='font-size:11px;color:#9ca3af;'>No trips yet.</div>";
    return;
  }
  el.innerHTML = tripHistory.map(function(t, i) {
    var isActive = (i === activeHistoryIdx);
    var durStr = t.duration_min >= 60
      ? Math.floor(t.duration_min/60)+"h "+(t.duration_min%60)+"m"
      : t.duration_min+" min";
    return "<div class='history-item"+(isActive?" active":"")+"' onclick='loadTripFromHistory("+i+")'>" +
      "<div class='hi-top'>" +
        "<span class='hi-num'>#"+(tripHistory.length-i)+"</span>" +
        "<span class='hi-mode'>"+t.mode_icon+" "+t.mode+"</span>" +
      "</div>" +
      "<div class='hi-meta'>"+durStr+" &nbsp;·&nbsp; "+t.distance_km+" km</div>" +
      "<div class='hi-time'>"+t.created_at+"</div>" +
    "</div>";
  }).join("");
}

function loadTripFromHistory(idx) {
  activeHistoryIdx = idx;
  renderHistoryPanel();
  var t = tripHistory[idx];

  // Restore start / end markers from stored routing_endpoints
  var ep = t.data.routing_endpoints;
  if (ep) {
    if (ep.start) {
      var s = ep.start;
      startPoint = {lat: s.lat, lon: s.lon};
      if (startMarker) map.removeLayer(startMarker);
      startMarker = makeMarker(s.lat, s.lon, "Start", "#16a34a");
      ge("start-out").textContent = s.lat.toFixed(6) + ", " + s.lon.toFixed(6);
    }
    if (ep.end) {
      var e = ep.end;
      endPoint = {lat: e.lat, lon: e.lon};
      if (endMarker) map.removeLayer(endMarker);
      endMarker = makeMarker(e.lat, e.lon, "End", "#dc2626");
      ge("end-out").textContent = e.lat.toFixed(6) + ", " + e.lon.toFixed(6);
    }
  }

  renderRoutes(t.data, true, true);

  // Smooth fly to fit the loaded trip
  try {
    var grp = L.featureGroup([startMarker, endMarker].concat(routeLayers).filter(Boolean));
    map.flyToBounds(grp.getBounds().pad(0.15), {duration: 0.8});
  } catch(e) {}
}

// ── Custom labeled marker icons ──────────────────────────────────────────────
function makeIcon(label, color) {
  var html = "<div style='" +
    "background:white;" +
    "border-radius:20px;" +
    "padding:5px 10px 5px 7px;" +
    "display:inline-flex;align-items:center;gap:5px;" +
    "box-shadow:0 1px 6px rgba(0,0,0,0.28);white-space:nowrap;" +
    "font-family:system-ui,-apple-system,sans-serif;" +
    "font-size:13px;font-weight:700;color:#111;" +
    "white-space:nowrap;" +
    "'>" +
    "<span style='width:10px;height:10px;border-radius:50%;background:" + color + ";display:inline-block;flex-shrink:0;'></span>" +
    label +
    "</div>";
  return L.divIcon({html: html, className: "", iconAnchor: [0, 18]});
}

function ge(id) { return document.getElementById(id); }
function showErr(t, h) { ge("err-title").textContent = t; ge("err-hint").textContent = h || ""; ge("error-banner").style.display = "block"; }
function hideErr() { ge("error-banner").style.display = "none"; }
function setClickMode(m) {
  clickMode = m;
  ge("mode-start").classList.toggle("active", m === "start");
  ge("mode-end").classList.toggle("active", m === "end");
}

ge("mode-start").onclick = function() { setClickMode("start"); };
ge("mode-end").onclick = function() { setClickMode("end"); };
ge("mode-select").onchange = function() {
  if (lastMode && ge("mode-select").value !== lastMode) {
    clearRoutes();
    ge("routes-out").innerHTML = "<div style='color:#92400e'>Mode changed - click Route to recalculate.</div>";
  }
};

function clearRouteLabels() {
  routeLabels.forEach(function(l) { map.removeLayer(l); });
  routeLabels = [];
}

function clearRoutes() {
  routeLayers.forEach(function(l) { map.removeLayer(l); });
  routeLayers = [];
  clearRouteLabels();
  ge("routes-out").innerHTML = "No routes yet.";
  ge("used-out").textContent = "-";
  hideErr();
}

function clearAll() {
  startPoint = null; endPoint = null;
  if (startMarker) map.removeLayer(startMarker);
  if (endMarker) map.removeLayer(endMarker);
  startMarker = null; endMarker = null;
  ge("start-out").textContent = "Not set";
  ge("end-out").textContent = "Not set";
  ge("poi-start-out").textContent = "Start not set.";
  ge("poi-end-out").textContent = "End not set.";
  lastMode = null;
  clearRoutes();
  ge("traffic-section").style.display = "none";
  ge("traffic-result").textContent = "";
}
ge("btn-clear").onclick = clearAll;

ge("btn-reverse").onclick = function() {
  if (!startPoint || !endPoint) {
    showErr("Missing points", "Set both Start and End before reversing.");
    return;
  }
  // Swap coordinates
  var tmp = startPoint;
  startPoint = endPoint;
  endPoint = tmp;

  // Recreate markers at swapped positions
  if (startMarker) map.removeLayer(startMarker);
  if (endMarker)   map.removeLayer(endMarker);
  startMarker = makeMarker(startPoint.lat, startPoint.lon, "Start", "#16a34a");
  endMarker   = makeMarker(endPoint.lat,   endPoint.lon,   "End",   "#dc2626");

  // Update coordinate display
  ge("start-out").textContent = startPoint.lat.toFixed(6) + ", " + startPoint.lon.toFixed(6);
  ge("end-out").textContent   = endPoint.lat.toFixed(6)   + ", " + endPoint.lon.toFixed(6);

  // Lookup nearest POI for swapped positions
  fetchPoi(startPoint.lat, startPoint.lon, ge("poi-start-out"));
  fetchPoi(endPoint.lat,   endPoint.lon,   ge("poi-end-out"));

  // Auto re-route
  ge("btn-route").click();
};

function inBounds(p) {
  if (!boundaryBounds || !p) return true;
  return p.lat >= boundaryBounds.min_lat && p.lat <= boundaryBounds.max_lat &&
         p.lon >= boundaryBounds.min_lon && p.lon <= boundaryBounds.max_lon;
}

async function fetchPoi(lat, lon, el) {
  try {
    var r = await fetch(API + "/pois/nearest?lat=" + lat + "&lon=" + lon + "&snap_radius_m=20");
    if (r.ok) {
      var d = await r.json();
      if (d.nearest_poi && d.within_snap_radius) {
        el.textContent = JSON.stringify(d.nearest_poi, null, 2);
      } else {
        el.textContent = "No POI within 20m (nearest: " + d.distance_m + "m away)";
        el.style.color = "#9ca3af";
      }
    } else { el.textContent = "Error"; }
  } catch(e) { el.textContent = "Error"; }
}

// ── Midpoint of a GeoJSON LineString for label placement ────────────────────
function midpointOfFeature(f) {
  var coords = f.geometry && f.geometry.coordinates;
  if (!coords || coords.length === 0) return null;
  var mid = coords[Math.floor(coords.length / 2)];
  return L.latLng(mid[1], mid[0]);
}

// ── Place a floating label on the route midpoint ────────────────────────────
function addRouteLabel(f, i, visible) {
  var pt = midpointOfFeature(f);
  if (!pt) return null;
  var p = f.properties || {};
  var dist = ((p.distance_m || 0) / 1000).toFixed(2);
  var dur = Math.ceil(p.duration_min);
  var color = COLORS[i] || "#7c3aed";
  var mode = (ge("mode-select") && ge("mode-select").value === "motorcycle") ? "&#127949;" : "&#128663;";
  var html = "<div style='" +
    "background:white;" +
    "border-radius:10px;" +
    "padding:7px 12px;" +
    "font-family:system-ui,-apple-system,sans-serif;" +
    "box-shadow:0 2px 8px rgba(0,0,0,0.22);" +
    "pointer-events:none;" +
    "display:" + (visible ? "inline-block" : "none") + ";" +
    "width:auto;" +
    "' class='route-lbl'>" +
    "<div style='display:flex;align-items:center;gap:5px;'>" +
    "<span style='font-size:14px;'>" + mode + "</span>" +
    "<span style='font-size:14px;font-weight:700;color:#111;'>" + (dur >= 60 ? Math.floor(dur/60) + " hr " + (dur%60) + " min" : dur + " min") + "</span>" +
    "</div>" +
    "<div style='font-size:11px;color:#666;margin-top:1px;padding-left:20px;'>" + dist + " km</div>" +
    "</div>";
  var icon = L.divIcon({html: html, className: "", iconSize: null, iconAnchor: [0, 0]});
  var lbl = L.marker(pt, {icon: icon, interactive: false, zIndexOffset: 600}).addTo(map);
  return lbl;
}

// ── Auto-reroute after drag ──────────────────────────────────────────────────
async function rerouteAfterDrag() {
  if (!startPoint || !endPoint) return;
  clearRoutes();
  ge("routes-out").innerHTML = "<span class='spinner'></span> Recalculating...";
  var body = JSON.stringify({
    start_lat: startPoint.lat, start_lon: startPoint.lon,
    end_lat: endPoint.lat, end_lon: endPoint.lon,
    mode: ge("mode-select").value,
    k: +ge("k-input").value,
    snap_to_poi: ge("snap-to-poi").checked,
    fallback_to_raw: ge("fallback-to-raw").checked,
    snap_radius_m: +ge("snap-radius").value
  });
  var res, data;
  try {
    res = await fetch(API + "/routes", {method: "POST", headers: {"Content-Type": "application/json"}, body: body});
    data = await res.json();
  } catch(e) {
    ge("routes-out").innerHTML = "";
    showErr("Network error", "Could not reach the server.");
    return;
  }
  if (!res.ok) {
    ge("routes-out").innerHTML = "";
    var msg = (data && data.error && data.error.message) ? data.error.message : JSON.stringify(data);
    showErr(res.status === 404 ? "No route found" : "Error " + res.status,
            res.status === 404 ? "Try moving points closer to a road, or enable fallback_to_raw." : msg);
    return;
  }
  renderRoutes(data);
}

// ── Create a draggable labeled marker ───────────────────────────────────────
function makeMarker(lat, lon, label, color) {
  var m = L.marker([lat, lon], {
    draggable: true,
    icon: makeIcon(label, color)
  }).addTo(map);
  m.on("dragend", function(e) {
    var ll = e.target.getLatLng();
    var p = {lat: ll.lat, lon: ll.lng};
    if (label === "Start") {
      startPoint = p;
      ge("start-out").textContent = p.lat.toFixed(6) + ", " + p.lon.toFixed(6);
      fetchPoi(p.lat, p.lon, ge("poi-start-out"));
    } else {
      endPoint = p;
      ge("end-out").textContent = p.lat.toFixed(6) + ", " + p.lon.toFixed(6);
      fetchPoi(p.lat, p.lon, ge("poi-end-out"));
    }
    // Debounce: wait 400ms after drag stops before rerouting
    clearTimeout(dragTimer);
    dragTimer = setTimeout(function() {
      if (startPoint && endPoint) rerouteAfterDrag();
    }, 400);
  });
  return m;
}

map.on("click", async function(e) {
  hideErr();
  var p = {lat: e.latlng.lat, lon: e.latlng.lng};
  if (!inBounds(p)) { showErr("Outside road data", "Click inside the grey boundary rectangle."); return; }
  if (clickMode === "start") {
    startPoint = p;
    if (startMarker) map.removeLayer(startMarker);
    startMarker = makeMarker(p.lat, p.lon, "Start", "#16a34a");
    ge("start-out").textContent = p.lat.toFixed(6) + ", " + p.lon.toFixed(6);
    fetchPoi(p.lat, p.lon, ge("poi-start-out"));
    setClickMode("end");
  } else {
    endPoint = p;
    if (endMarker) map.removeLayer(endMarker);
    endMarker = makeMarker(p.lat, p.lon, "End", "#dc2626");
    ge("end-out").textContent = p.lat.toFixed(6) + ", " + p.lon.toFixed(6);
    fetchPoi(p.lat, p.lon, ge("poi-end-out"));
    setClickMode("start");
  }
});

function highlight(idx) {
  routeLayers.forEach(function(l, i) {
    l.setStyle({
      color: COLORS[i] || "#7c3aed",
      weight: i === idx ? 8 : 5,
      opacity: i === idx ? 1 : 0.35
    });
    // Bring selected route to front
    if (i === idx && l.bringToFront) l.bringToFront();
  });
  // Show label only for selected route
  routeLabels.forEach(function(lbl, i) {
    if (!lbl) return;
    var el = lbl.getElement && lbl.getElement();
    if (el) {
      var inner = el.querySelector("div");
      if (inner) inner.style.display = i === idx ? "block" : "none";
    }
  });
  document.querySelectorAll(".route-card").forEach(function(c, i) {
    c.style.border = i === idx ? "2px solid " + (COLORS[i] || "#333") : "1px solid #ddd";
    c.style.background = i === idx ? "#f8faff" : "white";
  });
  showTrafficPanel(idx);
}

var lastFeats = [];

function renderRoutes(data, fromHistory, skipFitBounds) {
  clearRoutes();
  lastMode = ge("mode-select").value;
  ge("used-out").textContent = data.used_routing || "-";
  lastFeats = data.features || [];
  if (!fromHistory) addTripToHistory(data);
  lastFeats.forEach(function(f, i) {
    var isMain = (i === 0);
    var l = L.geoJSON(f, {
      style: {
        color: COLORS[i] || "#7c3aed",
        weight: isMain ? 8 : 5,
        opacity: isMain ? 1 : 0.45
      }
    }).addTo(map);
    // Make alternative routes clickable to highlight them
    l.on("click", function(e) {
      L.DomEvent.stopPropagation(e);
      highlight(i);
    });
    routeLayers.push(l);
    var lbl = addRouteLabel(f, i, isMain);
    routeLabels.push(lbl);
  });
  var cards = lastFeats.map(function(f, i) {
    var p = f.properties || {};
    var used = p.summary && p.summary.used_routing === "poi";
    var tag = used ? "<span class='badge badge-poi'>POI</span>" : "<span class='badge badge-raw'>raw</span>";
    var instr = (p.instructions || []).map(function(x) { return "<li>" + x + "</li>"; }).join("");
    var dist = ((p.distance_m || 0) / 1000).toFixed(2);
    var durMin = Math.ceil(p.duration_min);
    var hrStr = durMin >= 60 ? Math.floor(durMin/60) + " hr " + (durMin%60) + " min" : durMin + " min";
    var isMain = (i === 0);
    var cardStyle = isMain
      ? "background:white;border:2px solid #2563eb;border-radius:12px;padding:12px 14px;margin-bottom:10px;cursor:pointer;box-shadow:0 2px 8px rgba(37,99,235,0.15);"
      : "background:white;border:1px solid #e5e7eb;border-radius:12px;padding:12px 14px;margin-bottom:10px;cursor:pointer;";
    return "<div class='route-card' data-i='" + i + "' style='" + cardStyle + "'>" +
      "<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;'>" +
        "<div style='display:flex;align-items:center;gap:8px;'>" +
          "<span style='display:inline-block;width:12px;height:12px;border-radius:50%;background:" + (COLORS[i]||"#888") + ";flex-shrink:0;'></span>" +
          "<span style='font-size:18px;font-weight:700;color:#111;'>" + hrStr + "</span>" +
          (isMain ? "<span style='font-size:11px;background:#dbeafe;color:#1d4ed8;padding:2px 7px;border-radius:10px;font-weight:600;'>Best</span>" : "") +
        "</div>" +
        tag +
      "</div>" +
      "<div style='font-size:13px;color:#6b7280;margin-left:20px;'>" + dist + " km</div>" +
      (instr ? "<details><summary style='font-size:12px;cursor:pointer;color:#2563eb;margin-top:6px;margin-left:20px;'>Turn-by-turn</summary>" +
      "<ol style='font-size:12px;margin:6px 0 0 18px;padding:0;color:#374151;'>" + instr + "</ol></details>" : "") +
    "</div>";
  }).join("");
  ge("routes-out").innerHTML = cards || "<div style='color:#6b7280'>No routes returned.</div>";
  document.querySelectorAll(".route-card").forEach(function(c) {
    c.addEventListener("click", function() { highlight(+c.dataset.i); });
  });
  if (!skipFitBounds) {
    try {
      var grp = L.featureGroup([startMarker, endMarker].concat(routeLayers).filter(Boolean));
      map.fitBounds(grp.getBounds().pad(0.15));
    } catch(e) {}
  }
  if (routeLayers.length) highlight(0);
}

ge("btn-route").onclick = async function() {
  hideErr();
  if (!startPoint || !endPoint) { showErr("Missing points", "Set both Start and End on the map first."); return; }
  clearRoutes();
  ge("routes-out").innerHTML = "<span class='spinner'></span> Calculating...";
  var body = JSON.stringify({
    start_lat: startPoint.lat,
    start_lon: startPoint.lon,
    end_lat: endPoint.lat,
    end_lon: endPoint.lon,
    mode: ge("mode-select").value,
    k: +ge("k-input").value,
    snap_to_poi: ge("snap-to-poi").checked,
    fallback_to_raw: ge("fallback-to-raw").checked,
    snap_radius_m: +ge("snap-radius").value
  });
  var res, data;
  try {
    res = await fetch(API + "/routes", {method: "POST", headers: {"Content-Type": "application/json"}, body: body});
    data = await res.json();
  } catch(e) {
    ge("routes-out").innerHTML = "";
    showErr("Network error", "Could not reach the server.");
    return;
  }
  if (!res.ok) {
    ge("routes-out").innerHTML = "";
    var msg = (data && data.error && data.error.message) ? data.error.message : JSON.stringify(data);
    showErr(res.status === 404 ? "No route found" : "Error " + res.status,
            res.status === 404 ? "Try moving points closer to a road, or enable fallback_to_raw." : msg);
    return;
  }
  renderRoutes(data);
};

// ── Traffic annotation ──────────────────────────────────────────────────────
var TRAFFIC_RATIOS = {red: 0.1, orange: 0.3, yellow: 0.5};
var selectedTrafficColor = null;
var selectedRouteIdx = 0;

function selectTraffic(color) {
  selectedTrafficColor = color;
  ["red","orange","yellow"].forEach(function(c) {
    ge("td-" + c).classList.toggle("active", c === color);
  });
  // Focus the matching input
  ge("ti-" + color).focus();
}

function clearTraffic() {
  ["red","orange","yellow"].forEach(function(c) {
    ge("ti-" + c).value = 0;
    ge("td-" + c).classList.remove("active");
  });
  selectedTrafficColor = null;
  ge("traffic-result").textContent = "";
  // Restore original label for selected route
  updateRouteLabel(selectedRouteIdx, false);
}

function applyTraffic() {
  var f = lastFeats[selectedRouteIdx];
  if (!f) return;
  var p = f.properties || {};
  var totalDist = p.distance_m || 0;
  var baseDurS = (p.duration_min || 0) * 60;

  // base speed = total_dist / base_dur_s (m/s)
  if (baseDurS <= 0) { ge("traffic-result").textContent = "No route data."; return; }
  var baseSpeedMs = totalDist / baseDurS;

  // For each traffic segment: extra time vs free-flow
  // free-flow time for that segment = len / baseSpeedMs
  // traffic time = len / (baseSpeedMs * ratio)
  // extra delay = free-flow_time * (1/ratio - 1)
  var extraS = 0;
  ["red","orange","yellow"].forEach(function(c) {
    var lenM = parseFloat(ge("ti-" + c).value) || 0;
    if (lenM <= 0) return;
    var ratio = TRAFFIC_RATIOS[c];
    var freeFlowS = lenM / baseSpeedMs;
    extraS += freeFlowS * (1.0 / ratio - 1.0);
  });

  var newDurS = baseDurS + extraS;
  var newDurMin = Math.ceil(newDurS / 60);
  var origMin = Math.ceil(p.duration_min);
  var dist = (totalDist / 1000).toFixed(2);

  ge("traffic-result").innerHTML =
    "<span style='color:#ef4444'>+" + Math.ceil(extraS/60) + " min delay</span> &nbsp; " +
    "<strong>" + newDurMin + " min</strong> total &nbsp; (" + dist + " km)";

  // Update the map label for this route
  updateRouteLabel(selectedRouteIdx, true, newDurMin, dist);
}

function updateRouteLabel(idx, withTraffic, newDurMin, dist) {
  var lbl = routeLabels[idx];
  if (!lbl) return;
  var el = lbl.getElement && lbl.getElement();
  if (!el) return;
  var inner = el.querySelector("div");
  if (!inner) return;
  var f = lastFeats[idx];
  if (!f) return;
  var p = f.properties || {};
  var modeIcon = (ge("mode-select").value === "motorcycle") ? "&#127949;" : "&#128663;";
  var displayDur = withTraffic ? newDurMin : Math.ceil(p.duration_min);
  var displayDist = dist || ((p.distance_m||0)/1000).toFixed(2);
  var trafficBadge = withTraffic ? "<div style='font-size:10px;color:#ef4444;padding-left:20px;'>&#128678; with traffic</div>" : "";
  inner.innerHTML =
    "<div style='display:flex;align-items:center;gap:5px;'>" +
    "<span style='font-size:14px;'>" + modeIcon + "</span>" +
    "<span style='font-size:14px;font-weight:700;color:#111;'>" + displayDur + " min</span>" +
    "</div>" +
    "<div style='font-size:11px;color:#666;margin-top:1px;padding-left:20px;'>" + displayDist + " km</div>" +
    trafficBadge;
}

function showTrafficPanel(routeIdx) {
  selectedRouteIdx = routeIdx;
  ge("traffic-section").style.display = "block";
  // Reset inputs when switching routes
  ["red","orange","yellow"].forEach(function(c) { ge("ti-" + c).value = 0; });
  ge("traffic-result").textContent = "";
  selectedTrafficColor = null;
  ["red","orange","yellow"].forEach(function(c) { ge("td-" + c).classList.remove("active"); });
}

async function loadBounds() {
  try {
    var r = await fetch(API + "/routes/bounds");
    if (!r.ok) return;
    var b = await r.json();
    boundaryBounds = b;
    ge("boundary-out").textContent = b.min_lat.toFixed(5) + ", " + b.min_lon.toFixed(5) + " to " + b.max_lat.toFixed(5) + ", " + b.max_lon.toFixed(5);
    var bnds = [[b.min_lat, b.min_lon], [b.max_lat, b.max_lon]];
    L.rectangle(bnds, {color: "#6b7280", weight: 2, fillOpacity: 0.04}).addTo(map);
    map.fitBounds(bnds, {padding: [20, 20]});
  } catch(e) {}
}
loadBounds();
</script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
