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
    RouteRequest, RouteResponse, LatLon,
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
        return svc.compute_routes(
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


@routing_router.get("/routes/bounds", response_model=BoundsResponse,
                    summary="Bounding box of road data")
def roads_bounds(svc: RoutingService = Depends(get_service)):
    return BoundsResponse(**svc.roads_bounds())


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
    svc: RoutingService = Depends(get_service),
):
    try:
        idx = svc.nearest_poi_index(lat, lon)
    except ValueError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(e))
    return NearestPoiResponse(
        query=LatLon(lat=lat, lon=lon),
        nearest_poi=svc.poi_payload(idx),
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
    #wrap{display:flex;height:100%;}
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
    .route-card{background:white;border:1px solid #ddd;border-radius:8px;padding:10px;margin-bottom:10px;cursor:pointer;}
    .route-title{font-weight:700;margin-bottom:6px;}
    .badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;margin-left:6px;}
    .badge-poi{background:#d1fae5;color:#065f46;}
    .badge-raw{background:#fee2e2;color:#991b1b;}
    #error-banner{display:none;background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;padding:10px 12px;margin-bottom:12px;color:#991b1b;font-size:13px;}
    .spinner{display:inline-block;width:14px;height:14px;border:2px solid #ccc;border-top-color:#333;border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:6px;}
    @keyframes spin{to{transform:rotate(360deg);}}
  </style>
</head>
<body>
<div id="wrap">
  <div id="map"></div>
  <div id="panel">
    <div id="error-banner"><strong id="err-title"></strong><div id="err-hint" style="margin-top:4px;"></div></div>
    <div class="section">
      <div class="hint">Click map to set Start and End, then click Route.</div>
      <button id="mode-start" class="active">&#128205; Set Start</button>
      <button id="mode-end">&#127937; Set End</button>
      <button id="btn-route">&#128506; Route</button>
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
var routeLayers = [], boundaryBounds = null, lastMode = null;
var COLORS = ["#2563eb", "#ef4444", "#16a34a"];

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

function clearRoutes() {
  routeLayers.forEach(function(l) { map.removeLayer(l); });
  routeLayers = [];
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
}
ge("btn-clear").onclick = clearAll;

function inBounds(p) {
  if (!boundaryBounds || !p) return true;
  return p.lat >= boundaryBounds.min_lat && p.lat <= boundaryBounds.max_lat &&
         p.lon >= boundaryBounds.min_lon && p.lon <= boundaryBounds.max_lon;
}

async function fetchPoi(lat, lon, el) {
  try {
    var r = await fetch(API + "/pois/nearest?lat=" + lat + "&lon=" + lon);
    if (r.ok) {
      var d = await r.json();
      el.textContent = d.nearest_poi ? JSON.stringify(d.nearest_poi, null, 2) : "Not found";
    } else { el.textContent = "Error"; }
  } catch(e) { el.textContent = "Error"; }
}

map.on("click", async function(e) {
  hideErr();
  var p = {lat: e.latlng.lat, lon: e.latlng.lng};
  if (!inBounds(p)) { showErr("Outside road data", "Click inside the grey boundary rectangle."); return; }
  if (clickMode === "start") {
    startPoint = p;
    if (startMarker) map.removeLayer(startMarker);
    startMarker = L.marker([p.lat, p.lon]).addTo(map).bindPopup("Start");
    ge("start-out").textContent = p.lat.toFixed(6) + ", " + p.lon.toFixed(6);
    fetchPoi(p.lat, p.lon, ge("poi-start-out"));
    setClickMode("end");
  } else {
    endPoint = p;
    if (endMarker) map.removeLayer(endMarker);
    endMarker = L.marker([p.lat, p.lon]).addTo(map).bindPopup("End");
    ge("end-out").textContent = p.lat.toFixed(6) + ", " + p.lon.toFixed(6);
    fetchPoi(p.lat, p.lon, ge("poi-end-out"));
    setClickMode("start");
  }
});

function highlight(idx) {
  routeLayers.forEach(function(l, i) {
    l.setStyle({color: COLORS[i] || "#7c3aed", weight: i === idx ? 10 : 6, opacity: i === idx ? 1 : 0.35});
  });
  document.querySelectorAll(".route-card").forEach(function(c, i) {
    c.style.border = i === idx ? "2px solid " + (COLORS[i] || "#333") : "1px solid #ddd";
  });
}

function renderRoutes(data) {
  clearRoutes();
  lastMode = ge("mode-select").value;
  ge("used-out").textContent = data.used_routing || "-";
  var feats = data.features || [];
  feats.forEach(function(f, i) {
    var l = L.geoJSON(f, {style: {color: COLORS[i] || "#7c3aed", weight: 6, opacity: 0.9}}).addTo(map);
    routeLayers.push(l);
  });
  var cards = feats.map(function(f, i) {
    var p = f.properties || {};
    var used = p.summary && p.summary.used_routing === "poi";
    var tag = used ? "<span class='badge badge-poi'>POI</span>" : "<span class='badge badge-raw'>raw</span>";
    var instr = (p.instructions || []).map(function(x) { return "<li>" + x + "</li>"; }).join("");
    var dist = ((p.distance_m || 0) / 1000).toFixed(2);
    return "<div class='route-card' data-i='" + i + "'>" +
           "<div class='route-title'><span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:" + (COLORS[i] || "#888") + ";margin-right:6px'></span>Route #" + p.rank + tag + "</div>" +
           "<div style='font-size:13px'><strong>" + dist + " km</strong> &middot; <strong>" + Math.ceil(p.duration_min) + " min</strong> &middot; " + p.mode + "</div>" +
           "<details><summary style='font-size:12px;cursor:pointer;color:#6b7280;margin-top:4px'>Turn-by-turn</summary>" +
           "<ol style='font-size:12px;margin:6px 0 0 18px;padding:0'>" + instr + "</ol></details></div>";
  }).join("");
  ge("routes-out").innerHTML = cards || "<div style='color:#6b7280'>No routes returned.</div>";
  document.querySelectorAll(".route-card").forEach(function(c) {
    c.addEventListener("click", function() { highlight(+c.dataset.i); });
  });
  try {
    var grp = L.featureGroup([startMarker, endMarker].concat(routeLayers).filter(Boolean));
    map.fitBounds(grp.getBounds().pad(0.15));
  } catch(e) {}
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
