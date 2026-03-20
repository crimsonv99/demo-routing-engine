# Demo Routing Engine — Valhalla Branch

A Vietnam road routing API powered by [Valhalla](https://github.com/valhalla/valhalla) top-K diverse route generation, a built-in Leaflet map UI, and a multi-engine benchmark desktop tool.

---

## Overview

This branch migrates the routing backend from a custom NetworkX graph to Valhalla, a production-grade open-source routing engine. The API exposes route computation, POI snapping, trip recording, and a browser-based map interface. A separate benchmark tool lets you compare Valhalla, GraphHopper, OSRM, and NetworkX side by side.

**Coverage:** Vietnam (bounding box `[8.18°N–23.39°N, 102.14°E–109.46°E]`)

---

## Features

- **Valhalla-backed routing** — delegates all path-finding to a local Valhalla instance via HTTP; supports `car` and `motorcycle` travel modes with tuned costing parameters
- **Top-K diverse routes** — returns up to 5 alternative routes ranked by diversity
- **POI snapping** — snaps origin/destination coordinates to the nearest point of interest from the GrabMap sample CSV before routing; falls back to raw coordinates if no POI is found within the configured radius
- **Trip history** — every route request is assigned a UUID and persisted so it can be retrieved later
- **Built-in map UI** — a Leaflet-powered HTML page served at `/api/v1/map` with a route form, mode selector, K slider, snap controls, and a trip history panel
- **Multi-engine benchmark tool** — a PyQt6 desktop app (`benchmark_tool/routing_tool.py`) to compare NetworkX, Valhalla, GraphHopper, and OSRM on any origin/destination pair, with wall-clock timing (`elapsed_ms`) per engine
- **Road network restriction system** — the benchmark tool supports loading restriction/road-closure data to test routing around blocked segments
- **OpenAPI docs** — auto-generated at `/docs` (Swagger) and `/redoc`

---

## Architecture
```
main.py                  FastAPI application & HTTP routers
config.py                Pydantic-settings config (reads .env)
schemas.py               Request / response models
poi_loader.py            Loads POI CSV into a GeoDataFrame with spatial index
services/
  routing_service.py     Business logic: POI snapping, K-route selection, trip storage
  valhalla_client.py     Async HTTP client for Valhalla's /route endpoint
benchmark_tool/
  routing_tool.py        PyQt6 desktop benchmark UI
  benchmark_engines/     Engine adapters (Valhalla, GraphHopper, OSRM, NetworkX)
  benchmark_ui/          UI components
  benchmark_assets/      Static assets for the benchmark app
  Restriction/           Road closure test data
trip_history/            Persisted trip JSON files (OSM way IDs per route)
data/                    POI CSV (VN Sample Data.csv)
graphhopper_data/        GraphHopper OSM PBF + config for Docker
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| Docker + Docker Compose | any recent version |
| Valhalla tiles | Pre-built Vietnam tiles placed in `valhalla_data/` |

---

## Quick Start

### 1. Start the routing engines
```bash
docker compose up -d
```

This starts:
- **Valhalla** on `http://localhost:8002`  
- **GraphHopper** on `http://localhost:8989` (loads `graphhopper_data/vietnam-latest.osm.pbf`)

### 2. Configure the application
```bash
cp .env.example .env
# Edit .env if needed
```

Key settings:

| Variable | Default | Description |
|---|---|---|
| `POIS_PATH` | `data/VN Sample Data.csv` | Path to POI CSV |
| `VALHALLA_URL` | `http://localhost:8002` | Valhalla server URL |
| `DEFAULT_K` | `3` | Default number of routes |
| `MAX_K` | `5` | Maximum routes per request |
| `DEFAULT_SNAP_RADIUS_M` | `20.0` | POI snap radius in metres |
| `ROUTE_TIMEOUT_S` | `8.0` | Per-route timeout |
| `API_PREFIX` | `/api/v1` | URL prefix for all endpoints |
| `DEBUG` | `false` | Enable debug mode |

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the API server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API is now available at `http://localhost:8000`.

---

## Docker Deployment

A multi-stage `Dockerfile` is provided for the API service:
```bash
docker build -t vn-routing-api .
docker run -p 8000:8000 --env-file .env vn-routing-api
```

The container runs as a non-root user and includes a health check (`/api/v1/health`).

For Heroku-style platforms, the `Procfile` is:
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## API Reference

All endpoints are prefixed with `/api/v1` by default.

### `POST /api/v1/routes`

Compute top-K diverse routes between two coordinates.

**Request body:**
```json
{
  "start_lat": 21.028,
  "start_lon": 105.834,
  "end_lat": 21.005,
  "end_lon": 105.845,
  "mode": "car",
  "k": 3,
  "snap_to_poi": true,
  "fallback_to_raw": true,
  "poi_candidates": 3,
  "snap_radius_m": 20.0
}
```

**Travel modes:** `car`, `motorcycle`

**Response:** GeoJSON FeatureCollection with ranked routes. Each feature includes `distance_m`, `duration_s`, `duration_min`, `mode`, turn-by-turn `instructions`, and a `summary` with snapping details. The response also includes a `trip_id` for later retrieval.

---

### `GET /api/v1/health`

Returns service health, segment count, and POI count.

### `GET /api/v1/health/graph`

Returns graph connectivity statistics.

### `GET /api/v1/routes/bounds`

Returns the geographic bounding box of the road data (Vietnam).

### `GET /api/v1/trips/{trip_id}`

Retrieve a previously recorded trip by its UUID.

### `GET /api/v1/pois/nearest`

Find the nearest POI to a given latitude/longitude, with distance and snap-radius status.

**Query params:** `lat`, `lon`, `snap_radius_m` (optional)

### `GET /api/v1/map`

Serves the built-in Leaflet map UI (HTML, no redirect).

---

## Map UI

Open `http://localhost:8000/api/v1/map` in a browser. The map lets you:

- Click to set start and end points
- Choose travel mode (`car` / `motorcycle`) and number of routes (`k`)
- Toggle `snap_to_poi` and `fallback_to_raw`
- View colour-coded alternative routes with turn-by-turn instructions
- Browse trip history

---

## Benchmark Tool

A PyQt6 desktop application for comparing routing engines side by side.

### Install extra dependencies
```bash
pip install PyQt6 PyQt6-WebEngine polyline httpx
```

### Run
```bash
python benchmark_tool/routing_tool.py
```

### Supported engines

| Engine | Transport | Notes |
|---|---|---|
| **Valhalla** | Docker `localhost:8002` | Primary production engine |
| **GraphHopper** | Docker `localhost:8989` | Full feature parity with Valhalla |
| **OSRM** | Docker `localhost:5000` | Fastest cold-start |
| **NetworkX** | In-process | Requires road network file, ~14 min build |

Each engine result includes wall-clock response time (`elapsed_ms`), distance, duration, and turn-by-turn instructions rendered on a shared Leaflet map. Road closure/restriction files can be loaded from `benchmark_tool/Restriction/Road closure/` to test avoidance behaviour.

---

## Data

- **POI data:** `data/VN Sample Data.csv` — GrabMap sample POIs for Vietnam. Loaded into a GeoDataFrame with an R-tree spatial index (EPSG:3857) for fast nearest-neighbour queries.
- **Road network (GraphHopper):** Place `vietnam-latest.osm.pbf` in `graphhopper_data/` before starting the GraphHopper container.
- **Valhalla tiles:** Place pre-built tiles in `valhalla_data/` before starting the Valhalla container.
- **Trip history:** Saved as JSON files in `trip_history/`, keyed by timestamp and mode (e.g. `osm_ways_2026-03-19_11-50-39_car.json`).

---

## Tech Stack

| Layer | Library |
|---|---|
| Web framework | FastAPI 0.111, Uvicorn |
| Data validation | Pydantic v2, pydantic-settings |
| HTTP client | httpx (async) |
| Geospatial | GeoPandas, Shapely, PyProj, rtree |
| Numerics | NumPy, Pandas |
| Map UI | Leaflet (CDN, embedded HTML) |
| Benchmark UI | PyQt6, PyQt6-WebEngine |
| Routing engine | Valhalla (Docker), GraphHopper (Docker) |
| Containerisation | Docker, Docker Compose |

---

## Configuration Reference (`.env`)
```dotenv
POIS_PATH=data/VN Sample Data.csv
VALHALLA_URL=http://localhost:8002

DEFAULT_K=3
MAX_K=5
ROUTE_TIMEOUT_S=8.0
DIVERSITY_THRESHOLD=0.20
DEFAULT_SNAP_RADIUS_M=20.0
MAX_SNAP_RADIUS_M=200.0
MAX_POI_CANDIDATES=10

API_TITLE=VN Routing API
API_VERSION=1.0.0
API_PREFIX=/api/v1
DEBUG=false
```

---

## License

This is a demo project. Data files (GrabMap sample data) are for demonstration purposes only.
