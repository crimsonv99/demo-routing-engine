"""
OSM Highway Downloader — Desktop GUI
Run with: python osm_highway_downloader_gui.py
Requires: pip install requests geopandas shapely
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
from pathlib import Path

import requests
import geopandas as gpd
from shapely.geometry import LineString, box
from shapely.ops import unary_union

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

ALL_HIGHWAY_TYPES = [
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "residential", "living_street",
    "unclassified", "road",
]

DEFAULT_HIGHWAY_TYPES = set(ALL_HIGHWAY_TYPES)

OSM_TAGS = ["highway", "name", "oneway", "lanes", "maxspeed", "motorcycle", "motorcar", "layer"]

MIN_TILE_SIZE   = 0.01
BACKOFF_BASE    = 5       # seconds to wait after a 429 before trying next mirror
MAX_WORKERS     = 2       # parallel Overpass requests (respect their ToS)
REQUEST_DELAY   = 1.5     # minimum seconds between any two requests (global throttle)

# Global rate-limit gate: when one worker hits 429, all workers pause together
_rate_limit_lock  = threading.Lock()
_rate_limit_until = [0.0]   # epoch seconds; workers sleep until this time

# Global request throttle: enforce minimum gap between requests across all workers
_throttle_lock      = threading.Lock()
_last_request_time  = [0.0]

# Stop signal: set by the GUI Stop button; workers check this and exit early
_stop_event = threading.Event()

def _throttle():
    """Block until at least REQUEST_DELAY seconds have passed since the last request."""
    with _throttle_lock:
        now  = time.time()
        wait = _last_request_time[0] + REQUEST_DELAY - now
        if wait > 0:
            time.sleep(wait)
        _last_request_time[0] = time.time()


# ──────────────────────────────────────────────
# Core download logic (same as CLI tool)
# ──────────────────────────────────────────────

def build_query(bbox, timeout=180):
    minlat, minlon, maxlat, maxlon = bbox
    return (
        f"[out:json][timeout:{timeout}];\n"
        f"(\n"
        f"  way[highway]({minlat},{minlon},{maxlat},{maxlon});\n"
        f"  relation[type=restriction]({minlat},{minlon},{maxlat},{maxlon});\n"
        f");\n"
        f"out body;\n(._;>;);\nout skel qt;\n"
    )


def fetch_osm(bbox, log_fn=None):
    """
    Try each mirror once. On any failure → return None so the caller splits the tile.
    On 429 → wait briefly then try the next mirror (don't retry the same one).
    """
    query = build_query(bbox)

    # Honour global rate-limit gate set by another worker
    now = time.time()
    if _rate_limit_until[0] > now:
        time.sleep(_rate_limit_until[0] - now)

    for url in OVERPASS_MIRRORS:
        if _stop_event.is_set():
            return []
        try:
            _throttle()
            r = requests.get(url, params={"data": query}, timeout=200)

            if r.status_code == 429:
                if log_fn: log_fn(f"  Rate limited by {url}, pausing {BACKOFF_BASE}s…", "warn")
                with _rate_limit_lock:
                    _rate_limit_until[0] = max(_rate_limit_until[0], time.time() + BACKOFF_BASE)
                time.sleep(BACKOFF_BASE)
                continue  # try next mirror

            if r.status_code in (502, 503, 504):
                if log_fn: log_fn(f"  Server {r.status_code} from {url}, trying next mirror…", "warn")
                continue

            r.raise_for_status()

            try:
                data = r.json()
            except ValueError:
                if log_fn: log_fn(f"  {url} returned non-JSON body", "warn")
                continue

            remark = data.get("remark", "")
            if remark and any(kw in remark.lower() for kw in
                              ("runtime error", "out of memory", "timed out", "maxsize")):
                if log_fn: log_fn(f"  Overpass limit — will split tile", "warn")
                return None

            return data.get("elements", [])

        except requests.Timeout:
            if log_fn: log_fn(f"  Timeout on {url}, trying next mirror…", "warn")
        except Exception as e:
            # Trim urllib3 errors that embed the full encoded URL in the message
            msg = str(e)
            if "Caused by" in msg:
                msg = msg.split("Caused by")[-1].strip(" ()")
            if log_fn: log_fn(f"  {url} → {msg[:120]}", "warn")

    if log_fn: log_fn(f"  All mirrors failed — splitting tile into smaller pieces", "warn")
    return None  # all mirrors failed → caller splits the tile


def split_bbox(bbox):
    minlat, minlon, maxlat, maxlon = bbox
    midlat = (minlat + maxlat) / 2
    midlon = (minlon + maxlon) / 2
    return [
        (minlat, minlon, midlat, midlon),
        (minlat, midlon, midlat, maxlon),
        (midlat, minlon, maxlat, midlon),
        (midlat, midlon, maxlat, maxlon),
    ]


def crawl_bbox(bbox, log_fn=None, failed_out=None):
    """
    Recursively fetch OSM elements for a bbox.
    Splits the tile on failure (timeout / OOM) — NOT on empty results.
    Empty = no roads there (ocean, forest) → accepted as-is.
    failed_out: optional list; failed leaf tiles are appended to it.
    """
    queue = [bbox]
    all_elements = []
    while queue:
        if _stop_event.is_set():
            break
        current = queue.pop(0)
        elements = fetch_osm(current, log_fn)

        if elements is None:
            # Request failed — split if the tile is still large enough
            minlat, minlon, maxlat, maxlon = current
            lat_sz = maxlat - minlat
            lon_sz = maxlon - minlon
            if lat_sz > MIN_TILE_SIZE * 2 and lon_sz > MIN_TILE_SIZE * 2:
                if log_fn: log_fn(
                    f"  ↔ Splitting ({minlat:.3f},{minlon:.3f})→({maxlat:.3f},{maxlon:.3f})", "warn")
                queue.extend(split_bbox(current))
            else:
                if log_fn: log_fn(
                    f"  ✗ Giving up on small tile ({minlat:.4f},{minlon:.4f})→"
                    f"({maxlat:.4f},{maxlon:.4f})", "error")
                if failed_out is not None:
                    failed_out.append(current)
        else:
            # elements == 0 just means no roads in that area — that's fine
            all_elements.extend(elements)

    return all_elements


def elements_to_gdf(elements, highway_types, log_fn=None):
    if log_fn: log_fn("Converting to GeoDataFrame…", "info")
    nodes = {el["id"]: el for el in elements if el["type"] == "node"}
    rows = []
    for el in elements:
        if el["type"] != "way":
            continue
        tags = el.get("tags", {})
        if "highway" not in tags:
            continue
        coords = [
            (nodes[nid]["lon"], nodes[nid]["lat"])
            for nid in el.get("nodes", []) if nid in nodes
        ]
        if len(coords) < 2:
            continue
        props = {"osm_id": el["id"]}
        for tag in OSM_TAGS:
            props[tag] = tags.get(tag)
        props["last_editor"] = el.get("user")
        props["last_edit_timestamp"] = el.get("timestamp")
        props["usage"] = "Yes" if props["highway"] in highway_types else "No"
        props["geometry"] = LineString(coords)
        rows.append(props)
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    if log_fn: log_fn(f"Total features: {len(gdf)}", "ok")
    return gdf


def parse_restrictions(elements, log_fn=None):
    """
    Extract turn restriction relations from OSM elements.
    Handles all via types: via=node, via=way, multiple via nodes.
    Returns a list of restriction dicts to save as JSON.
    """
    restrictions = []
    skipped = 0

    for el in elements:
        if el.get("type") != "relation":
            continue
        tags = el.get("tags", {})
        if tags.get("type") != "restriction":
            continue

        # Get restriction type — also handle vehicle-specific and conditional
        rtype = (tags.get("restriction") or
                 tags.get("restriction:motorcar") or
                 tags.get("restriction:conditional", "").split("@")[0].strip())
        if not rtype:
            skipped += 1
            continue

        from_ways, via_nodes, via_ways, to_ways = [], [], [], []
        for member in el.get("members", []):
            role  = member.get("role", "")
            mtype = member.get("type", "")
            ref   = member.get("ref")
            if ref is None:
                continue
            if role == "from" and mtype == "way":
                from_ways.append(ref)
            elif role == "via" and mtype == "node":
                via_nodes.append(ref)
            elif role == "via" and mtype == "way":
                via_ways.append(ref)
            elif role == "to" and mtype == "way":
                to_ways.append(ref)

        # Must have from + to + at least one via
        if not from_ways or not to_ways or (not via_nodes and not via_ways):
            skipped += 1
            continue

        restrictions.append({
            "osm_id":      el["id"],
            "restriction": rtype,
            "from_ways":   from_ways,
            "via_nodes":   via_nodes,
            "via_ways":    via_ways,
            "to_ways":     to_ways,
        })

    if log_fn:
        log_fn(f"Turn restrictions: {len(restrictions)} parsed, {skipped} skipped", "info")
    return restrictions


def run_download(config, log_fn, progress_fn, done_fn):
    """Main download job — runs in a background thread."""
    _stop_event.clear()           # reset stop signal
    _rate_limit_until[0] = 0.0   # reset rate-limit gate
    _last_request_time[0] = 0.0  # reset throttle
    global REQUEST_DELAY
    REQUEST_DELAY = config.get("request_delay", REQUEST_DELAY)
    log_fn(f"Request delay: {REQUEST_DELAY}s between requests", "info")
    try:
        input_mode   = config["input_mode"]
        output_path  = config["output_path"]
        output_fmt   = config["output_fmt"]
        layer_name   = config["layer_name"]
        highway_types = config["highway_types"]
        tile_size    = config["tile_size"]

        # ── Resolve boundary ──
        if input_mode == "geojson":
            log_fn(f"Loading boundary: {config['geojson_path']}", "info")
            gdf_boundary = gpd.read_file(config["geojson_path"])
            boundary = unary_union(gdf_boundary.geometry)
            minx, miny, maxx, maxy = boundary.bounds
        else:
            minlon = config["minlon"]; minlat = config["minlat"]
            maxlon = config["maxlon"]; maxlat = config["maxlat"]
            minx, miny, maxx, maxy = minlon, minlat, maxlon, maxlat
            boundary = box(minx, miny, maxx, maxy)
            log_fn(f"Bbox: ({minx},{miny}) → ({maxx},{maxy})", "info")

        # ── Build tile list ──
        tiles = []
        lat = miny
        while lat < maxy:
            lon = minx
            while lon < maxx:
                tb = box(lon, lat, min(lon + tile_size, maxx), min(lat + tile_size, maxy))
                if boundary.intersects(tb):
                    tiles.append(tb.bounds)
                lon += tile_size
            lat += tile_size

        log_fn(f"Processing {len(tiles)} tile(s)…", "info")

        # ── Crawl tiles (parallel, MAX_WORKERS concurrent requests) ──
        all_elements = []
        failed_tiles = []
        completed_count = [0]
        lock = threading.Lock()

        def download_tile(idx, tile):
            tminx, tminy, tmaxx, tmaxy = tile
            tag = f"Tile {idx}/{len(tiles)} ({tminx:.3f},{tminy:.3f})→({tmaxx:.3f},{tmaxy:.3f})"
            log_fn(tag, "")
            local_failed = []
            elems = crawl_bbox((tminy, tminx, tmaxy, tmaxx), log_fn, local_failed)
            with lock:
                all_elements.extend(elems)
                failed_tiles.extend(local_failed)
                completed_count[0] += 1
                progress_fn(int(completed_count[0] / len(tiles) * 80))
            status = "ok" if elems else ""
            log_fn(f"  → {len(elems)} elements {'✓' if elems else '(empty)'}", status)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(download_tile, i, tile): i
                for i, tile in enumerate(tiles, 1)
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    log_fn(f"  Worker error: {e}", "error")

        if _stop_event.is_set():
            log_fn("⏹ Download stopped by user.", "warn")
            done_fn(False)
            return

        if failed_tiles:
            log_fn(f"⚠ {len(failed_tiles)} tile(s) permanently failed — data may have gaps:", "warn")
            for ft in failed_tiles:
                log_fn(f"  · {ft}", "warn")

        # ── Dedup ──
        seen, unique = set(), []
        for el in all_elements:
            k = (el["type"], el["id"])
            if k not in seen:
                seen.add(k); unique.append(el)
        log_fn(f"Unique elements: {len(unique)}", "info")
        progress_fn(85)

        # ── Convert ──
        gdf = elements_to_gdf(unique, highway_types, log_fn)
        progress_fn(92)

        if gdf.empty:
            log_fn("No highway features found.", "warn")
            done_fn(False)
            return

        # ── Save roads ──
        log_fn(f"Saving → {output_path}", "info")
        out = Path(output_path)
        if output_fmt == "gpkg":
            gdf.to_file(out, layer=layer_name, driver="GPKG")
        elif output_fmt == "geojson":
            gdf.to_file(out, driver="GeoJSON")
        elif output_fmt == "shp":
            gdf.to_file(out, driver="ESRI Shapefile")

        # ── Save turn restrictions as JSON alongside road file ──
        import json
        restrictions = parse_restrictions(unique, log_fn)
        restrictions_path = out.with_name(out.stem + "_restrictions.json")
        with open(restrictions_path, "w", encoding="utf-8") as f:
            json.dump(restrictions, f, ensure_ascii=False, indent=2)
        log_fn(f"✓ Restrictions saved: {restrictions_path} ({len(restrictions)} rules)", "ok")

        progress_fn(100)
        log_fn(f"✓ Done! Saved to: {output_path}", "ok")
        done_fn(True)

    except Exception as e:
        log_fn(f"✗ Error: {e}", "error")
        done_fn(False)


# ──────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OSM Highway Downloader")
        self.resizable(True, True)
        self.minsize(660, 680)

        # ── Theme colours ──
        self.bg      = "#0f1113"
        self.surface = "#181b1f"
        self.surf2   = "#1f2328"
        self.border  = "#2d3139"
        self.fg      = "#c9d1d9"
        self.fg_dim  = "#6e7681"
        self.accent  = "#4493f8"
        self.green   = "#3fb950"
        self.orange  = "#d29922"
        self.red     = "#f85149"

        self.configure(bg=self.bg)
        self._build_styles()
        self._build_ui()

        # State
        self.hw_vars = {}
        self._build_hw_checkboxes()
        self._update_output_hint()

    # ── Styles ──────────────────────────────────
    def _build_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(".",
            background=self.bg,
            foreground=self.fg,
            fieldbackground=self.surf2,
            troughcolor=self.surf2,
            bordercolor=self.border,
            darkcolor=self.border,
            lightcolor=self.border,
            insertcolor=self.fg,
            font=("Courier New", 10),
        )
        style.configure("TFrame",  background=self.bg)
        style.configure("Card.TFrame", background=self.surface, relief="flat")
        style.configure("TLabel",  background=self.bg, foreground=self.fg, font=("Courier New", 10))
        style.configure("Dim.TLabel", background=self.bg, foreground=self.fg_dim, font=("Courier New", 9))
        style.configure("Head.TLabel", background=self.surface, foreground=self.accent,
                         font=("Courier New", 9, "bold"))
        style.configure("TEntry",
            fieldbackground=self.surf2, foreground=self.fg,
            insertcolor=self.fg, borderwidth=1, relief="flat",
            padding=(6, 4),
        )
        style.map("TEntry", bordercolor=[("focus", self.accent)])
        style.configure("TCombobox",
            fieldbackground=self.surf2, foreground=self.fg,
            background=self.surf2, arrowcolor=self.fg_dim,
            borderwidth=1, relief="flat",
        )
        style.configure("Accent.TButton",
            background=self.accent, foreground="#000000",
            font=("Courier New", 10, "bold"),
            relief="flat", borderwidth=0, padding=(10, 6),
        )
        style.map("Accent.TButton",
            background=[("active", "#6ab4ff"), ("disabled", self.surf2)],
            foreground=[("disabled", self.fg_dim)],
        )
        style.configure("TCheckbutton",
            background=self.surface, foreground=self.fg,
            font=("Courier New", 9),
        )
        style.map("TCheckbutton",
            background=[("active", self.surface)],
            foreground=[("active", self.accent)],
        )
        style.configure("TProgressbar",
            troughcolor=self.surf2, background=self.accent,
            borderwidth=0, thickness=6,
        )
        style.configure("TNotebook",
            background=self.bg, tabmargins=0,
        )
        style.configure("TNotebook.Tab",
            background=self.surf2, foreground=self.fg_dim,
            font=("Courier New", 9), padding=(10, 4),
        )
        style.map("TNotebook.Tab",
            background=[("selected", self.surface)],
            foreground=[("selected", self.accent)],
        )
        style.configure("TScale",
            background=self.bg, troughcolor=self.surf2,
            sliderrelief="flat",
        )

    # ── UI Layout ────────────────────────────────
    def _build_ui(self):
        root = ttk.Frame(self, padding=16)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)

        # Title bar
        title_row = ttk.Frame(root)
        title_row.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        tk.Label(title_row, text="OSM HIGHWAY DOWNLOADER",
                 bg=self.bg, fg=self.accent,
                 font=("Courier New", 13, "bold")).pack(side="left")
        tk.Label(title_row, text="  //  OpenStreetMap → GeoPackage / GeoJSON / SHP",
                 bg=self.bg, fg=self.fg_dim,
                 font=("Courier New", 9)).pack(side="left", pady=(3, 0))

        row = 1

        # ── 1. Boundary input ──
        row = self._section(root, row, "① BOUNDARY INPUT")
        boundary_frame = self._card(root, row); row += 1

        self.input_mode = tk.StringVar(value="geojson")

        nb = ttk.Notebook(boundary_frame)
        nb.pack(fill="x", padx=10, pady=10)
        nb.bind("<<NotebookTabChanged>>", lambda e: self._on_tab_change(nb))

        # GeoJSON tab
        tab_geo = ttk.Frame(nb, padding=8)
        nb.add(tab_geo, text="  GeoJSON File  ")

        geo_row = ttk.Frame(tab_geo)
        geo_row.pack(fill="x")
        ttk.Label(geo_row, text="File path:", style="Dim.TLabel").pack(side="left")

        self.geojson_var = tk.StringVar()
        geo_entry = ttk.Entry(geo_row, textvariable=self.geojson_var, width=50)
        geo_entry.pack(side="left", padx=(6, 6), fill="x", expand=True)

        ttk.Button(geo_row, text="Browse…",
                   command=self._browse_input,
                   style="Accent.TButton").pack(side="left")

        # BBox tab
        tab_bbox = ttk.Frame(nb, padding=8)
        nb.add(tab_bbox, text="  Bounding Box  ")

        bbox_grid = ttk.Frame(tab_bbox)
        bbox_grid.pack(fill="x")

        self.bbox_vars = {}
        fields = [
            ("MIN LON (west)",  "minlon", 0, 0),
            ("MIN LAT (south)", "minlat", 0, 2),
            ("MAX LON (east)",  "maxlon", 1, 0),
            ("MAX LAT (north)", "maxlat", 1, 2),
        ]
        for label, key, r, c in fields:
            ttk.Label(bbox_grid, text=label, style="Dim.TLabel").grid(
                row=r, column=c, sticky="w", padx=(0 if c == 0 else 14, 4), pady=3)
            v = tk.StringVar()
            self.bbox_vars[key] = v
            ttk.Entry(bbox_grid, textvariable=v, width=14).grid(
                row=r, column=c + 1, sticky="ew", padx=(0, 4), pady=3)

        # ── 2. Output ──
        row = self._section(root, row, "② OUTPUT"); 
        out_card = self._card(root, row); row += 1

        # Output path row
        out_path_row = ttk.Frame(out_card)
        out_path_row.pack(fill="x", padx=10, pady=(10, 4))
        ttk.Label(out_path_row, text="Output path:", style="Dim.TLabel").pack(side="left")

        self.output_var = tk.StringVar()
        out_entry = ttk.Entry(out_path_row, textvariable=self.output_var, width=50)
        out_entry.pack(side="left", padx=(6, 6), fill="x", expand=True)
        self.output_var.trace_add("write", lambda *_: self._auto_detect_format())

        ttk.Button(out_path_row, text="Save as…",
                   command=self._browse_output,
                   style="Accent.TButton").pack(side="left")

        # Format + layer
        fmt_row = ttk.Frame(out_card)
        fmt_row.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Label(fmt_row, text="Format:", style="Dim.TLabel").pack(side="left")
        self.fmt_var = tk.StringVar(value="gpkg")
        fmt_combo = ttk.Combobox(fmt_row, textvariable=self.fmt_var, width=18,
                                  values=["gpkg", "geojson", "shp"], state="readonly")
        fmt_combo.pack(side="left", padx=(6, 20))
        fmt_combo.bind("<<ComboboxSelected>>", lambda e: self._update_output_hint())

        ttk.Label(fmt_row, text="Layer name:", style="Dim.TLabel").pack(side="left")
        self.layer_var = tk.StringVar(value="highways")
        ttk.Entry(fmt_row, textvariable=self.layer_var, width=16).pack(side="left", padx=(6, 0))

        self.fmt_hint = tk.Label(fmt_row, text="", bg=self.surface, fg=self.fg_dim,
                                  font=("Courier New", 8))
        self.fmt_hint.pack(side="left", padx=(10, 0))

        # ── 3. Highway types ──
        row = self._section(root, row, "③ HIGHWAY TYPES")
        self.hw_card = self._card(root, row); row += 1

        self.hw_inner = ttk.Frame(self.hw_card, padding=(10, 8, 10, 4))
        self.hw_inner.pack(fill="x")

        # Select all / none / default buttons
        hw_btn_row = ttk.Frame(self.hw_card)
        hw_btn_row.pack(fill="x", padx=10, pady=(0, 8))
        for label, cmd in [("Select all", self._hw_all),
                             ("Clear", self._hw_none),
                             ("Default", self._hw_default)]:
            tk.Button(hw_btn_row, text=label,
                      bg=self.surf2, fg=self.accent,
                      font=("Courier New", 8), relief="flat",
                      activebackground=self.surf2, activeforeground=self.fg,
                      cursor="hand2", command=cmd,
                      padx=8, pady=2).pack(side="left", padx=(0, 6))

        # ── 4. Options ──
        row = self._section(root, row, "④ OPTIONS")
        opt_card = self._card(root, row); row += 1

        opt_inner = ttk.Frame(opt_card, padding=(10, 8))
        opt_inner.pack(fill="x")

        # Tile size
        tile_row = ttk.Frame(opt_inner)
        tile_row.pack(fill="x", pady=(0, 6))
        ttk.Label(tile_row, text="Tile size (°):", style="Dim.TLabel").pack(side="left")
        self.tile_var = tk.DoubleVar(value=0.5)
        tile_lbl = tk.Label(tile_row, text="0.5°",
                             bg=self.bg, fg=self.accent,
                             font=("Courier New", 9, "bold"), width=5)
        tile_lbl.pack(side="right")
        tile_scale = ttk.Scale(tile_row, from_=0.1, to=1.0,
                                variable=self.tile_var, orient="horizontal",
                                command=lambda v: tile_lbl.config(text=f"{float(v):.1f}°"))
        tile_scale.pack(side="left", fill="x", expand=True, padx=(8, 8))

        # Request delay
        delay_row = ttk.Frame(opt_inner)
        delay_row.pack(fill="x", pady=(0, 6))
        ttk.Label(delay_row, text="Request delay (s):", style="Dim.TLabel").pack(side="left")
        self.delay_var = tk.DoubleVar(value=REQUEST_DELAY)
        delay_lbl = tk.Label(delay_row, text=f"{REQUEST_DELAY:.1f}s",
                             bg=self.bg, fg=self.accent,
                             font=("Courier New", 9, "bold"), width=5)
        delay_lbl.pack(side="right")
        delay_scale = ttk.Scale(delay_row, from_=0.5, to=5.0,
                                variable=self.delay_var, orient="horizontal",
                                command=lambda v: delay_lbl.config(text=f"{float(v):.1f}s"))
        delay_scale.pack(side="left", fill="x", expand=True, padx=(8, 8))

        # Verbose
        self.verbose_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_inner, text="Verbose logging",
                         variable=self.verbose_var).pack(anchor="w")

        # ── 5. Run ──
        run_card = self._card(root, row); row += 1

        btn_row = ttk.Frame(run_card)
        btn_row.pack(fill="x", padx=10, pady=(10, 6))

        self.run_btn = ttk.Button(btn_row, text="▶  START DOWNLOAD",
                                   style="Accent.TButton",
                                   command=self._start)
        self.run_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.stop_btn = ttk.Button(btn_row, text="⏹  STOP",
                                    style="Accent.TButton",
                                    command=self._stop,
                                    state="disabled")
        self.stop_btn.pack(side="left")

        self.progress = ttk.Progressbar(run_card, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=10, pady=(0, 6))

        # Log
        log_frame = ttk.Frame(run_card, padding=(10, 0, 10, 10))
        log_frame.pack(fill="both", expand=True)

        self.log_box = tk.Text(
            log_frame, height=10,
            bg=self.surf2, fg=self.fg,
            font=("Courier New", 9),
            relief="flat", bd=0,
            wrap="word",
            state="disabled",
            insertbackground=self.fg,
        )
        self.log_box.pack(side="left", fill="both", expand=True)
        self.log_box.tag_config("info",  foreground=self.accent)
        self.log_box.tag_config("ok",    foreground=self.green)
        self.log_box.tag_config("warn",  foreground=self.orange)
        self.log_box.tag_config("error", foreground=self.red)

        scrollbar = ttk.Scrollbar(log_frame, command=self.log_box.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_box.configure(yscrollcommand=scrollbar.set)

        self._log("Ready. Configure parameters and click Start Download.", "info")

    # ── Helper widgets ───────────────────────────
    def _section(self, parent, row, text):
        f = ttk.Frame(parent)
        f.grid(row=row, column=0, sticky="ew", pady=(10, 2))
        tk.Label(f, text=text,
                 bg=self.bg, fg=self.fg_dim,
                 font=("Courier New", 8, "bold")).pack(side="left")
        tk.Frame(f, bg=self.border, height=1).pack(side="left", fill="x", expand=True, padx=(8, 0))
        return row + 1

    def _card(self, parent, row):
        f = tk.Frame(parent, bg=self.surface,
                     highlightbackground=self.border,
                     highlightthickness=1)
        f.grid(row=row, column=0, sticky="ew", pady=(0, 2))
        parent.rowconfigure(row, weight=0)
        return f

    # ── Highway checkboxes ───────────────────────
    def _build_hw_checkboxes(self):
        for w in self.hw_inner.winfo_children():
            w.destroy()
        self.hw_vars = {}

        cols = 4
        for i, hw in enumerate(ALL_HIGHWAY_TYPES):
            var = tk.BooleanVar(value=hw in DEFAULT_HIGHWAY_TYPES)
            self.hw_vars[hw] = var
            cb = ttk.Checkbutton(self.hw_inner, text=hw, variable=var)
            cb.grid(row=i // cols, column=i % cols, sticky="w", padx=4, pady=1)

    def _hw_all(self):
        for v in self.hw_vars.values(): v.set(True)

    def _hw_none(self):
        for v in self.hw_vars.values(): v.set(False)

    def _hw_default(self):
        for k, v in self.hw_vars.items():
            v.set(k in DEFAULT_HIGHWAY_TYPES)

    # ── File dialogs ─────────────────────────────
    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select boundary GeoJSON",
            filetypes=[("GeoJSON", "*.geojson *.json"), ("All files", "*.*")]
        )
        if path:
            self.geojson_var.set(path)

    def _browse_output(self):
        fmt = self.fmt_var.get()
        ext_map = {"gpkg": ".gpkg", "geojson": ".geojson", "shp": ".shp"}
        ft_map  = {"gpkg": [("GeoPackage", "*.gpkg")],
                   "geojson": [("GeoJSON", "*.geojson")],
                   "shp": [("Shapefile", "*.shp")]}
        path = filedialog.asksaveasfilename(
            title="Save output as",
            defaultextension=ext_map.get(fmt, ".gpkg"),
            filetypes=ft_map.get(fmt, [("All files", "*.*")])
        )
        if path:
            self.output_var.set(path)

    # ── Auto detect format ───────────────────────
    def _auto_detect_format(self):
        p = self.output_var.get()
        ext = Path(p).suffix.lower()
        if ext == ".gpkg": self.fmt_var.set("gpkg")
        elif ext in (".geojson", ".json"): self.fmt_var.set("geojson")
        elif ext == ".shp": self.fmt_var.set("shp")
        self._update_output_hint()

    def _update_output_hint(self):
        hints = {
            "gpkg": "(recommended, supports layers)",
            "geojson": "(single layer, human-readable)",
            "shp": "(legacy, splits large files)",
        }
        self.fmt_hint.config(text=hints.get(self.fmt_var.get(), ""))

    def _on_tab_change(self, nb):
        idx = nb.index(nb.select())
        self.input_mode.set("geojson" if idx == 0 else "bbox")

    # ── Logging ──────────────────────────────────
    def _log(self, msg, tag=""):
        def _do():
            ts = datetime.now().strftime("%H:%M:%S")
            self.log_box.configure(state="normal")
            self.log_box.insert("end", f"[{ts}] {msg}\n", tag)
            self.log_box.configure(state="disabled")
            self.log_box.see("end")
        self.after(0, _do)

    def _set_progress(self, val):
        self.after(0, lambda: self.progress.configure(value=val))

    # ── Start ────────────────────────────────────
    def _start(self):
        # Validate
        output = self.output_var.get().strip()
        if not output:
            messagebox.showerror("Missing output", "Please set an output file path.")
            return

        hw_types = {k for k, v in self.hw_vars.items() if v.get()}
        if not hw_types:
            messagebox.showerror("No highway types", "Select at least one highway type.")
            return

        mode = self.input_mode.get()
        if mode == "geojson":
            gjpath = self.geojson_var.get().strip()
            if not gjpath:
                messagebox.showerror("Missing input", "Please select a GeoJSON boundary file.")
                return
        else:
            for k in ("minlon", "minlat", "maxlon", "maxlat"):
                if not self.bbox_vars[k].get().strip():
                    messagebox.showerror("Missing bbox", f"Please fill in {k.upper()}.")
                    return

        config = {
            "input_mode":   mode,
            "geojson_path": self.geojson_var.get().strip() if mode == "geojson" else "",
            "minlon": float(self.bbox_vars["minlon"].get()) if mode == "bbox" else 0,
            "minlat": float(self.bbox_vars["minlat"].get()) if mode == "bbox" else 0,
            "maxlon": float(self.bbox_vars["maxlon"].get()) if mode == "bbox" else 0,
            "maxlat": float(self.bbox_vars["maxlat"].get()) if mode == "bbox" else 0,
            "output_path":  output,
            "output_fmt":   self.fmt_var.get(),
            "layer_name":   self.layer_var.get().strip() or "highways",
            "highway_types":   hw_types,
            "tile_size":       round(self.tile_var.get(), 1),
            "request_delay":   round(self.delay_var.get(), 1),
            "verbose":         self.verbose_var.get(),
        }

        self.run_btn.configure(state="disabled", text="⏳  Running…")
        self.stop_btn.configure(state="normal")
        self.progress.configure(value=0)

        def on_done(success):
            def _reset():
                self.run_btn.configure(state="normal", text="▶  START DOWNLOAD")
                self.stop_btn.configure(state="disabled", text="⏹  STOP")
            self.after(0, _reset)
            if success:
                messagebox.showinfo("Done", f"File saved:\n{output}")

        thread = threading.Thread(
            target=run_download,
            args=(config, self._log, self._set_progress, on_done),
            daemon=True,
        )
        thread.start()

    def _stop(self):
        _stop_event.set()
        self.stop_btn.configure(state="disabled", text="Stopping…")
        self._log("⏹ Stop requested — finishing current request then halting…", "warn")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()