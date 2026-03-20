from __future__ import annotations
import csv, json, os, random
from datetime import datetime
from pathlib import Path

import webbrowser
from PyQt6.QtCore    import Qt, QObject, QThread, QUrl, QTimer, pyqtSlot, pyqtSignal
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QGroupBox, QLineEdit, QListWidget, QStatusBar, QProgressBar,
    QSizePolicy, QAbstractItemView, QMenu, QSpinBox, QTabWidget,
    QScrollArea,
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore    import QWebEngineSettings
from PyQt6.QtWebChannel       import QWebChannel
from PyQt6.QtGui              import QColor, QFont, QFontMetrics, QPainter
from PyQt6.QtCore             import QSize, QRect

from benchmark_engines import (
    NetworkXEngine, ValhallaEngine, GraphHopperEngine, OSRMEngine,
)
from benchmark_engines.valhalla_engine import TRAFFIC_PROFILES
from benchmark_ui.worker import LoadWorker, RouteWorker
from benchmark_ui.restriction_dialog import CreateRestrictionDialog

# Directory where restriction JSON files are stored
RESTRICTION_DIR = Path(__file__).parent.parent / "Restriction"

MAP_HTML = str(
    Path(__file__).parent.parent / "benchmark_assets" / "map.html"
)

ENGINE_LIST = [
    NetworkXEngine(),
    ValhallaEngine(),
    GraphHopperEngine(),
    OSRMEngine(),
]


# ── Custom horizontal bar chart widget ─────────────────────────────────────
class HorizontalBarChart(QWidget):
    """Draws a multi-group horizontal bar chart using QPainter.

    data  — list of (label, {metric_name: value}, color_hex)
    metrics — ordered list of metric names to show per bar group
    palette — {metric_name: bar_color_hex}  (overrides per-engine color for metrics)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows:    list[tuple[str, dict, str]] = []   # (label, {metric: val}, eng_color)
        self._metrics: list[str] = []
        self._unit    = ""
        self._title   = ""
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def set_data(self, rows: list[tuple[str, dict, str]],
                 metrics: list[str], unit: str = "", title: str = ""):
        self._rows    = rows
        self._metrics = metrics
        self._unit    = unit
        self._title   = title
        n   = len(rows)
        m   = len(metrics)
        bar = max(14, min(28, 22))
        gap = 6
        top = 28 if title else 10
        self.setMinimumHeight(top + n * (m * (bar + gap) + 10) + 10)
        self.update()

    def sizeHint(self) -> QSize:
        n   = len(self._rows)
        m   = max(1, len(self._metrics))
        return QSize(400, 28 + n * (m * 28 + 12) + 12)

    def paintEvent(self, event):
        if not self._rows or not self._metrics:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        W = self.width()
        label_w = 110
        val_w   = 68
        top_pad = 28 if self._title else 10
        bar_h   = max(12, min(24, 20))
        gap_bar = 4      # gap between metric bars in same group
        gap_grp = 10     # gap between engine groups

        # Metric alpha shading: first metric = full, subsequent = lighter
        metric_alphas = [255, 170, 110]

        # Global max across all metrics for uniform scale
        max_val = max(
            (v for _, d, _ in self._rows for v in d.values() if isinstance(v, (int, float))),
            default=1.0
        ) or 1.0

        bar_area_w = W - label_w - val_w - 16

        # Title
        if self._title:
            p.setPen(QColor("#a6adc8"))
            p.setFont(QFont("Arial", 9))
            p.drawText(QRect(4, 4, W - 8, 20), Qt.AlignmentFlag.AlignLeft, self._title)

        y = top_pad
        for label, data_dict, eng_color in self._rows:
            group_top = y

            for mi, metric in enumerate(self._metrics):
                val = data_dict.get(metric, 0.0)
                if val <= 0:
                    y += bar_h + gap_bar
                    continue

                # Label (only on first metric of each group)
                if mi == 0:
                    p.setPen(QColor(eng_color))
                    p.setFont(QFont("Arial", 9, QFont.Weight.Bold))
                    fm = QFontMetrics(p.font())
                    lbl = fm.elidedText(label, Qt.TextElideMode.ElideRight, label_w - 8)
                    p.drawText(QRect(4, y, label_w - 8, bar_h),
                               Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                               lbl)
                else:
                    # Metric sub-label
                    p.setPen(QColor("#6c7086"))
                    p.setFont(QFont("Arial", 8))
                    p.drawText(QRect(14, y, label_w - 18, bar_h),
                               Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                               metric)

                # Track background
                tx = label_w
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor("#313244"))
                p.drawRoundedRect(tx, y, bar_area_w, bar_h, 3, 3)

                # Bar fill
                fill_w = max(4, int(val / max_val * bar_area_w))
                c = QColor(eng_color)
                c.setAlpha(metric_alphas[min(mi, len(metric_alphas) - 1)])
                p.setBrush(c)
                p.drawRoundedRect(tx, y, fill_w, bar_h, 3, 3)

                # Value text
                val_str = f"{val:.0f}{(' ' + self._unit) if self._unit else ''}"
                p.setPen(QColor("#cdd6f4"))
                p.setFont(QFont("Arial", 8))
                p.drawText(QRect(label_w + bar_area_w + 4, y, val_w - 4, bar_h),
                           Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                           val_str)

                y += bar_h + gap_bar

            y += gap_grp

        p.end()


# ── Background thread: fetch way geometry from Overpass ────────────────────
class FetchWayThread(QThread):
    """Fetches an OSM way's node coordinates from Overpass API."""
    done = pyqtSignal(int, str)   # (way_id, json_latlngs_or_empty)

    def __init__(self, way_id: int):
        super().__init__()
        self.way_id = way_id

    def run(self):
        from benchmark_engines.valhalla_engine import fetch_way_geometry_overpass
        nodes = fetch_way_geometry_overpass(self.way_id)
        latlngs = [[n[0], n[1]] for n in nodes]
        self.done.emit(self.way_id, json.dumps(latlngs) if latlngs else "")


# ── Reroute speed benchmark worker ─────────────────────────────────────────
class RerouteWorker(QThread):
    """Runs N reroutes for a fixed start/end with random via-points.

    Each iteration picks a random point along the straight-line start→end,
    offsets it by 50–400 m in a random direction, then times engine.route()
    with that via-point.  Emits per-attempt results for live dashboard update.
    """
    result_ready = pyqtSignal(float, bool, float)   # elapsed_ms, ok, dist_km
    progress     = pyqtSignal(int, int)              # done, total
    finished     = pyqtSignal()

    def __init__(self, engine, start: tuple, end: tuple, mode: str, n: int):
        super().__init__()
        self._engine = engine
        self._start  = start
        self._end    = end
        self._mode   = mode
        self._n      = n
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        import time as _time
        import math as _math
        import random as _rand

        s_lat, s_lon = self._start
        e_lat, e_lon = self._end

        for i in range(self._n):
            if self._stop_flag:
                break

            # Random via-point: linear interp along start→end + random offset
            t        = _rand.uniform(0.15, 0.85)
            base_lat = s_lat + t * (e_lat - s_lat)
            base_lon = s_lon + t * (e_lon - s_lon)

            angle  = _rand.uniform(0, 2 * _math.pi)
            dist_m = _rand.uniform(50, 400)
            dlat   = dist_m / 111_320 * _math.cos(angle)
            dlon   = dist_m / 111_320 * _math.sin(angle) / max(
                        _math.cos(_math.radians(base_lat)), 0.001)

            via = (base_lat + dlat, base_lon + dlon)

            t0 = _time.monotonic()
            try:
                result     = self._engine.route(
                    s_lat, s_lon, e_lat, e_lon,
                    mode=self._mode,
                    via_points=[via],
                )
                elapsed_ms = (_time.monotonic() - t0) * 1000
                ok         = result.ok
                dist_km    = result.distance_km if result.ok else 0.0
            except Exception:
                elapsed_ms = (_time.monotonic() - t0) * 1000
                ok         = False
                dist_km    = 0.0

            self.result_ready.emit(elapsed_ms, ok, dist_km)
            self.progress.emit(i + 1, self._n)

        self.finished.emit()


class FetchNodeThread(QThread):
    """Fetches coordinates for a list of OSM node IDs from Overpass."""
    done = pyqtSignal(str, str)   # (restriction_id, json_node_coords_or_empty)

    def __init__(self, restriction_id: str, node_ids: list[int]):
        super().__init__()
        self.restriction_id = restriction_id
        self.node_ids       = node_ids

    def run(self):
        from benchmark_engines.valhalla_engine import fetch_node_coords_overpass
        coords = fetch_node_coords_overpass(self.node_ids)
        # {node_id: (lat, lon)} → JSON-serialisable
        out = {str(nid): list(ll) for nid, ll in coords.items()}
        self.done.emit(self.restriction_id, json.dumps(out) if out else "")


class VerifyGeometryThread(QThread):
    """Background thread that checks each RC restriction's stored geometry
    against the current Overpass API data and reports any mismatches.

    Signals
    -------
    mismatch_found(rec_id, way_id, reason)
        Emitted once per way whose geometry has changed since the restriction
        was created.  `reason` is a short human-readable description.
    finished_all(checked, mismatches)
        Emitted when all restrictions have been verified.
    """
    mismatch_found = pyqtSignal(str, int, str)   # rec_id, way_id, reason
    finished_all   = pyqtSignal(int, int)         # total_checked, total_mismatches

    def __init__(self, restrictions: list[dict]):
        super().__init__()
        self.restrictions = restrictions

    def run(self):
        from benchmark_engines.valhalla_engine import (
            fetch_way_geometry_overpass, detect_geometry_change,
        )
        checked = mismatches = 0
        for rec in self.restrictions:
            if rec.get("type") != "RC":
                continue
            stored_geom = rec.get("geometry", {})
            for way_id in rec.get("way_ids", []):
                stored = stored_geom.get(str(way_id), [])
                if not stored:
                    continue   # geometry not yet fetched — skip
                checked += 1
                current = fetch_way_geometry_overpass(int(way_id))
                reason  = detect_geometry_change(stored, current)
                if reason:
                    mismatches += 1
                    self.mismatch_found.emit(rec["id"], way_id, reason)
        self.finished_all.emit(checked, mismatches)


# ── Python ↔ JS bridge ─────────────────────────────────────────────────────
class Bridge(QObject):
    click_mode_changed = pyqtSignal(str)   # exposed to JS

    def __init__(self, window):
        super().__init__()
        self._win  = window
        self._mode = None   # "start" | "end" | None

    def set_click_mode(self, mode: str | None):
        self._mode = mode
        self.click_mode_changed.emit(mode or "")

    @pyqtSlot(float, float)
    def map_clicked(self, lat: float, lon: float):
        if self._mode == "start":
            self._win.on_start_set(lat, lon)
        elif self._mode == "end":
            self._win.on_end_set(lat, lon)

    @pyqtSlot(str, float, float)
    def marker_dragged(self, kind: str, lat: float, lon: float):
        if kind == "start":
            self._win.on_start_set(lat, lon, from_drag=True)
        else:
            self._win.on_end_set(lat, lon, from_drag=True)

    @pyqtSlot(str, float, float)
    def set_point(self, kind: str, lat: float, lon: float):
        """Called from right-click context menu — sets start or end directly."""
        if kind == "start":
            self._win.on_start_set(lat, lon)
        else:
            self._win.on_end_set(lat, lon)

    @pyqtSlot(str)
    def copy_to_clipboard(self, text: str):
        """Copy text to OS clipboard via Qt (works inside QWebEngineView sandbox)."""
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(text)

    @pyqtSlot(str)
    def via_points_changed(self, data: str):
        """Called from JS when via markers are added, dragged, or removed."""
        import json
        pts = json.loads(data)   # [[lat, lon], ...]
        self._win.on_via_points_changed([(p[0], p[1]) for p in pts])

    @pyqtSlot(str)
    def open_url(self, url: str):
        webbrowser.open(url)


# ── Main Window ────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Routing Benchmark Tool")
        self.resize(1400, 920)

        self._start:      tuple[float, float] | None = None
        self._end:        tuple[float, float] | None = None
        self._via_points: list[tuple[float, float]]  = []
        self._route_worker:      RouteWorker | None = None
        self._load_worker:       LoadWorker  | None = None
        self._nx_loaded:         bool = False
        self._network_path:      str  = ""
        self._restrictions_path: str  = ""

        # Trip history — list of dicts, newest first
        self._trip_history: list[dict] = []
        self._current_run_results: list = []   # accumulates during _do_route

        # Per-engine benchmark stats {name: {trips, ok, total_ms, min_ms, max_ms, total_km}}
        self._bench_stats: dict[str, dict] = {
            eng.NAME: {"trips": 0, "ok": 0, "total_ms": 0.0,
                       "min_ms": float("inf"), "max_ms": 0.0, "total_km": 0.0}
            for eng in ENGINE_LIST
        }

        # Batch random trip runner state
        self._batch_remaining: int = 0
        self._batch_total:     int = 0

        # Failed trips log for investigation
        self._failed_trips: list[dict] = []

        # Reroute speed test state
        self._reroute_worker:  QThread | None = None
        self._reroute_stats: dict = {
            "engine": "", "n": 0, "ok": 0,
            "total_ms": 0.0, "min_ms": float("inf"),
            "max_ms": 0.0,   "total_km": 0.0,
        }
        self._reroute_history: list[dict] = []

        # Restriction records (RC / TR), loaded from disk + added live
        self._restrictions: list[dict] = []
        self._fetch_threads: list[QThread] = []   # keep refs alive

        self._build_ui()
        self._setup_map()
        self._refresh_engine_status()
        self._load_restrictions_from_disk()   # loads data only — no JS yet

    # ── UI construction ────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left control panel (scrollable) ────────────────────────────
        _sidebar_scroll = QScrollArea()
        _sidebar_scroll.setFixedWidth(265)
        _sidebar_scroll.setWidgetResizable(True)
        _sidebar_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        _sidebar_scroll.setStyleSheet(
            "QScrollArea{background:#1e1e2e;border:none;}"
            "QScrollBar:vertical{background:#1e1e2e;width:8px;margin:0px;}"
            "QScrollBar::handle:vertical{background:#45475a;border-radius:4px;"
            "min-height:20px;}"
            "QScrollBar::handle:vertical:hover{background:#585b70;}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical"
            "{height:0px;}"
        )
        panel = QWidget()
        panel.setStyleSheet("background:#1e1e2e; color:#cdd6f4;")
        playout = QVBoxLayout(panel)
        playout.setContentsMargins(10, 10, 10, 10)
        playout.setSpacing(8)

        title = QLabel("🗺 Routing Benchmark")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title.setStyleSheet("color:#cba6f7; padding-bottom:4px;")
        playout.addWidget(title)

        # ── Network file ───────────────────────────────────────────────
        g_data = self._group("Data")
        self._network_label = QLabel("No network loaded")
        self._network_label.setWordWrap(True)
        self._network_label.setStyleSheet("color:#a6adc8; font-size:10px;")
        btn_network = self._button("Load Road Network …")
        btn_network.clicked.connect(self._load_network)
        btn_restrict = self._button("Load Restrictions (opt.)")
        btn_restrict.clicked.connect(self._load_restrictions)
        self._restrict_label = QLabel("none")
        self._restrict_label.setStyleSheet("color:#a6adc8; font-size:10px;")
        g_data.layout().addWidget(self._network_label)
        g_data.layout().addWidget(btn_network)
        g_data.layout().addWidget(btn_restrict)
        g_data.layout().addWidget(self._restrict_label)
        playout.addWidget(g_data)

        # ── Basemap ────────────────────────────────────────────────────
        g_base = self._group("Basemap")
        self._basemap_combo = QComboBox()
        self._basemap_combo.addItems(
            ["OSM", "CartoDB Dark", "CartoDB Light", "ESRI Satellite"]
        )
        self._basemap_combo.setStyleSheet(self._combo_style())
        self._basemap_combo.currentTextChanged.connect(self._change_basemap)
        g_base.layout().addWidget(self._basemap_combo)
        playout.addWidget(g_base)

        # ── Mode ───────────────────────────────────────────────────────
        g_mode = self._group("Travel Mode")
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["car", "motorcycle", "bike", "walk"])
        self._mode_combo.setStyleSheet(self._combo_style())
        g_mode.layout().addWidget(self._mode_combo)
        playout.addWidget(g_mode)

        # ── Traffic profile ────────────────────────────────────────────
        g_traffic = self._group("Traffic Profile")
        g_traffic.setToolTip("Controls Valhalla duration factor.\nAuto uses current system time.")
        self._traffic_combo = QComboBox()
        self._traffic_combo.addItems([label for label, _ in TRAFFIC_PROFILES])
        self._traffic_combo.setStyleSheet(self._combo_style())
        self._traffic_combo.currentIndexChanged.connect(self._on_traffic_profile_changed)

        self._traffic_factor_lbl = QLabel("factor: auto")
        self._traffic_factor_lbl.setStyleSheet("color:#a6adc8; font-size:10px;")

        g_traffic.layout().addWidget(self._traffic_combo)
        g_traffic.layout().addWidget(self._traffic_factor_lbl)
        playout.addWidget(g_traffic)

        # ── Restrictions (RC / TR) ─────────────────────────────────────
        g_restrict = self._group("Restrictions")
        g_restrict.setToolTip(
            "RC = Road Closure (block one or more ways)\n"
            "TR = Turn Restriction (block a specific turn sequence)\n"
            "Records are saved to benchmark_tool/Restriction/ as JSON files."
        )

        # Create buttons row
        create_row = QHBoxLayout()
        btn_new_rc = QPushButton("🚧 Road Closure")
        btn_new_rc.setFixedHeight(26)
        btn_new_rc.setStyleSheet(
            "QPushButton{background:#f38ba8;color:#1e1e2e;border-radius:4px;"
            "font-weight:700;font-size:10px;padding:2px 6px;}"
            "QPushButton:hover{background:#ff9aac;}"
        )
        btn_new_rc.clicked.connect(lambda: self._create_restriction("RC"))
        create_row.addWidget(btn_new_rc)

        btn_new_tr = QPushButton("↩ Turn Restr.")
        btn_new_tr.setFixedHeight(26)
        btn_new_tr.setStyleSheet(
            "QPushButton{background:#fab387;color:#1e1e2e;border-radius:4px;"
            "font-weight:700;font-size:10px;padding:2px 6px;}"
            "QPushButton:hover{background:#ffc59a;}"
        )
        btn_new_tr.clicked.connect(lambda: self._create_restriction("TR"))
        create_row.addWidget(btn_new_tr)
        g_restrict.layout().addLayout(create_row)

        # List of active restrictions
        self._restrict_list = QListWidget()
        self._restrict_list.setMaximumHeight(90)
        self._restrict_list.setStyleSheet(
            "QListWidget{background:#181825;color:#cdd6f4;border-radius:4px;"
            "border:none;font-size:10px;}"
            "QListWidget::item{padding:2px 4px;}"
            "QListWidget::item:selected{background:#313244;}"
        )
        self._restrict_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._restrict_list.customContextMenuRequested.connect(self._restriction_context_menu)
        self._restrict_list.itemDoubleClicked.connect(self._zoom_to_restriction_item)
        g_restrict.layout().addWidget(self._restrict_list)

        restrict_btn_row = QHBoxLayout()
        btn_rm_r = QPushButton("Remove")
        btn_rm_r.setFixedHeight(22)
        btn_rm_r.setStyleSheet(
            "QPushButton{background:#313244;color:#cdd6f4;border-radius:4px;"
            "padding:2px 6px;font-size:10px;}"
            "QPushButton:hover{background:#45475a;}"
        )
        btn_rm_r.clicked.connect(self._remove_restriction)
        restrict_btn_row.addWidget(btn_rm_r)

        btn_clr_r = QPushButton("Clear All")
        btn_clr_r.setFixedHeight(22)
        btn_clr_r.setStyleSheet(
            "QPushButton{background:#313244;color:#f38ba8;border-radius:4px;"
            "padding:2px 6px;font-size:10px;}"
            "QPushButton:hover{background:#45475a;}"
        )
        btn_clr_r.clicked.connect(self._clear_restrictions)
        restrict_btn_row.addWidget(btn_clr_r)
        g_restrict.layout().addLayout(restrict_btn_row)

        # Buffer size control
        buf_row = QHBoxLayout()
        buf_lbl = QLabel("Buffer zone:")
        buf_lbl.setStyleSheet("color:#a6adc8; font-size:10px;")
        self._buf_spinbox = QSpinBox()
        self._buf_spinbox.setRange(1, 500)
        self._buf_spinbox.setValue(10)
        self._buf_spinbox.setSuffix(" m")
        self._buf_spinbox.setFixedWidth(72)
        self._buf_spinbox.setStyleSheet(
            "QSpinBox{background:#313244;color:#cdd6f4;border:1px solid #45475a;"
            "border-radius:4px;padding:2px 4px;font-size:10px;}"
            "QSpinBox::up-button,QSpinBox::down-button{width:14px;}"
        )
        self._buf_spinbox.valueChanged.connect(self._on_buffer_changed)
        buf_row.addWidget(buf_lbl)
        buf_row.addStretch()
        buf_row.addWidget(self._buf_spinbox)
        g_restrict.layout().addLayout(buf_row)
        playout.addWidget(g_restrict)

        # ── Engines ────────────────────────────────────────────────────
        g_eng = self._group("Engines")
        self._eng_checks: dict[str, QCheckBox] = {}
        self._eng_status: dict[str, QLabel]    = {}
        for eng in ENGINE_LIST:
            row = QHBoxLayout()
            cb  = QCheckBox(eng.NAME)
            cb.setChecked(True)
            cb.setStyleSheet(f"color:{eng.COLOR}; font-weight:600;")
            lbl = QLabel("…")
            lbl.setStyleSheet("color:#a6adc8; font-size:10px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(cb)
            row.addWidget(lbl)
            g_eng.layout().addLayout(row)
            self._eng_checks[eng.NAME] = cb
            self._eng_status[eng.NAME] = lbl
        playout.addWidget(g_eng)

        # ── Point buttons ──────────────────────────────────────────────
        g_pts = self._group("Points")
        self._btn_start = self._button("📍 Set Start")
        self._btn_end   = self._button("🏁 Set End")
        self._start_lbl = QLabel("not set")
        self._end_lbl   = QLabel("not set")
        for lbl in (self._start_lbl, self._end_lbl):
            lbl.setStyleSheet("color:#a6adc8; font-size:10px;")
        self._btn_start.clicked.connect(lambda: self._set_click_mode("start"))
        self._btn_end.clicked.connect(lambda: self._set_click_mode("end"))

        # Swap button (⇅) between start and end
        self._btn_swap = QPushButton("⇅  Swap")
        self._btn_swap.setToolTip("Swap Start and End")
        self._btn_swap.setStyleSheet(
            "QPushButton{background:#45475a;color:#cdd6f4;border-radius:4px;"
            "font-size:11px;padding:3px 8px;}"
            "QPushButton:hover{background:#585b70;}"
        )
        self._btn_swap.clicked.connect(self._swap_start_end)

        g_pts.layout().addWidget(self._btn_start)
        g_pts.layout().addWidget(self._start_lbl)
        g_pts.layout().addWidget(self._btn_swap)
        g_pts.layout().addWidget(self._btn_end)
        g_pts.layout().addWidget(self._end_lbl)
        playout.addWidget(g_pts)

        # ── Action buttons ─────────────────────────────────────────────
        self._btn_route = self._button("▶  Route", accent=True)
        self._btn_route.clicked.connect(self._do_route)
        self._btn_clear = self._button("✕  Clear")
        self._btn_clear.clicked.connect(self._do_clear)
        playout.addWidget(self._btn_route)
        playout.addWidget(self._btn_clear)

        # ── Random Trip ────────────────────────────────────────────────
        g_rand = self._group("Benchmark")
        self._btn_random = self._button("🎲  Random Trip")
        self._btn_random.setToolTip("Generate random start+end within current map view and route")
        self._btn_random.clicked.connect(self._random_trip)

        batch_row = QHBoxLayout()
        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 9999)
        self._batch_spin.setValue(10)
        self._batch_spin.setFixedWidth(65)
        self._batch_spin.setStyleSheet(
            "QSpinBox{background:#313244;color:#cdd6f4;border:1px solid #45475a;"
            "border-radius:4px;padding:2px 4px;font-size:10px;}"
            "QSpinBox::up-button,QSpinBox::down-button{width:14px;}"
        )
        self._btn_batch = QPushButton("▶ Run N")
        self._btn_batch.setFixedHeight(24)
        self._btn_batch.setStyleSheet(
            "QPushButton{background:#89dceb;color:#1e1e2e;border-radius:4px;"
            "font-weight:700;font-size:10px;padding:2px 6px;}"
            "QPushButton:hover{background:#a6e8f5;}"
            "QPushButton:disabled{background:#45475a;color:#6c7086;}"
        )
        self._btn_batch.clicked.connect(self._start_batch)
        self._btn_batch_stop = QPushButton("■ Stop")
        self._btn_batch_stop.setFixedHeight(24)
        self._btn_batch_stop.setEnabled(False)
        self._btn_batch_stop.setStyleSheet(
            "QPushButton{background:#f38ba8;color:#1e1e2e;border-radius:4px;"
            "font-weight:700;font-size:10px;padding:2px 6px;}"
            "QPushButton:hover{background:#ff9aac;}"
            "QPushButton:disabled{background:#45475a;color:#6c7086;}"
        )
        self._btn_batch_stop.clicked.connect(self._stop_batch)
        batch_row.addWidget(self._batch_spin)
        batch_row.addWidget(self._btn_batch)
        batch_row.addWidget(self._btn_batch_stop)

        self._batch_progress = QProgressBar()
        self._batch_progress.setRange(0, 100)
        self._batch_progress.setVisible(False)
        self._batch_progress.setStyleSheet(
            "QProgressBar{background:#313244;border-radius:4px;height:6px;text-align:center;}"
            "QProgressBar::chunk{background:#89dceb;border-radius:4px;}"
        )
        g_rand.layout().addWidget(self._btn_random)
        g_rand.layout().addLayout(batch_row)
        g_rand.layout().addWidget(self._batch_progress)
        playout.addWidget(g_rand)

        # ── Reroute Speed Test ─────────────────────────────────────────
        g_reroute = self._group("Reroute Speed Test")

        rr_eng_row = QHBoxLayout()
        rr_eng_lbl = QLabel("Engine:")
        rr_eng_lbl.setStyleSheet("color:#a6adc8; font-size:10px;")
        self._reroute_eng_combo = QComboBox()
        for eng in ENGINE_LIST:
            self._reroute_eng_combo.addItem(eng.NAME)
        self._reroute_eng_combo.setStyleSheet(self._combo_style())
        rr_eng_row.addWidget(rr_eng_lbl)
        rr_eng_row.addWidget(self._reroute_eng_combo, 1)
        g_reroute.layout().addLayout(rr_eng_row)

        rr_run_row = QHBoxLayout()
        self._reroute_spin = QSpinBox()
        self._reroute_spin.setRange(1, 999)
        self._reroute_spin.setValue(20)
        self._reroute_spin.setFixedWidth(58)
        self._reroute_spin.setStyleSheet(
            "QSpinBox{background:#313244;color:#cdd6f4;border:1px solid #45475a;"
            "border-radius:4px;padding:2px 4px;font-size:10px;}"
            "QSpinBox::up-button,QSpinBox::down-button{width:14px;}"
        )
        self._btn_reroute_run = QPushButton("▶ Run")
        self._btn_reroute_run.setFixedHeight(24)
        self._btn_reroute_run.setEnabled(False)
        self._btn_reroute_run.setStyleSheet(
            "QPushButton{background:#a6e3a1;color:#1e1e2e;border-radius:4px;"
            "font-weight:700;font-size:10px;padding:2px 6px;}"
            "QPushButton:hover{background:#b8f0b0;}"
            "QPushButton:disabled{background:#45475a;color:#6c7086;}"
        )
        self._btn_reroute_run.clicked.connect(self._start_reroute)

        self._btn_reroute_stop = QPushButton("■ Stop")
        self._btn_reroute_stop.setFixedHeight(24)
        self._btn_reroute_stop.setEnabled(False)
        self._btn_reroute_stop.setStyleSheet(
            "QPushButton{background:#f38ba8;color:#1e1e2e;border-radius:4px;"
            "font-weight:700;font-size:10px;padding:2px 6px;}"
            "QPushButton:hover{background:#ff9aac;}"
            "QPushButton:disabled{background:#45475a;color:#6c7086;}"
        )
        self._btn_reroute_stop.clicked.connect(self._stop_reroute)
        rr_run_row.addWidget(self._reroute_spin)
        rr_run_row.addWidget(self._btn_reroute_run)
        rr_run_row.addWidget(self._btn_reroute_stop)
        g_reroute.layout().addLayout(rr_run_row)

        self._reroute_progress = QProgressBar()
        self._reroute_progress.setRange(0, 100)
        self._reroute_progress.setVisible(False)
        self._reroute_progress.setStyleSheet(
            "QProgressBar{background:#313244;border-radius:4px;height:6px;"
            "text-align:center;}"
            "QProgressBar::chunk{background:#a6e3a1;border-radius:4px;}"
        )
        self._reroute_status_lbl = QLabel("Route first to enable")
        self._reroute_status_lbl.setStyleSheet("color:#a6adc8; font-size:10px;")
        self._reroute_status_lbl.setWordWrap(True)
        g_reroute.layout().addWidget(self._reroute_progress)
        g_reroute.layout().addWidget(self._reroute_status_lbl)
        playout.addWidget(g_reroute)

        # ── NetworkX load progress ─────────────────────────────────────
        self._nx_progress = QProgressBar()
        self._nx_progress.setRange(0, 0)
        self._nx_progress.setVisible(False)
        self._nx_progress.setStyleSheet(
            "QProgressBar{background:#313244;border-radius:4px;height:6px;}"
            "QProgressBar::chunk{background:#cba6f7;border-radius:4px;}"
        )
        playout.addWidget(self._nx_progress)

        playout.addStretch()

        _sidebar_scroll.setWidget(panel)

        # ── Right side: tab widget (Map | Dashboard) ───────────────────
        right   = QWidget()
        rlayout = QVBoxLayout(right)
        rlayout.setContentsMargins(0, 0, 0, 0)
        rlayout.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(
            "QTabWidget::pane{border:none;}"
            "QTabBar::tab{background:#313244;color:#a6adc8;padding:6px 18px;"
            "border-radius:4px 4px 0 0;margin-right:2px;font-size:11px;}"
            "QTabBar::tab:selected{background:#1e1e2e;color:#cba6f7;font-weight:700;}"
            "QTabBar::tab:hover{background:#45475a;}"
        )

        # ── Map tab ────────────────────────────────────────────────────
        map_tab = QWidget()
        map_tab_layout = QVBoxLayout(map_tab)
        map_tab_layout.setContentsMargins(0, 0, 0, 0)
        map_tab_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # Map
        self._map_view = QWebEngineView()
        self._map_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        splitter.addWidget(self._map_view)

        # ── Current results table ──────────────────────────────────────
        results_container = QWidget()
        results_container.setStyleSheet("background:#181825;")
        rc_layout = QVBoxLayout(results_container)
        rc_layout.setContentsMargins(0, 0, 0, 0)
        rc_layout.setSpacing(0)

        results_header = QLabel("  📊 Current Results")
        results_header.setStyleSheet(
            "color:#cba6f7; font-weight:700; font-size:11px;"
            "background:#181825; padding:4px 8px;"
            "border-bottom:1px solid #313244;"
        )
        rc_layout.addWidget(results_header)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Engine", "Duration", "Distance", "Status", "Color"]
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setStyleSheet(
            "QTableWidget{background:#181825;color:#cdd6f4;gridline-color:#313244;border:none;}"
            "QHeaderView::section{background:#313244;color:#cdd6f4;border:none;padding:4px;}"
            "QTableWidget::item:selected{background:#45475a;}"
        )
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.cellClicked.connect(self._on_table_click)
        rc_layout.addWidget(self._table)

        results_container.setMaximumHeight(185)
        splitter.addWidget(results_container)

        # ── Trip history panel ─────────────────────────────────────────
        history_container = QWidget()
        history_container.setStyleSheet("background:#11111b;")
        hc_layout = QVBoxLayout(history_container)
        hc_layout.setContentsMargins(0, 0, 0, 0)
        hc_layout.setSpacing(0)

        # History header row
        hist_header_row = QWidget()
        hist_header_row.setStyleSheet(
            "background:#181825; border-bottom:1px solid #313244;"
        )
        hhr_layout = QHBoxLayout(hist_header_row)
        hhr_layout.setContentsMargins(8, 4, 8, 4)

        hist_title = QLabel("🕒 Trip History")
        hist_title.setStyleSheet("color:#cba6f7; font-weight:700; font-size:11px;")
        hhr_layout.addWidget(hist_title)
        hhr_layout.addStretch()

        self._hist_count_lbl = QLabel("0 trips")
        self._hist_count_lbl.setStyleSheet("color:#a6adc8; font-size:10px;")
        hhr_layout.addWidget(self._hist_count_lbl)

        btn_replay = QPushButton("↩ Replay")
        btn_replay.setFixedHeight(22)
        btn_replay.setStyleSheet(
            "QPushButton{background:#313244;color:#cdd6f4;border-radius:4px;"
            "padding:2px 8px;font-size:10px;}"
            "QPushButton:hover{background:#45475a;}"
        )
        btn_replay.clicked.connect(self._replay_selected_history)
        hhr_layout.addWidget(btn_replay)

        btn_export_osm = QPushButton("⬇ Export OSM IDs")
        btn_export_osm.setFixedHeight(22)
        btn_export_osm.setStyleSheet(
            "QPushButton{background:#313244;color:#89dceb;border-radius:4px;"
            "padding:2px 8px;font-size:10px;}"
            "QPushButton:hover{background:#45475a;}"
        )
        btn_export_osm.clicked.connect(self._export_osm_ids)
        hhr_layout.addWidget(btn_export_osm)

        btn_clear_hist = QPushButton("🗑 Clear")
        btn_clear_hist.setFixedHeight(22)
        btn_clear_hist.setStyleSheet(
            "QPushButton{background:#313244;color:#f38ba8;border-radius:4px;"
            "padding:2px 8px;font-size:10px;}"
            "QPushButton:hover{background:#45475a;}"
        )
        btn_clear_hist.clicked.connect(self._clear_history)
        hhr_layout.addWidget(btn_clear_hist)

        hc_layout.addWidget(hist_header_row)

        # History table
        self._hist_table = QTableWidget(0, 9)
        self._hist_table.setHorizontalHeaderLabels([
            "#", "Time", "Mode", "Start", "End",
            "Best Engine", "Duration", "Distance", "OSM Ways",
        ])
        hh = self._hist_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(6, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(7, QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(8, QHeaderView.ResizeMode.Fixed)
        self._hist_table.setColumnWidth(0, 36)
        self._hist_table.setColumnWidth(1, 130)
        self._hist_table.setColumnWidth(2, 78)
        self._hist_table.setColumnWidth(5, 90)
        self._hist_table.setColumnWidth(6, 80)
        self._hist_table.setColumnWidth(7, 80)
        self._hist_table.setColumnWidth(8, 76)
        self._hist_table.setStyleSheet(
            "QTableWidget{background:#11111b;color:#cdd6f4;"
            "gridline-color:#1e1e2e;border:none;}"
            "QHeaderView::section{background:#181825;color:#a6adc8;"
            "border:none;padding:3px;font-size:10px;}"
            "QTableWidget::item{padding:3px;}"
            "QTableWidget::item:selected{background:#313244;}"
        )
        self._hist_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._hist_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._hist_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._hist_table.verticalHeader().setVisible(False)
        self._hist_table.cellDoubleClicked.connect(self._on_history_double_click)
        hc_layout.addWidget(self._hist_table)

        history_container.setMinimumHeight(160)
        splitter.addWidget(history_container)

        splitter.setSizes([560, 160, 160])

        map_tab_layout.addWidget(splitter)

        # ── Dashboard tab ──────────────────────────────────────────────
        self._dash_tab = self._build_dashboard_tab()

        self._tabs.addTab(map_tab,        "🗺  Map")
        self._tabs.addTab(self._dash_tab, "📊  Dashboard")

        rlayout.addWidget(self._tabs)

        root.addWidget(_sidebar_scroll)
        root.addWidget(right, 1)

        # Status bar
        self._status = QStatusBar()
        self._status.setStyleSheet("background:#181825; color:#a6adc8;")
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — load a road network and click Set Start / Set End")

    # ── Dashboard tab builder ──────────────────────────────────────────
    def _build_dashboard_tab(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background:#11111b; color:#cdd6f4;")
        root = QVBoxLayout(w)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        # ── Header ──────────────────────────────────────────────────
        hdr_row = QHBoxLayout()
        title_lbl = QLabel("📊 Benchmark Dashboard")
        title_lbl.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        title_lbl.setStyleSheet("color:#cba6f7;")
        hdr_row.addWidget(title_lbl)
        hdr_row.addStretch()
        btn_clear_stats = QPushButton("🗑 Clear All")
        btn_clear_stats.setFixedHeight(26)
        btn_clear_stats.setStyleSheet(
            "QPushButton{background:#313244;color:#f38ba8;border-radius:4px;"
            "padding:2px 10px;font-size:10px;}"
            "QPushButton:hover{background:#45475a;}"
        )
        btn_clear_stats.clicked.connect(self._clear_bench_stats)
        hdr_row.addWidget(btn_clear_stats)
        root.addLayout(hdr_row)

        self._dash_summary_lbl = QLabel(
            "No trips recorded yet — use 🎲 Random Trip or ▶ Run N"
        )
        self._dash_summary_lbl.setStyleSheet("color:#a6adc8; font-size:11px;")
        root.addWidget(self._dash_summary_lbl)

        # ── Sub-tab widget ───────────────────────────────────────────
        _sub_tab_style = (
            "QTabWidget::pane{border:none;}"
            "QTabBar::tab{background:#313244;color:#a6adc8;padding:5px 14px;"
            "border-radius:4px 4px 0 0;margin-right:2px;font-size:10px;}"
            "QTabBar::tab:selected{background:#1e1e2e;color:#cba6f7;font-weight:700;}"
            "QTabBar::tab:hover{background:#45475a;}"
        )
        sub_tabs = QTabWidget()
        sub_tabs.setStyleSheet(_sub_tab_style)

        # ── Sub-tab 1: Trip Benchmark ────────────────────────────────
        bench_tab = QWidget()
        bench_tab.setStyleSheet("background:transparent;")
        bench_layout = QVBoxLayout(bench_tab)
        bench_layout.setContentsMargins(0, 8, 0, 0)
        bench_layout.setSpacing(8)

        # Stats table (left) + bar chart (right)
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(12)

        self._dash_table = QTableWidget(0, 7)
        self._dash_table.setHorizontalHeaderLabels([
            "Engine", "Trips", "Success %", "Avg ms", "Min ms", "Max ms", "Avg km",
        ])
        hh = self._dash_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, 7):
            hh.setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
        for col, w_px in enumerate([55, 75, 70, 65, 65, 65], start=1):
            self._dash_table.setColumnWidth(col, w_px)
        self._dash_table.setStyleSheet(
            "QTableWidget{background:#181825;color:#cdd6f4;"
            "gridline-color:#313244;border:none;border-radius:6px;}"
            "QHeaderView::section{background:#313244;color:#a6adc8;"
            "border:none;padding:5px;font-size:10px;}"
            "QTableWidget::item{padding:5px;}"
            "QTableWidget::item:selected{background:#313244;}"
        )
        self._dash_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._dash_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._dash_table.verticalHeader().setVisible(False)
        top_layout.addWidget(self._dash_table, 3)

        chart_panel = QWidget()
        chart_panel.setStyleSheet("background:#181825; border-radius:6px;")
        chart_panel_layout = QVBoxLayout(chart_panel)
        chart_panel_layout.setContentsMargins(10, 10, 10, 10)
        chart_panel_layout.setSpacing(4)

        chart_lbl = QLabel("Response time  (Avg · Min · Max) — lower is faster")
        chart_lbl.setStyleSheet("color:#a6adc8; font-size:10px;")
        chart_panel_layout.addWidget(chart_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea{border:none; background:transparent;}"
            "QScrollBar:vertical{width:6px;background:#181825;}"
            "QScrollBar::handle:vertical{background:#45475a;border-radius:3px;}"
        )
        self._bar_chart = HorizontalBarChart()
        self._bar_chart.setStyleSheet("background:transparent;")
        scroll.setWidget(self._bar_chart)
        chart_panel_layout.addWidget(scroll, 1)
        top_layout.addWidget(chart_panel, 4)

        bench_layout.addLayout(top_layout, 1)
        sub_tabs.addTab(bench_tab, "📊 Trip Benchmark")

        # ── Sub-tab 2: Reroute Speed ─────────────────────────────────
        reroute_tab = QWidget()
        reroute_tab.setStyleSheet("background:transparent;")
        reroute_layout = QVBoxLayout(reroute_tab)
        reroute_layout.setContentsMargins(0, 8, 0, 0)
        reroute_layout.setSpacing(8)

        # Reroute header row
        rr_hdr_row = QHBoxLayout()
        rr_title_lbl = QLabel("🔄 Reroute Response Time")
        rr_title_lbl.setStyleSheet("color:#a6e3a1; font-weight:700; font-size:11px;")
        rr_hdr_row.addWidget(rr_title_lbl)
        rr_hdr_row.addStretch()
        btn_rr_clear = QPushButton("🗑 Clear")
        btn_rr_clear.setFixedHeight(24)
        btn_rr_clear.setStyleSheet(
            "QPushButton{background:#313244;color:#f38ba8;border-radius:4px;"
            "padding:2px 8px;font-size:10px;}"
            "QPushButton:hover{background:#45475a;}"
        )
        btn_rr_clear.clicked.connect(self._clear_reroute_stats)
        rr_hdr_row.addWidget(btn_rr_clear)
        reroute_layout.addLayout(rr_hdr_row)

        # Top area: stats table (left) + bar chart (right)
        rr_top = QHBoxLayout()
        rr_top.setContentsMargins(0, 0, 0, 0)
        rr_top.setSpacing(12)

        self._rr_table = QTableWidget(0, 6)
        self._rr_table.setHorizontalHeaderLabels([
            "Engine", "Attempts", "Success%", "Avg ms", "Min ms", "Max ms",
        ])
        rr_hh = self._rr_table.horizontalHeader()
        rr_hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, 6):
            rr_hh.setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
        for col, w_px in enumerate([68, 75, 70, 65, 65], start=1):
            self._rr_table.setColumnWidth(col, w_px)
        self._rr_table.setStyleSheet(
            "QTableWidget{background:#181825;color:#cdd6f4;"
            "gridline-color:#313244;border:none;border-radius:6px;}"
            "QHeaderView::section{background:#313244;color:#a6adc8;"
            "border:none;padding:5px;font-size:10px;}"
            "QTableWidget::item{padding:5px;}"
            "QTableWidget::item:selected{background:#313244;}"
        )
        self._rr_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._rr_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._rr_table.verticalHeader().setVisible(False)
        rr_top.addWidget(self._rr_table, 3)

        rr_chart_panel = QWidget()
        rr_chart_panel.setStyleSheet("background:#181825; border-radius:6px;")
        rr_chart_layout = QVBoxLayout(rr_chart_panel)
        rr_chart_layout.setContentsMargins(10, 10, 10, 10)
        rr_chart_layout.setSpacing(4)

        rr_chart_lbl = QLabel("Response time  (Avg · Min · Max) — lower is faster")
        rr_chart_lbl.setStyleSheet("color:#a6adc8; font-size:10px;")
        rr_chart_layout.addWidget(rr_chart_lbl)

        rr_scroll = QScrollArea()
        rr_scroll.setWidgetResizable(True)
        rr_scroll.setStyleSheet(
            "QScrollArea{border:none; background:transparent;}"
            "QScrollBar:vertical{width:6px;background:#181825;}"
            "QScrollBar::handle:vertical{background:#45475a;border-radius:3px;}"
        )
        self._rr_bar_chart = HorizontalBarChart()
        self._rr_bar_chart.setStyleSheet("background:transparent;")
        rr_scroll.setWidget(self._rr_bar_chart)
        rr_chart_layout.addWidget(rr_scroll, 1)
        rr_top.addWidget(rr_chart_panel, 4)

        reroute_layout.addLayout(rr_top)

        # History table
        self._rr_history_table = QTableWidget(0, 5)
        self._rr_history_table.setHorizontalHeaderLabels([
            "Time", "Engine", "Response (ms)", "OK", "Dist (km)",
        ])
        rr_hhist = self._rr_history_table.horizontalHeader()
        rr_hhist.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        rr_hhist.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        rr_hhist.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        rr_hhist.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        rr_hhist.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self._rr_history_table.setColumnWidth(0, 75)
        self._rr_history_table.setColumnWidth(2, 100)
        self._rr_history_table.setColumnWidth(3, 40)
        self._rr_history_table.setColumnWidth(4, 80)
        self._rr_history_table.setStyleSheet(
            "QTableWidget{background:#181825;color:#cdd6f4;"
            "gridline-color:#313244;border:none;border-radius:6px;}"
            "QHeaderView::section{background:#313244;color:#a6adc8;"
            "border:none;padding:4px;font-size:10px;}"
            "QTableWidget::item{padding:3px;}"
            "QTableWidget::item:selected{background:#45475a;}"
        )
        self._rr_history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._rr_history_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._rr_history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._rr_history_table.verticalHeader().setVisible(False)
        reroute_layout.addWidget(self._rr_history_table, 1)

        sub_tabs.addTab(reroute_tab, "🔄 Reroute Speed")

        # ── Vertical splitter: sub_tabs (top) | failed trips (bottom) ─
        v_split = QSplitter(Qt.Orientation.Vertical)
        v_split.setStyleSheet(
            "QSplitter::handle{background:#313244; height:3px;}"
        )

        v_split.addWidget(sub_tabs)

        fail_widget = QWidget()
        fail_widget.setStyleSheet("background:transparent;")
        fail_layout = QVBoxLayout(fail_widget)
        fail_layout.setContentsMargins(0, 4, 0, 0)
        fail_layout.setSpacing(4)

        fail_hdr = QHBoxLayout()
        self._fail_count_lbl = QLabel("❌ Failed Trips: 0")
        self._fail_count_lbl.setStyleSheet(
            "color:#f38ba8; font-weight:700; font-size:11px;"
        )
        fail_hdr.addWidget(self._fail_count_lbl)
        fail_hdr.addStretch()

        btn_fail_replay = QPushButton("↩ Replay on Map")
        btn_fail_replay.setFixedHeight(24)
        btn_fail_replay.setStyleSheet(
            "QPushButton{background:#313244;color:#cdd6f4;border-radius:4px;"
            "padding:2px 10px;font-size:10px;}"
            "QPushButton:hover{background:#45475a;}"
        )
        btn_fail_replay.clicked.connect(self._replay_failed_trip)
        fail_hdr.addWidget(btn_fail_replay)

        btn_fail_clear = QPushButton("🗑 Clear")
        btn_fail_clear.setFixedHeight(24)
        btn_fail_clear.setStyleSheet(
            "QPushButton{background:#313244;color:#f38ba8;border-radius:4px;"
            "padding:2px 8px;font-size:10px;}"
            "QPushButton:hover{background:#45475a;}"
        )
        btn_fail_clear.clicked.connect(self._clear_failed_trips)
        fail_hdr.addWidget(btn_fail_clear)
        fail_layout.addLayout(fail_hdr)

        self._fail_table = QTableWidget(0, 6)
        self._fail_table.setHorizontalHeaderLabels([
            "#", "Time", "Engine", "Mode", "Error", "Coords (start → end)",
        ])
        fhh = self._fail_table.horizontalHeader()
        fhh.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        fhh.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        fhh.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        fhh.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        fhh.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        fhh.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self._fail_table.setColumnWidth(0, 36)
        self._fail_table.setColumnWidth(1, 130)
        self._fail_table.setColumnWidth(2, 90)
        self._fail_table.setColumnWidth(3, 70)
        self._fail_table.setStyleSheet(
            "QTableWidget{background:#181825;color:#cdd6f4;"
            "gridline-color:#313244;border:none;border-radius:6px;}"
            "QHeaderView::section{background:#313244;color:#a6adc8;"
            "border:none;padding:4px;font-size:10px;}"
            "QTableWidget::item{padding:4px;}"
            "QTableWidget::item:selected{background:#45475a;}"
        )
        self._fail_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._fail_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._fail_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._fail_table.verticalHeader().setVisible(False)
        self._fail_table.cellDoubleClicked.connect(
            lambda r, _: self._replay_failed_trip_row(r)
        )
        fail_layout.addWidget(self._fail_table, 1)

        v_split.addWidget(fail_widget)
        v_split.setSizes([480, 200])

        root.addWidget(v_split, 1)
        return w

    # ── Map setup ──────────────────────────────────────────────────────
    def _setup_map(self):
        self._bridge  = Bridge(self)
        self._channel = QWebChannel()
        self._channel.registerObject("bridge", self._bridge)
        self._map_view.page().setWebChannel(self._channel)

        settings = self._map_view.page().settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)

        # Draw restriction overlays only after the page JS is fully ready
        self._map_view.page().loadFinished.connect(self._on_map_loaded)
        self._map_view.load(QUrl.fromLocalFile(MAP_HTML))

    def _js(self, code: str):
        self._map_view.page().runJavaScript(code)

    # ── Engine status refresh ──────────────────────────────────────────
    def _refresh_engine_status(self):
        for eng in ENGINE_LIST:
            if eng.NAME == "NetworkX":
                text = "loaded" if self._nx_loaded else "needs network file"
            else:
                text = "online" if eng.is_available() else "offline"
            lbl = self._eng_status[eng.NAME]
            lbl.setText(text)
            lbl.setStyleSheet(
                f"color:{'#a6e3a1' if text in ('online','loaded') else '#f38ba8'};"
                f"font-size:10px;"
            )

    # ── Load network file ──────────────────────────────────────────────
    def _load_network(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Road Network", "",
            "GeoJSON / GPKG (*.geojson *.gpkg *.json)"
        )
        if not path:
            return
        self._network_path = path
        self._network_label.setText(os.path.basename(path))
        self._start_nx_load()

    def _load_restrictions(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Restrictions File", "", "JSON (*.json)"
        )
        if path:
            self._restrictions_path = path
            self._restrict_label.setText(os.path.basename(path))

    def _start_nx_load(self):
        nx_eng = next(e for e in ENGINE_LIST if isinstance(e, NetworkXEngine))
        self._nx_progress.setVisible(True)
        self._status.showMessage("Building NetworkX graph … this may take several minutes")
        self._load_worker = LoadWorker(nx_eng, self._network_path,
                                       self._restrictions_path or None)
        self._load_worker.finished.connect(self._on_nx_loaded)
        self._load_worker.error.connect(self._on_nx_error)
        self._load_worker.status.connect(self._status.showMessage)
        self._load_worker.start()

    def _on_nx_loaded(self):
        self._nx_loaded = True
        self._nx_progress.setVisible(False)
        self._refresh_engine_status()
        self._status.showMessage("NetworkX graph ready")

    def _on_nx_error(self, msg: str):
        self._nx_progress.setVisible(False)
        self._status.showMessage(f"NetworkX load error: {msg}")

    # ── Traffic profile ────────────────────────────────────────────────
    def _on_traffic_profile_changed(self, idx: int):
        from benchmark_engines.valhalla_engine import TRAFFIC_PROFILES, _BASE_FACTOR, _time_of_day_multiplier
        label, multiplier = TRAFFIC_PROFILES[idx]

        for eng in ENGINE_LIST:
            if hasattr(eng, "traffic_multiplier"):
                eng.traffic_multiplier = multiplier

        if multiplier is None:
            mult = _time_of_day_multiplier()
            effective = _BASE_FACTOR["car"] * mult
            self._traffic_factor_lbl.setText(
                f"auto → ×{mult:.2f} now  (car factor ×{effective:.2f})"
            )
        else:
            effective = _BASE_FACTOR["car"] * multiplier
            self._traffic_factor_lbl.setText(
                f"car factor ×{effective:.2f}  |  moto ×{_BASE_FACTOR['motorcycle'] * multiplier:.2f}"
            )

    # ── Click mode ─────────────────────────────────────────────────────
    def _set_click_mode(self, mode: str):
        self._bridge.set_click_mode(mode)
        self._btn_start.setStyleSheet(
            "background:#a6e3a1; color:#1e1e2e;" if mode == "start" else ""
        )
        self._btn_end.setStyleSheet(
            "background:#f38ba8; color:#1e1e2e;" if mode == "end" else ""
        )

    def on_start_set(self, lat: float, lon: float, from_drag: bool = False):
        self._start = (lat, lon)
        self._start_lbl.setText(f"{lat:.5f}, {lon:.5f}")
        self._js(f"setStart({lat}, {lon})")
        if not from_drag:
            self._bridge.set_click_mode(None)
            self._btn_start.setStyleSheet("")

    def on_end_set(self, lat: float, lon: float, from_drag: bool = False):
        self._end = (lat, lon)
        self._end_lbl.setText(f"{lat:.5f}, {lon:.5f}")
        self._js(f"setEnd({lat}, {lon})")
        if not from_drag:
            self._bridge.set_click_mode(None)
            self._btn_end.setStyleSheet("")

    def on_via_points_changed(self, pts: list[tuple[float, float]]):
        """Called from Bridge when JS via markers change — auto re-routes."""
        self._via_points = pts
        if self._start and self._end:
            self._do_route()

    def _swap_start_end(self):
        """Swap start and end points (like Google Maps swap button)."""
        if not self._start and not self._end:
            return
        old_start, old_end = self._start, self._end
        if old_end:
            self.on_start_set(old_end[0], old_end[1])
        else:
            self._start = None
            self._start_lbl.setText("not set")
            self._js("if(startMarker){map.removeLayer(startMarker);startMarker=null;}")
        if old_start:
            self.on_end_set(old_start[0], old_start[1])
        else:
            self._end = None
            self._end_lbl.setText("not set")
            self._js("if(endMarker){map.removeLayer(endMarker);endMarker=null;}")

    # ── Random Trip ────────────────────────────────────────────────────
    def _random_trip(self):
        """Pick random start+end within current Leaflet viewport and route."""
        self._map_view.page().runJavaScript(
            "[map.getBounds().getSouthWest().lat,"
            " map.getBounds().getSouthWest().lng,"
            " map.getBounds().getNorthEast().lat,"
            " map.getBounds().getNorthEast().lng]",
            self._apply_random_trip,
        )

    def _apply_random_trip(self, bounds):
        if not bounds or len(bounds) < 4:
            self._status.showMessage("Cannot get map bounds — is the map loaded?")
            return
        min_lat, min_lon, max_lat, max_lon = bounds
        lat1 = random.uniform(min_lat, max_lat)
        lon1 = random.uniform(min_lon, max_lon)
        lat2 = random.uniform(min_lat, max_lat)
        lon2 = random.uniform(min_lon, max_lon)
        self._do_clear()
        self.on_start_set(lat1, lon1)
        self.on_end_set(lat2, lon2)
        self._do_route()

    # ── Batch runner ───────────────────────────────────────────────────
    def _start_batch(self):
        n = self._batch_spin.value()
        self._batch_total     = n
        self._batch_remaining = n
        self._btn_batch.setEnabled(False)
        self._btn_batch_stop.setEnabled(True)
        self._batch_progress.setRange(0, n)
        self._batch_progress.setValue(0)
        self._batch_progress.setVisible(True)
        # Switch to dashboard tab so user can watch stats live
        self._tabs.setCurrentWidget(self._dash_tab)
        self._run_next_batch_trip()

    def _stop_batch(self):
        self._batch_remaining = 0
        self._btn_batch.setEnabled(True)
        self._btn_batch_stop.setEnabled(False)
        self._batch_progress.setVisible(False)
        self._status.showMessage("Batch stopped")

    def _run_next_batch_trip(self):
        if self._batch_remaining <= 0:
            self._btn_batch.setEnabled(True)
            self._btn_batch_stop.setEnabled(False)
            self._batch_progress.setVisible(False)
            done = self._batch_total - self._batch_remaining
            self._status.showMessage(f"Batch complete — {done} trips run")
            return
        done = self._batch_total - self._batch_remaining
        self._batch_progress.setValue(done)
        self._status.showMessage(
            f"Batch: {done}/{self._batch_total} — running trip {done+1}…"
        )
        self._map_view.page().runJavaScript(
            "[map.getBounds().getSouthWest().lat,"
            " map.getBounds().getSouthWest().lng,"
            " map.getBounds().getNorthEast().lat,"
            " map.getBounds().getNorthEast().lng]",
            self._apply_batch_trip,
        )

    def _apply_batch_trip(self, bounds):
        if not bounds or len(bounds) < 4 or self._batch_remaining <= 0:
            self._stop_batch()
            return
        min_lat, min_lon, max_lat, max_lon = bounds
        lat1 = random.uniform(min_lat, max_lat)
        lon1 = random.uniform(min_lon, max_lon)
        lat2 = random.uniform(min_lat, max_lat)
        lon2 = random.uniform(min_lon, max_lon)
        self._batch_remaining -= 1
        # Set points silently (no map transition)
        self._start = (lat1, lon1)
        self._end   = (lat2, lon2)
        self._start_lbl.setText(f"{lat1:.5f}, {lon1:.5f}")
        self._end_lbl.setText(f"{lat2:.5f}, {lon2:.5f}")
        self._js(f"setStart({lat1}, {lon1})")
        self._js(f"setEnd({lat2}, {lon2})")
        self._do_route()
        # _on_all_done → _on_batch_trip_done is wired below

    # ── Dashboard stats ────────────────────────────────────────────────
    def _record_bench_result(self, result):
        """Update per-engine stats from a single RouteResult."""
        s = self._bench_stats.get(result.engine)
        if s is None:
            return
        s["trips"] += 1
        if result.ok:
            s["ok"]       += 1
            s["total_ms"] += result.elapsed_ms
            s["min_ms"]    = min(s["min_ms"], result.elapsed_ms)
            s["max_ms"]    = max(s["max_ms"], result.elapsed_ms)
            s["total_km"] += result.distance_km
        else:
            self._failed_trips.append({
                "ts":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "engine": result.engine,
                "mode":   self._mode_combo.currentText(),
                "start":  self._start,
                "end":    self._end,
                "error":  result.error or "unknown",
            })

    def _update_dashboard(self):
        """Refresh the dashboard table, bar chart, and failed trips from _bench_stats."""
        t = self._dash_table
        t.setRowCount(0)

        valid = {
            name: s for name, s in self._bench_stats.items()
            if s["trips"] > 0
        }
        total_trips = sum(s["trips"] for s in valid.values())

        # Update failed trips table
        self._refresh_fail_table()

        if total_trips == 0:
            self._dash_summary_lbl.setText(
                "No trips recorded yet — use 🎲 Random Trip or ▶ Run N"
            )
            self._bar_chart.set_data([], [])
            return

        total_ok = sum(s["ok"] for s in valid.values())
        n_fail   = sum(s["trips"] - s["ok"] for s in valid.values())
        self._dash_summary_lbl.setText(
            f"Total trips: {total_trips}  |  "
            f"Successful: {total_ok}  |  "
            f"Failed: {n_fail}  |  "
            f"Engines: {len(valid)}"
        )

        # Sort by avg response (fastest first)
        eng_order = sorted(
            valid.keys(),
            key=lambda n: (
                valid[n]["total_ms"] / valid[n]["ok"]
                if valid[n]["ok"] else float("inf")
            )
        )

        avg_ms_list = [
            valid[n]["total_ms"] / valid[n]["ok"] if valid[n]["ok"] else 0
            for n in eng_order
        ]
        fastest_avg = min((v for v in avg_ms_list if v > 0), default=None)

        def dcell(text, color=None, align=Qt.AlignmentFlag.AlignCenter):
            item = QTableWidgetItem(text)
            item.setTextAlignment(align)
            if color:
                item.setForeground(QColor(color))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            return item

        for name in eng_order:
            s       = self._bench_stats[name]
            eng_obj = next((e for e in ENGINE_LIST if e.NAME == name), None)
            color   = eng_obj.COLOR if eng_obj else "#cdd6f4"

            ok    = s["ok"]
            trips = s["trips"]
            avg_ms  = s["total_ms"] / ok if ok else 0
            min_ms  = s["min_ms"]    if ok else 0
            max_ms  = s["max_ms"]    if ok else 0
            avg_km  = s["total_km"] / ok if ok else 0
            pct     = int(100 * ok / trips) if trips else 0

            row = t.rowCount()
            t.insertRow(row)
            t.setRowHeight(row, 30)
            t.setItem(row, 0, dcell(name, color))
            t.setItem(row, 1, dcell(str(trips)))
            t.setItem(row, 2, dcell(
                f"{pct}%",
                "#a6e3a1" if pct == 100 else ("#fab387" if pct >= 80 else "#f38ba8")
            ))
            t.setItem(row, 3, dcell(
                f"{avg_ms:.0f}",
                "#89dceb" if fastest_avg and avg_ms == fastest_avg else "#cdd6f4"
            ))
            t.setItem(row, 4, dcell(f"{min_ms:.0f}", "#a6e3a1"))
            t.setItem(row, 5, dcell(f"{max_ms:.0f}", "#f38ba8"))
            t.setItem(row, 6, dcell(f"{avg_km:.2f}"))

        # Horizontal bar chart: Avg · Min · Max response time per engine
        chart_rows = []
        for name in eng_order:
            s = self._bench_stats[name]
            if not s["ok"]:
                continue
            eng_obj = next((e for e in ENGINE_LIST if e.NAME == name), None)
            color   = eng_obj.COLOR if eng_obj else "#cdd6f4"
            chart_rows.append((
                name,
                {
                    "Avg": s["total_ms"] / s["ok"],
                    "Min": s["min_ms"],
                    "Max": s["max_ms"],
                },
                color,
            ))
        self._bar_chart.set_data(
            chart_rows,
            metrics=["Avg", "Min", "Max"],
            unit="ms",
        )

    def _refresh_fail_table(self):
        ft = self._fail_table
        ft.setRowCount(0)
        n = len(self._failed_trips)
        self._fail_count_lbl.setText(
            f"{'❌' if n else '✓'} Failed Trips: {n}"
        )
        self._fail_count_lbl.setStyleSheet(
            f"color:{'#f38ba8' if n else '#a6e3a1'}; font-weight:700; font-size:11px;"
        )

        for i, entry in enumerate(reversed(self._failed_trips)):
            row = ft.rowCount()
            ft.insertRow(row)
            ft.setRowHeight(row, 26)

            def fc(text, color=None):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if color:
                    item.setForeground(QColor(color))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                return item

            eng_obj = next((e for e in ENGINE_LIST if e.NAME == entry["engine"]), None)
            eng_color = eng_obj.COLOR if eng_obj else "#cdd6f4"

            start = entry.get("start") or (0, 0)
            end   = entry.get("end")   or (0, 0)
            coords_str = (
                f"{start[0]:.5f},{start[1]:.5f} → {end[0]:.5f},{end[1]:.5f}"
            )
            mode_icons = {"car": "🚗", "motorcycle": "🏍", "bike": "🚲", "walk": "🚶"}

            ft.setItem(row, 0, fc(str(n - i), "#6c7086"))
            ft.setItem(row, 1, fc(entry["ts"], "#a6adc8"))
            ft.setItem(row, 2, fc(entry["engine"], eng_color))
            ft.setItem(row, 3, fc(
                f"{mode_icons.get(entry['mode'], '')} {entry['mode']}"
            ))
            err = entry["error"]
            ft.setItem(row, 4, fc(err[:80] if len(err) > 80 else err, "#f38ba8"))
            ft.setItem(row, 5, fc(coords_str, "#a6adc8"))

    def _replay_failed_trip(self):
        row = self._fail_table.currentRow()
        if row < 0:
            self._status.showMessage("Select a failed trip row first")
            return
        self._replay_failed_trip_row(row)

    def _replay_failed_trip_row(self, row: int):
        # Table shows newest first (reversed), map back to _failed_trips index
        n   = len(self._failed_trips)
        idx = n - 1 - row
        if idx < 0 or idx >= n:
            return
        entry = self._failed_trips[idx]
        start = entry.get("start")
        end   = entry.get("end")
        if not start or not end:
            self._status.showMessage("No coordinates stored for this failure")
            return
        self._tabs.setCurrentIndex(0)   # switch to Map tab
        self._do_clear()
        self.on_start_set(start[0], start[1])
        self.on_end_set(end[0], end[1])
        self._status.showMessage(
            f"Replaying failed trip #{n - row} — {entry['engine']} ({entry['error'][:60]})"
        )
        self._do_route()

    def _clear_failed_trips(self):
        self._failed_trips.clear()
        self._refresh_fail_table()
        self._status.showMessage("Failed trips cleared")

    def _clear_bench_stats(self):
        for s in self._bench_stats.values():
            s["trips"]    = 0
            s["ok"]       = 0
            s["total_ms"] = 0.0
            s["min_ms"]   = float("inf")
            s["max_ms"]   = 0.0
            s["total_km"] = 0.0
        self._failed_trips.clear()
        self._update_dashboard()
        self._status.showMessage("Benchmark stats cleared")

    # ── Reroute speed test ─────────────────────────────────────────────
    def _enable_reroute(self):
        """Call after a route is successfully computed."""
        self._btn_reroute_run.setEnabled(True)
        self._reroute_status_lbl.setText("Ready — click Run to start reroute test")

    def _start_reroute(self):
        if not self._start or not self._end:
            self._status.showMessage("Set start and end first")
            return

        eng_name = self._reroute_eng_combo.currentText()
        engine   = next((e for e in ENGINE_LIST if e.NAME == eng_name), None)
        if engine is None:
            return

        n    = self._reroute_spin.value()
        mode = self._mode_combo.currentText()

        # Reset stats for this run (keep history cumulative)
        self._reroute_stats = {
            "engine": eng_name, "n": 0, "ok": 0,
            "total_ms": 0.0, "min_ms": float("inf"),
            "max_ms": 0.0,   "total_km": 0.0,
        }

        self._reroute_worker = RerouteWorker(engine, self._start, self._end, mode, n)
        self._reroute_worker.result_ready.connect(self._on_reroute_result)
        self._reroute_worker.progress.connect(self._on_reroute_progress)
        self._reroute_worker.finished.connect(self._on_reroute_finished)
        self._reroute_worker.start()

        self._btn_reroute_run.setEnabled(False)
        self._btn_reroute_stop.setEnabled(True)
        self._reroute_progress.setVisible(True)
        self._reroute_progress.setValue(0)
        self._reroute_status_lbl.setText(f"Running 0/{n} …")
        self._status.showMessage(f"Reroute test: {n} attempts on {eng_name}")

    def _stop_reroute(self):
        if self._reroute_worker:
            self._reroute_worker.stop()

    def _on_reroute_result(self, elapsed_ms: float, ok: bool, dist_km: float):
        s = self._reroute_stats
        s["n"] += 1
        if ok:
            s["ok"]       += 1
            s["total_ms"] += elapsed_ms
            s["min_ms"]    = min(s["min_ms"], elapsed_ms)
            s["max_ms"]    = max(s["max_ms"], elapsed_ms)
            s["total_km"] += dist_km
        self._reroute_history.append({
            "ts":  datetime.now().strftime("%H:%M:%S"),
            "ms":  elapsed_ms,
            "ok":  ok,
            "km":  dist_km,
            "eng": s["engine"],
        })
        self._update_reroute_dashboard()

    def _on_reroute_progress(self, done: int, total: int):
        self._reroute_progress.setValue(int(done / total * 100))
        self._reroute_status_lbl.setText(f"Running {done}/{total} …")

    def _on_reroute_finished(self):
        self._btn_reroute_run.setEnabled(True)
        self._btn_reroute_stop.setEnabled(False)
        self._reroute_progress.setVisible(False)
        s = self._reroute_stats
        if s["ok"] > 0:
            avg = s["total_ms"] / s["ok"]
            self._reroute_status_lbl.setText(
                f"Done ✓  {s['ok']}/{s['n']} OK  avg {avg:.0f} ms"
            )
        else:
            self._reroute_status_lbl.setText(f"Done  0/{s['n']} OK")
        self._update_reroute_dashboard()
        self._tabs.setCurrentIndex(1)   # switch to Dashboard

    def _update_reroute_dashboard(self):
        """Refresh reroute stats table, bar chart, and history table."""
        s = self._reroute_stats
        if not s["engine"]:
            return

        # ── Stats table ───────────────────────────────────────────────
        t = self._rr_table
        t.setRowCount(0)

        ok    = s["ok"]
        n     = s["n"]
        avg   = s["total_ms"] / ok if ok else 0
        pct   = int(100 * ok / n)  if n  else 0

        eng_obj  = next((e for e in ENGINE_LIST if e.NAME == s["engine"]), None)
        color    = eng_obj.COLOR if eng_obj else "#cdd6f4"

        def rc(text, clr=None):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if clr:
                item.setForeground(QColor(clr))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            return item

        t.insertRow(0)
        t.setRowHeight(0, 30)
        t.setItem(0, 0, rc(s["engine"], color))
        t.setItem(0, 1, rc(str(n)))
        t.setItem(0, 2, rc(f"{pct}%",
            "#a6e3a1" if pct == 100 else ("#fab387" if pct >= 80 else "#f38ba8")))
        t.setItem(0, 3, rc(f"{avg:.0f}", "#89dceb"))
        t.setItem(0, 4, rc(f"{s['min_ms']:.0f}" if ok else "—", "#a6e3a1"))
        t.setItem(0, 5, rc(f"{s['max_ms']:.0f}" if ok else "—", "#f38ba8"))

        # ── Bar chart ─────────────────────────────────────────────────
        if ok:
            self._rr_bar_chart.set_data(
                [(s["engine"],
                  {"Avg": avg, "Min": s["min_ms"], "Max": s["max_ms"]},
                  color)],
                metrics=["Avg", "Min", "Max"],
                unit="ms",
            )

        # ── History table ─────────────────────────────────────────────
        ht = self._rr_history_table
        ht.setRowCount(0)
        for entry in reversed(self._reroute_history[-200:]):   # show last 200
            row = ht.rowCount()
            ht.insertRow(row)
            ht.setRowHeight(row, 22)
            ms_color = (
                "#a6e3a1" if entry["ms"] < 100 else
                "#fab387" if entry["ms"] < 500 else "#f38ba8"
            )
            ht.setItem(row, 0, rc(entry["ts"], "#6c7086"))
            ht.setItem(row, 1, rc(entry["eng"],
                eng_obj.COLOR if eng_obj else "#cdd6f4"))
            ht.setItem(row, 2, rc(f"{entry['ms']:.0f} ms", ms_color))
            ht.setItem(row, 3, rc("✓" if entry["ok"] else "✗",
                "#a6e3a1" if entry["ok"] else "#f38ba8"))
            ht.setItem(row, 4, rc(f"{entry['km']:.2f} km" if entry["ok"] else "—"))

    def _clear_reroute_stats(self):
        self._reroute_stats = {
            "engine": "", "n": 0, "ok": 0,
            "total_ms": 0.0, "min_ms": float("inf"),
            "max_ms": 0.0,   "total_km": 0.0,
        }
        self._reroute_history.clear()
        self._update_reroute_dashboard()
        self._rr_table.setRowCount(0)
        self._rr_bar_chart.set_data([], [])
        self._rr_history_table.setRowCount(0)

    # ── Routing ────────────────────────────────────────────────────────
    def _do_route(self):
        if not self._start or not self._end:
            self._status.showMessage("Set Start and End points first")
            return

        selected = [
            e for e in ENGINE_LIST
            if self._eng_checks[e.NAME].isChecked()
        ]
        if not selected:
            self._status.showMessage("Select at least one engine")
            return

        self._js("clearRoutes()")
        self._table.setRowCount(0)
        self._current_run_results = []
        self._btn_route.setEnabled(False)
        self._status.showMessage("Routing …")

        mode = self._mode_combo.currentText()
        active = [r for r in self._restrictions if self._is_restriction_active(r)]
        self._route_worker = RouteWorker(
            selected,
            self._start[0], self._start[1],
            self._end[0],   self._end[1],
            mode,
            restrictions=active,
            buffer_deg=self._buffer_deg,
            via_points=self._via_points or None,
        )
        self._route_worker.result_ready.connect(self._on_result)
        self._route_worker.all_done.connect(self._on_all_done)
        self._route_worker.start()

    def _on_result(self, result):
        self._current_run_results.append(result)
        self._record_bench_result(result)

        if result.ok:
            coords_json = json.dumps(result.coordinates)
            dist  = f"{result.distance_km:.2f} km"
            dur_s = int(result.duration_min * 60)
            h, m  = divmod(dur_s // 60, 60)
            dur   = f"{h}h {m}min" if h > 0 else f"{m} min"
            self._js(
                f"addRoute({json.dumps(result.engine)}, {json.dumps(result.color)}, "
                f"{json.dumps(coords_json)}, {json.dumps(dist)}, {json.dumps(dur)})"
            )
            if self._table.rowCount() == 0 and self._start and self._end:
                sl, slo = self._start
                el, elo = self._end
                dur_s_raw = result.duration_min * 60
                self._js(
                    f"updateInfoPanel({sl}, {slo}, {el}, {elo}, "
                    f"{json.dumps(dur)}, {json.dumps(dist)}, "
                    f"{json.dumps(result.engine)}, {json.dumps(result.color)}, "
                    f"{result.distance_km}, {dur_s_raw})"
                )

        row = self._table.rowCount()
        self._table.insertRow(row)

        def cell(text, color=None):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if color:
                item.setForeground(QColor(color))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            return item

        self._table.setItem(row, 0, cell(result.engine, result.color))
        self._table.setItem(row, 1, cell(
            f"{result.duration_min:.0f} min" if result.ok else "—"
        ))
        self._table.setItem(row, 2, cell(
            f"{result.distance_km:.2f} km" if result.ok else "—"
        ))
        status_text  = "✓" if result.ok else f"✗ {result.error[:40]}"
        status_color = "#a6e3a1" if result.ok else "#f38ba8"
        self._table.setItem(row, 3, cell(status_text, status_color))

        swatch = QWidget()
        swatch.setStyleSheet(f"background:{result.color}; border-radius:4px;")
        self._table.setCellWidget(row, 4, swatch)

    def _on_all_done(self):
        self._update_dashboard()
        self._enable_reroute()
        self._btn_route.setEnabled(True)
        active = [r for r in self._restrictions if self._is_restriction_active(r)]
        n_r = len(active)

        # Check if any RC way leaked into the result
        leaked_ways: list[int] = []
        if active and self._current_run_results:
            rc_way_ids = [
                wid for rec in active if rec.get("type") == "RC"
                for wid in rec.get("way_ids", [])
            ]
            for r in self._current_run_results:
                if r.ok and r.osm_way_ids:
                    leaked = [w for w in rc_way_ids if w in r.osm_way_ids]
                    if leaked:
                        leaked_ways = leaked
                        break

        if leaked_ways:
            self._status.showMessage(
                f"⚠ Route leaked through restricted way(s) {leaked_ways} — "
                f"No viable detour found. Try different start/end or add adjacent ways."
            )
        else:
            suffix = f"  |  {n_r} restriction{'s' if n_r != 1 else ''} active ✓" if n_r else ""
            self._status.showMessage(f"Done{suffix}")

        self._js("fitToRoutes()")

        if self._current_run_results and self._start and self._end:
            self._add_to_history(
                mode    = self._mode_combo.currentText(),
                start   = self._start,
                end     = self._end,
                results = list(self._current_run_results),
            )

        # Continue batch if running
        if self._batch_remaining > 0:
            QTimer.singleShot(50, self._run_next_batch_trip)

    def _do_clear(self):
        self._js("clearAll()")   # also calls clearViaPoints() in JS
        self._js("clearInfoPanel()")
        self._table.setRowCount(0)
        self._start = self._end = None
        self._via_points = []
        self._start_lbl.setText("not set")
        self._end_lbl.setText("not set")
        self._bridge.set_click_mode(None)
        self._btn_start.setStyleSheet("")
        self._btn_end.setStyleSheet("")
        self._btn_reroute_run.setEnabled(False)
        self._reroute_status_lbl.setText("Route first to enable")
        # Re-draw restriction overlays from stored geometry (no Overpass needed)
        for rec in self._restrictions:
            self._draw_restriction_on_map(rec)

    # ── Table click → highlight route ──────────────────────────────────
    def _on_table_click(self, row: int, _col: int):
        item = self._table.item(row, 0)
        if item:
            self._js(f"highlightRoute({json.dumps(item.text())})")

    # ── Trip history ───────────────────────────────────────────────────
    def _add_to_history(
        self,
        mode:    str,
        start:   tuple[float, float],
        end:     tuple[float, float],
        results: list,
    ):
        ok_results = [r for r in results if r.ok]
        best = min(ok_results, key=lambda r: r.duration_min) if ok_results else None

        # Grab OSM way IDs from Valhalla result if available
        osm_way_ids: list[int] = []
        for r in results:
            if r.ok and r.osm_way_ids:
                osm_way_ids = r.osm_way_ids
                break

        entry = {
            "ts":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode":        mode,
            "start":       start,
            "end":         end,
            "results":     results,
            "best":        best,
            "osm_way_ids": osm_way_ids,
        }
        self._trip_history.insert(0, entry)   # newest first
        self._render_history_table()

    def _render_history_table(self):
        self._hist_table.setRowCount(0)
        total = len(self._trip_history)
        self._hist_count_lbl.setText(f"{total} trip{'s' if total != 1 else ''}")

        for i, entry in enumerate(self._trip_history):
            row = self._hist_table.rowCount()
            self._hist_table.insertRow(row)

            run_num  = total - i
            best     = entry["best"]
            sl, slo  = entry["start"]
            el, elo  = entry["end"]

            mode_icons = {"car": "🚗", "motorcycle": "🏍", "bike": "🚲", "walk": "🚶"}
            mode_icon  = mode_icons.get(entry["mode"], "")

            def hcell(text, color=None, align=Qt.AlignmentFlag.AlignCenter):
                item = QTableWidgetItem(text)
                item.setTextAlignment(align)
                if color:
                    item.setForeground(QColor(color))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                return item

            self._hist_table.setItem(row, 0, hcell(str(run_num), "#cba6f7"))
            self._hist_table.setItem(row, 1, hcell(entry["ts"], "#a6adc8"))
            self._hist_table.setItem(row, 2, hcell(f"{mode_icon} {entry['mode']}"))
            self._hist_table.setItem(row, 3, hcell(
                f"{sl:.5f}, {slo:.5f}",
                align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            ))
            self._hist_table.setItem(row, 4, hcell(
                f"{el:.5f}, {elo:.5f}",
                align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            ))

            if best:
                self._hist_table.setItem(row, 5, hcell(best.engine, best.color))
                h, m = divmod(int(best.duration_min * 60) // 60, 60)
                dur_str = f"{h}h {m}min" if h > 0 else f"{int(best.duration_min)} min"
                self._hist_table.setItem(row, 6, hcell(dur_str, "#89dceb"))
                self._hist_table.setItem(row, 7, hcell(f"{best.distance_km:.2f} km", "#a6e3a1"))
            else:
                self._hist_table.setItem(row, 5, hcell("—", "#f38ba8"))
                self._hist_table.setItem(row, 6, hcell("—"))
                self._hist_table.setItem(row, 7, hcell("—"))

            osm_ids = entry.get("osm_way_ids", [])
            if osm_ids:
                self._hist_table.setItem(row, 8, hcell(f"{len(osm_ids)} ways", "#fab387"))
            else:
                self._hist_table.setItem(row, 8, hcell("—", "#585b70"))

            self._hist_table.setRowHeight(row, 26)

    def _replay_selected_history(self):
        rows = self._hist_table.selectedItems()
        if not rows:
            self._status.showMessage("Select a history row first, then click Replay")
            return
        row = self._hist_table.currentRow()
        self._load_history_entry(row)

    def _on_history_double_click(self, row: int, _col: int):
        self._load_history_entry(row)

    def _load_history_entry(self, row: int):
        if row < 0 or row >= len(self._trip_history):
            return
        entry = self._trip_history[row]

        # Restore points
        sl, slo = entry["start"]
        el, elo = entry["end"]
        self._start = entry["start"]
        self._end   = entry["end"]
        self._start_lbl.setText(f"{sl:.5f}, {slo:.5f}")
        self._end_lbl.setText(f"{el:.5f}, {elo:.5f}")

        # Redraw map
        self._js("clearRoutes()")
        self._js(f"setStart({sl}, {slo})")
        self._js(f"setEnd({el}, {elo})")

        # Restore results table and re-draw routes
        self._table.setRowCount(0)
        first_ok = True
        for result in entry["results"]:
            if result.ok:
                coords_json = json.dumps(result.coordinates)
                dist  = f"{result.distance_km:.2f} km"
                dur_s = int(result.duration_min * 60)
                h, m  = divmod(dur_s // 60, 60)
                dur   = f"{h}h {m}min" if h > 0 else f"{m} min"
                self._js(
                    f"addRoute({json.dumps(result.engine)}, {json.dumps(result.color)}, "
                    f"{json.dumps(coords_json)}, {json.dumps(dist)}, {json.dumps(dur)})"
                )
                if first_ok:
                    dur_s_raw = result.duration_min * 60
                    self._js(
                        f"updateInfoPanel({sl}, {slo}, {el}, {elo}, "
                        f"{json.dumps(dur)}, {json.dumps(dist)}, "
                        f"{json.dumps(result.engine)}, {json.dumps(result.color)}, "
                        f"{result.distance_km}, {dur_s_raw})"
                    )
                    first_ok = False

            # Results table row
            trow = self._table.rowCount()
            self._table.insertRow(trow)

            def cell(text, color=None):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if color:
                    item.setForeground(QColor(color))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                return item

            self._table.setItem(trow, 0, cell(result.engine, result.color))
            self._table.setItem(trow, 1, cell(
                f"{result.duration_min:.0f} min" if result.ok else "—"
            ))
            self._table.setItem(trow, 2, cell(
                f"{result.distance_km:.2f} km" if result.ok else "—"
            ))
            status_text  = "✓" if result.ok else f"✗ {result.error[:40]}"
            status_color = "#a6e3a1" if result.ok else "#f38ba8"
            self._table.setItem(trow, 3, cell(status_text, status_color))

            swatch = QWidget()
            swatch.setStyleSheet(f"background:{result.color}; border-radius:4px;")
            self._table.setCellWidget(trow, 4, swatch)

        self._js("fitToRoutes()")
        self._status.showMessage(
            f"Loaded trip #{len(self._trip_history) - row}  |  {entry['ts']}  |  {entry['mode']}"
        )

    # ── Restriction management ─────────────────────────────────────────
    def _load_restrictions_from_disk(self):
        """Load restriction JSON files into memory + list widget only.
        Map drawing is deferred to _on_map_loaded() — JS not ready yet."""
        for subdir, rtype in [("Road closure", "RC"), ("Turn restriction", "TR")]:
            folder = RESTRICTION_DIR / subdir
            if not folder.exists():
                continue
            for fpath in sorted(folder.glob("*.json")):
                try:
                    rec = json.loads(fpath.read_text(encoding="utf-8"))
                    rec["type"]    = rtype
                    rec.setdefault("enabled", True)
                    self._restrictions.append(rec)
                    self._restrict_list.addItem(self._restriction_label(rec))
                except Exception as e:
                    print(f"[Restrictions] Could not load {fpath.name}: {e}")

    def _on_map_loaded(self, ok: bool):
        """Called once the Leaflet page has fully loaded — safe to run JS now."""
        if not ok:
            return
        for rec in self._restrictions:
            self._draw_restriction_on_map(rec)
        n = len(self._restrictions)
        if n:
            self._status.showMessage(
                f"Loaded {n} restriction(s) from disk — verifying geometry against OSM…"
            )
            self._start_geometry_verification()

    def _start_geometry_verification(self):
        """Launch background thread to detect OSM geometry changes for all RC restrictions."""
        rc_recs = [r for r in self._restrictions if r.get("type") == "RC"
                   and r.get("geometry")]
        if not rc_recs:
            return
        self._verify_thread = VerifyGeometryThread(rc_recs)
        self._verify_thread.mismatch_found.connect(self._on_geometry_mismatch)
        self._verify_thread.finished_all.connect(self._on_verify_finished)
        self._verify_thread.start()

    def _on_geometry_mismatch(self, rec_id: str, way_id: int, reason: str):
        """Called for each restriction way whose OSM geometry has changed."""
        # Store mismatch info on the record (in-memory only, not persisted)
        for rec in self._restrictions:
            if rec.get("id") == rec_id:
                rec.setdefault("_mismatches", {})[str(way_id)] = reason
                break
        # Refresh list so the ⚠️ badge appears immediately
        self._refresh_restriction_list()
        print(f"[GeomVerify] ⚠ {rec_id} way {way_id}: {reason}")

    def _on_verify_finished(self, checked: int, mismatches: int):
        """Called when geometry verification is complete."""
        if mismatches == 0:
            self._status.showMessage(
                f"Geometry check: all {checked} way(s) match current OSM ✓"
            )
        else:
            self._status.showMessage(
                f"⚠ {mismatches} restriction way(s) differ from current OSM — "
                f"right-click the highlighted rule(s) to refresh geometry"
            )

    # ── Restriction helpers ────────────────────────────────────────────
    def _is_restriction_active(self, rec: dict) -> bool:
        """True if restriction is enabled AND current datetime is in its time window."""
        from datetime import datetime
        if not rec.get("enabled", True):
            return False
        now = datetime.now()
        # Date validity
        try:
            d_create = datetime.strptime(rec.get("date_create", "2000-01-01"), "%Y-%m-%d").date()
            d_end    = datetime.strptime(rec.get("date_end",    "9999-01-01"), "%Y-%m-%d").date()
            if not (d_create <= now.date() <= d_end):
                return False
        except ValueError:
            pass
        # Time-of-day window
        try:
            fmt = "%H:%M"
            t_s = datetime.strptime(rec.get("time_start", "00:00"), fmt).time()
            t_e = datetime.strptime(rec.get("time_end",   "23:59"), fmt).time()
            cur = now.time().replace(second=0, microsecond=0)
            if not (t_s <= cur <= t_e):
                return False
        except ValueError:
            pass
        return True

    def _restriction_label(self, rec: dict) -> str:
        enabled = rec.get("enabled", True)
        active  = self._is_restriction_active(rec)
        if not enabled:
            status = "🔴"   # manually disabled
        elif active:
            status = "🟢"   # enabled & within time window
        else:
            status = "🟡"   # enabled but outside time window right now

        type_icon = "🚧" if rec.get("type") == "RC" else "↩"

        # Direction badge (RC only)
        if rec.get("type") == "RC":
            _dir = rec.get("direction", "2way")
            dir_badge = "→" if _dir == "1way" else "←" if _dir == "1way_reverse" else "↔"
        else:
            dir_badge = ""

        rid   = rec.get("id", "?")
        name  = rec.get("name", "")
        vtype = rec.get("vehicle_type", "")
        dt_s  = rec.get("dt_start") or f"{rec.get('date_create','')} {rec.get('time_start','00:00')}"
        dt_e  = rec.get("dt_end")   or f"{rec.get('date_end','')} {rec.get('time_end','23:59')}"
        dir_part = f" {dir_badge}" if dir_badge else ""
        # ⚠️ geometry mismatch badge (in-memory flag set by VerifyGeometryThread)
        mismatch_badge = " ⚠" if rec.get("_mismatches") else ""
        return f"{status}{mismatch_badge} {type_icon}{dir_part} {rid}  {name}  [{vtype}]  {dt_s}→{dt_e}"

    def _refresh_restriction_list(self):
        """Rebuild the list widget labels from current _restrictions state."""
        self._restrict_list.clear()
        for rec in self._restrictions:
            self._restrict_list.addItem(self._restriction_label(rec))

    # ── Double-click → zoom to restriction ────────────────────────────
    def _zoom_to_restriction_item(self, item):
        row = self._restrict_list.row(item)
        if row < 0 or row >= len(self._restrictions):
            return
        rec = self._restrictions[row]
        all_coords: list[list[float]] = []

        if rec["type"] == "RC":
            for latlngs in rec.get("geometry", {}).values():
                all_coords.extend(latlngs)
        else:  # TR
            for coords in rec.get("node_coords", {}).values():
                all_coords.append(coords)

        if not all_coords:
            self._status.showMessage(f"{rec.get('id','?')}: no geometry yet — still fetching?")
            return

        if len(all_coords) == 1:
            lat, lon = all_coords[0]
            self._js(f"map.setView([{lat}, {lon}], 18)")
        else:
            self._js(f"map.fitBounds({json.dumps(all_coords)}, {{padding:[50,50]}})")
        self._status.showMessage(f"Zoomed to {rec.get('id','?')}")

    # ── Right-click context menu ───────────────────────────────────────
    def _restriction_context_menu(self, pos):
        row = self._restrict_list.currentRow()
        if row < 0 or row >= len(self._restrictions):
            return
        rec = self._restrictions[row]

        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu{background:#313244;color:#cdd6f4;border:1px solid #45475a;"
            "border-radius:6px;padding:4px;}"
            "QMenu::item{padding:5px 18px;border-radius:4px;}"
            "QMenu::item:selected{background:#45475a;}"
            "QMenu::separator{height:1px;background:#45475a;margin:3px 0;}"
        )

        enabled = rec.get("enabled", True)
        if enabled:
            act_toggle = menu.addAction("🔴  Disable restriction")
        else:
            act_toggle = menu.addAction("🟢  Enable restriction")

        act_edit = menu.addAction("✏️  Edit restriction")
        menu.addSeparator()

        # Show refresh option if a geometry mismatch was detected
        act_refresh = None
        if rec.get("_mismatches") and rec.get("type") == "RC":
            mismatches = rec["_mismatches"]
            detail = "  |  ".join(
                f"way {wid}: {reason}"
                for wid, reason in mismatches.items()
            )
            act_refresh = menu.addAction(f"🔄  Refresh geometry from OSM")
            act_refresh.setToolTip(detail)
            menu.addSeparator()

        act_zoom = menu.addAction("🔍  Zoom to on map")
        menu.addSeparator()
        act_remove = menu.addAction("🗑  Remove from session")

        chosen = menu.exec(self._restrict_list.viewport().mapToGlobal(pos))
        if chosen == act_toggle:
            self._toggle_restriction(row, not enabled)
        elif chosen == act_edit:
            self._edit_restriction(row)
        elif act_refresh and chosen == act_refresh:
            self._refresh_restriction_geometry(row)
        elif chosen == act_zoom:
            self._zoom_to_restriction_item(self._restrict_list.item(row))
        elif chosen == act_remove:
            self._restrict_list.setCurrentRow(row)
            self._remove_restriction()

    def _toggle_restriction(self, row: int, enabled: bool):
        if row < 0 or row >= len(self._restrictions):
            return
        rec = self._restrictions[row]
        rec["enabled"] = enabled
        self._save_restriction_to_disk(rec)
        self._refresh_restriction_list()
        state = "enabled 🟢" if enabled else "disabled 🔴"
        self._status.showMessage(f"{rec.get('id','?')} {state}")

    def _refresh_restriction_geometry(self, row: int):
        """Re-fetch geometry from Overpass for a restriction whose OSM way was modified.

        Clears the stored geometry and mismatch flags, then re-fetches fresh node
        data from Overpass — exactly as if the restriction had just been created.
        The new geometry is saved to disk once all ways have been fetched.
        """
        if row < 0 or row >= len(self._restrictions):
            return
        rec = self._restrictions[row]
        if rec.get("type") != "RC":
            return

        rid = rec.get("id", "?")

        # Remove old overlays from the map
        for way_id in rec.get("way_ids", []):
            self._js(f"removeRestrictedWay({way_id})")
        self._js(f"removeRestrictedPolygon({json.dumps(rid)})")

        # Clear stored geometry and mismatch flags
        rec["geometry"] = {}
        rec.pop("_mismatches", None)

        # Invalidate the per-session Overpass cache so fresh data is fetched
        from benchmark_engines.valhalla_engine import _WAY_GEOMETRY_CACHE
        for way_id in rec.get("way_ids", []):
            _WAY_GEOMETRY_CACHE.pop(int(way_id), None)

        self._refresh_restriction_list()
        self._status.showMessage(
            f"{rid} — re-fetching geometry from Overpass…"
        )

        # Fetch fresh geometry (same path as create)
        self._fetch_restriction_geometry(rec)

    def _edit_restriction(self, row: int):
        """Open the restriction dialog pre-filled with existing data for editing."""
        if row < 0 or row >= len(self._restrictions):
            return
        rec = self._restrictions[row]

        dlg = CreateRestrictionDialog(rec["type"], parent=self, prefill=rec)
        if dlg.exec() != CreateRestrictionDialog.DialogCode.Accepted:
            return

        updated = dlg.result_data
        old_way_ids  = set(str(w) for w in rec.get("way_ids",  []))
        new_way_ids  = set(str(w) for w in updated.get("way_ids",  []))
        old_node_ids = rec.get("node_ids",  [])
        new_node_ids = updated.get("node_ids", [])

        # Preserve identity fields
        updated["id"]      = rec["id"]
        updated["type"]    = rec["type"]
        updated["enabled"] = rec.get("enabled", True)

        if rec["type"] == "RC":
            if new_way_ids == old_way_ids:
                # Same ways — keep existing geometry
                updated["geometry"] = rec.get("geometry", {})
            else:
                # Ways changed — clear geometry and re-fetch
                updated["geometry"] = {}

        else:  # TR
            if new_node_ids == old_node_ids:
                updated["node_coords"] = rec.get("node_coords", {})
            else:
                updated["node_coords"] = {}

        # Replace the record in-place
        self._restrictions[row] = updated
        self._save_restriction_to_disk(updated)

        # Redraw on map (remove old, draw new)
        if rec["type"] == "RC":
            for way_id in rec.get("way_ids", []):
                self._js(f"removeRestrictedWay({way_id})")
        else:
            rid = rec.get("id", "")
            self._js(f"removeRestrictedWay('tr_{rid}')")
        self._js(f"removeRestrictedPolygon({json.dumps(rec.get('id','?'))})")

        self._refresh_restriction_list()
        self._draw_restriction_on_map(updated)

        # If way_ids / node_ids changed, re-fetch geometry
        if rec["type"] == "RC" and new_way_ids != old_way_ids:
            self._fetch_restriction_geometry(updated)
            self._status.showMessage(
                f"{updated['id']} updated — fetching new geometry from Overpass…"
            )
        elif rec["type"] == "TR" and new_node_ids != old_node_ids:
            self._fetch_restriction_geometry(updated)
            self._status.showMessage(
                f"{updated['id']} updated — fetching new node coords from Overpass…"
            )
        else:
            self._status.showMessage(f"{updated['id']} updated")

    def _create_restriction(self, rtype: str):
        dlg = CreateRestrictionDialog(rtype, parent=self)
        if dlg.exec() != CreateRestrictionDialog.DialogCode.Accepted:
            return
        rec = dlg.result_data

        # Auto-generate ID: RC_01, RC_02 … or TR_01, TR_02 …
        prefix = rtype
        existing_ids = [
            r.get("id", "") for r in self._restrictions
            if r.get("type") == rtype
        ]
        num = 1
        while f"{prefix}_{num:02d}" in existing_ids:
            num += 1
        rec["id"]      = f"{prefix}_{num:02d}"
        rec["enabled"] = True

        self._restrictions.append(rec)
        self._restrict_list.addItem(self._restriction_label(rec))
        self._status.showMessage(
            f"Created {rec['id']} — fetching geometry from Overpass…"
        )

        # Fetch geometry in background, then save to disk
        self._fetch_restriction_geometry(rec)

    def _fetch_restriction_geometry(self, rec: dict):
        """Start background thread(s) to fetch geometry for a new restriction."""
        if rec["type"] == "RC":
            for way_id in rec.get("way_ids", []):
                thread = FetchWayThread(int(way_id))
                thread.done.connect(
                    lambda wid, lj, r=rec: self._on_rc_way_fetched(r, wid, lj)
                )
                self._fetch_threads.append(thread)
                thread.start()
        else:  # TR
            node_ids = rec.get("node_ids", [])
            if node_ids:
                thread = FetchNodeThread(rec["id"], node_ids)
                thread.done.connect(self._on_tr_nodes_fetched)
                self._fetch_threads.append(thread)
                thread.start()

    def _on_rc_way_fetched(self, rec: dict, way_id: int, latlngs_json: str):
        """Called when one way's geometry arrives for an RC restriction."""
        if latlngs_json:
            latlngs = json.loads(latlngs_json)
            rec.setdefault("geometry", {})[str(way_id)] = latlngs
            direction = rec.get("direction", "2way")
            self._js(f"addRestrictedWay({way_id}, {json.dumps(latlngs_json)}, {json.dumps(direction)})")

        # Check if all ways have been fetched
        fetched = set(rec.get("geometry", {}).keys())
        all_ways = {str(w) for w in rec.get("way_ids", [])}
        if fetched >= all_ways:
            self._save_restriction_to_disk(rec)
            self._status.showMessage(
                f"{rec['id']} saved — geometry for {len(all_ways)} way(s) stored"
            )

    def _on_tr_nodes_fetched(self, restriction_id: str, coords_json: str):
        """Called when node coordinates arrive for a TR restriction."""
        rec = next((r for r in self._restrictions if r.get("id") == restriction_id), None)
        if rec is None:
            return
        if coords_json:
            coords = json.loads(coords_json)
            rec["node_coords"] = coords
            # Draw via node on map
            node_ids = rec.get("node_ids", [])
            if len(node_ids) >= 3:
                via_key = str(node_ids[1])
                if via_key in coords:
                    lat, lon = coords[via_key]
                    self._js(
                        f"addRestrictedWay('tr_{restriction_id}', "
                        f"{json.dumps([[lat, lon]])})"
                    )
        self._save_restriction_to_disk(rec)
        self._status.showMessage(f"{restriction_id} saved — node coords stored")

    def _save_restriction_to_disk(self, rec: dict):
        """Persist a restriction record as JSON file."""
        subdir = "Road closure" if rec["type"] == "RC" else "Turn restriction"
        folder = RESTRICTION_DIR / subdir
        folder.mkdir(parents=True, exist_ok=True)
        fpath = folder / f"{rec['id']}.json"
        fpath.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")

    @property
    def _buffer_m(self) -> float:
        return self._buf_spinbox.value()

    @property
    def _buffer_deg(self) -> float:
        return self._buffer_m / 111_320.0   # approx degrees (good enough near equator)

    def _on_buffer_changed(self, _val: int):
        """Redraw all restriction buffer polygons when the slider changes."""
        self._js("clearRestrictedPolygons()")
        for rec in self._restrictions:
            self._draw_buffer_polygon(rec)

    def _draw_restriction_on_map(self, rec: dict):
        """Draw a restriction's geometry (dashed line + buffer polygon) on the map."""
        if rec["type"] == "RC":
            geom      = rec.get("geometry", {})
            direction = rec.get("direction", "2way")
            for way_id, latlngs in geom.items():
                lj = json.dumps(latlngs)
                self._js(f"addRestrictedWay({way_id}, {json.dumps(lj)}, {json.dumps(direction)})")
        else:  # TR — draw via node
            node_ids    = rec.get("node_ids", [])
            node_coords = rec.get("node_coords", {})
            if len(node_ids) >= 3:
                via_key = str(node_ids[1])
                coords  = node_coords.get(via_key)
                if coords:
                    lat, lon = coords[0], coords[1]
                    rid = rec.get("id", "tr")
                    self._js(
                        f"addRestrictedWay('tr_{rid}', "
                        f"{json.dumps([[lat, lon]])})"
                    )
        self._draw_buffer_polygon(rec)

    def _draw_buffer_polygon(self, rec: dict):
        """Compute and draw the orange buffer zone polygon on the map.

        RC: one polygon per way using INTERIOR nodes only (latlngs[1:-1]).
            Endpoint/junction nodes are shared with cross-streets and
            roundabouts — including them would make the orange overlay bleed
            into those junction areas (same rule as exclude_locations).
            Ways with ≤2 nodes (no interior) are skipped.
        TR: buffer around all via-node coordinates (no junction issue).
        """
        from benchmark_engines.valhalla_engine import _line_buffer_polygon
        rid = rec.get("id", "?")

        if rec["type"] == "RC":
            # Build one polygon per way using interior nodes only
            all_rings = []
            for latlngs in rec.get("geometry", {}).values():
                interior = latlngs[1:-1]   # skip endpoint junction nodes
                if len(interior) < 2:
                    continue               # 2-node way — nothing to draw
                nodes = [(pt[0], pt[1]) for pt in interior]
                poly = _line_buffer_polygon(nodes, self._buffer_deg)
                if poly:
                    all_rings.append(poly["coordinates"][0])
            for ring in all_rings:
                self._js(f"addRestrictedPolygon({json.dumps(rid)}, {json.dumps(ring)})")
        else:  # TR — buffer around all 3 nodes (via-node cluster)
            all_nodes: list[tuple[float, float]] = [
                (coords[0], coords[1])
                for coords in rec.get("node_coords", {}).values()
            ]
            if len(all_nodes) < 2:
                return
            poly = _line_buffer_polygon(all_nodes, self._buffer_deg)
            if not poly:
                return
            ring = poly["coordinates"][0]   # [[lon, lat], ...]
            self._js(f"addRestrictedPolygon({json.dumps(rid)}, {json.dumps(ring)})")

    def _remove_restriction(self):
        row = self._restrict_list.currentRow()
        if row < 0:
            self._status.showMessage("Select a restriction row first")
            return
        rec = self._restrictions.pop(row)
        self._restrict_list.takeItem(row)

        # Remove from map
        if rec["type"] == "RC":
            for way_id in rec.get("way_ids", []):
                self._js(f"removeRestrictedWay({way_id})")
        else:
            rid = rec.get("id", "")
            self._js(f"removeRestrictedWay('tr_{rid}')")

        self._status.showMessage(f"Removed restriction {rec.get('id','?')} from session "
                                 f"(file on disk unchanged)")

    def _clear_restrictions(self):
        self._restrictions.clear()
        self._restrict_list.clear()
        self._js("clearRestrictedWays()")
        self._status.showMessage("All restrictions cleared from session (files on disk unchanged)")

    def _export_osm_ids(self):
        row = self._hist_table.currentRow()
        if row < 0 or row >= len(self._trip_history):
            self._status.showMessage("Select a history row first, then click Export OSM IDs")
            return

        entry = self._trip_history[row]
        osm_ids = entry.get("osm_way_ids", [])
        if not osm_ids:
            self._status.showMessage("No OSM way IDs for this trip (Valhalla engine required)")
            return

        ts_safe = entry["ts"].replace(":", "-").replace(" ", "_")
        default_name = f"osm_ways_{ts_safe}_{entry['mode']}"

        path, fmt = QFileDialog.getSaveFileName(
            self,
            "Export OSM Way IDs",
            default_name,
            "JSON (*.json);;CSV (*.csv)",
        )
        if not path:
            return

        sl, slo = entry["start"]
        el, elo = entry["end"]
        meta = {
            "trip_time": entry["ts"],
            "mode":      entry["mode"],
            "start":     {"lat": sl, "lon": slo},
            "end":       {"lat": el, "lon": elo},
            "engine":    entry["best"].engine if entry["best"] else "unknown",
            "way_count": len(osm_ids),
        }

        if path.endswith(".csv"):
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["osm_way_id"])
                for wid in osm_ids:
                    w.writerow([wid])
            self._status.showMessage(f"Exported {len(osm_ids)} way IDs → {os.path.basename(path)}")
        else:
            if not path.endswith(".json"):
                path += ".json"
            payload = {**meta, "osm_way_ids": osm_ids}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            self._status.showMessage(f"Exported {len(osm_ids)} way IDs → {os.path.basename(path)}")

    def _clear_history(self):
        self._trip_history.clear()
        self._hist_table.setRowCount(0)
        self._hist_count_lbl.setText("0 trips")
        self._status.showMessage("Trip history cleared")

    # ── Basemap ────────────────────────────────────────────────────────
    def _change_basemap(self, name: str):
        self._js(f"setBasemap({json.dumps(name)})")

    # ── Style helpers ──────────────────────────────────────────────────
    def _group(self, title: str) -> QGroupBox:
        g = QGroupBox(title)
        g.setStyleSheet(
            "QGroupBox{color:#cba6f7;font-weight:600;border:1px solid #313244;"
            "border-radius:6px;margin-top:8px;padding-top:4px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;}"
        )
        g.setLayout(QVBoxLayout())
        g.layout().setContentsMargins(8, 12, 8, 8)
        g.layout().setSpacing(4)
        return g

    def _button(self, text: str, accent: bool = False) -> QPushButton:
        btn = QPushButton(text)
        if accent:
            btn.setStyleSheet(
                "QPushButton{background:#cba6f7;color:#1e1e2e;border-radius:6px;"
                "padding:6px;font-weight:700;}"
                "QPushButton:hover{background:#d4b8fb;}"
                "QPushButton:disabled{background:#45475a;color:#6c7086;}"
            )
        else:
            btn.setStyleSheet(
                "QPushButton{background:#313244;color:#cdd6f4;border-radius:6px;"
                "padding:5px;}"
                "QPushButton:hover{background:#45475a;}"
            )
        return btn

    def _combo_style(self) -> str:
        return (
            "QComboBox{background:#313244;color:#cdd6f4;border-radius:4px;"
            "padding:4px;border:none;}"
            "QComboBox::drop-down{border:none;}"
            "QComboBox QAbstractItemView{background:#313244;color:#cdd6f4;"
            "selection-background-color:#45475a;}"
        )
