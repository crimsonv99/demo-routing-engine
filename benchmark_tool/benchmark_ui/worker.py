from __future__ import annotations
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from PyQt6.QtCore import QThread, pyqtSignal

from benchmark_engines.base import BaseEngine, RouteResult


class LoadWorker(QThread):
    """Builds the NetworkX graph in a background thread."""
    finished  = pyqtSignal()
    error     = pyqtSignal(str)
    status    = pyqtSignal(str)

    def __init__(self, engine: BaseEngine, network_path: str,
                 restrictions_path: str = None):
        super().__init__()
        self.engine           = engine
        self.network_path     = network_path
        self.restrictions_path = restrictions_path

    def run(self):
        try:
            self.status.emit(f"Loading {self.engine.NAME} …")
            self.engine.load(self.network_path, self.restrictions_path)
            self.finished.emit()
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))


class RouteWorker(QThread):
    """Runs selected engines in parallel and emits each result as it arrives."""
    result_ready = pyqtSignal(object)   # RouteResult
    all_done     = pyqtSignal()

    def __init__(self, engines: list[BaseEngine],
                 start_lat: float, start_lon: float,
                 end_lat:   float, end_lon:   float,
                 mode: str = "car",
                 restrictions: list[dict] | None = None,
                 buffer_deg: float = 0.00009,
                 via_points: list[tuple[float, float]] | None = None):
        super().__init__()
        self.engines      = engines
        self.start_lat    = start_lat
        self.start_lon    = start_lon
        self.end_lat      = end_lat
        self.end_lon      = end_lon
        self.mode         = mode
        self.restrictions = restrictions or []
        self.buffer_deg   = buffer_deg
        self.via_points   = via_points or []

    def run(self):
        available = [e for e in self.engines if e.is_available()]
        if not available:
            self.all_done.emit()
            return

        with ThreadPoolExecutor(max_workers=len(available)) as pool:
            futures = {
                pool.submit(
                    e.route,
                    self.start_lat, self.start_lon,
                    self.end_lat,   self.end_lon,
                    self.mode,
                    self.restrictions or None,
                    self.buffer_deg,
                    self.via_points or None,
                ): e
                for e in available
            }
            for future in as_completed(futures):
                try:
                    result: RouteResult = future.result()
                except Exception as exc:
                    eng = futures[future]
                    result = RouteResult(eng.NAME, eng.COLOR, 0, 0, [],
                                        error=str(exc))
                self.result_ready.emit(result)

        self.all_done.emit()
