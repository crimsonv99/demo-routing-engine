"""
Routing Benchmark Desktop Tool
================================
Compare NetworkX, Valhalla, GraphHopper, and OSRM side by side on a Leaflet map.

Usage
-----
  cd /path/to/prod
  pip install PyQt6 PyQt6-WebEngine polyline httpx
  python benchmark_tool/routing_tool.py

Requirements
------------
  - PyQt6 + PyQt6-WebEngine  (pip install PyQt6 PyQt6-WebEngine)
  - polyline                 (pip install polyline)
  - httpx                    (pip install httpx)
  - geopandas, shapely, networkx, pyproj  (already in your venv)

Engines
-------
  NetworkX     — in-process, requires road network file + ~14 min build
  Valhalla     — Docker on localhost:8002  (docker-compose up valhalla)
  GraphHopper  — Docker/JAR on localhost:8989
  OSRM         — Docker on localhost:5000
"""
import sys
import os

# ── Ensure prod/ is on the path so routing_engine.py can be imported ──
_PROD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROD_DIR not in sys.path:
    sys.path.insert(0, _PROD_DIR)

# ── Ensure benchmark_tool/ is on the path for benchmark_* packages ────
_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore    import Qt
from benchmark_ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Routing Benchmark Tool")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
