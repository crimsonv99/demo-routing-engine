from __future__ import annotations
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def load_pois_csv(path: str) -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["poi_longitude"], df["poi_latitude"])],
        crs="EPSG:4326"
    )
    return gdf