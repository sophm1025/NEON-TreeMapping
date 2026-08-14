import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ===== User parameters =====
csv_path = "/work/zl465/Vegetation_struture/MLBS/MLBS_merged.csv"   # Input CSV path
shp_path = "/work/zl465/Vegetation_struture/MLBS/MLBS_merged.shp"   # Output Shapefile path
utm_epsg = 32617                                                    # UTM Zone 17N EPSG code

# 1. Read CSV
df = pd.read_csv(csv_path)

# 2. Filter rows with height >= 5
df_filtered = df[df["height"] >= 5].copy()

# 3. Create geometry objects
geometry = [Point(xy) for xy in zip(df_filtered["adjEasting"], df_filtered["adjNorthing"])]

# 4. Create GeoDataFrame
gdf = gpd.GeoDataFrame(df_filtered, geometry=geometry, crs=f"EPSG:{utm_epsg}")

# 5. Save as Shapefile
gdf.to_file(shp_path)

print(f"Shapefile has been generated: {shp_path}")
