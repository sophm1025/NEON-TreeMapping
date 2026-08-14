import os
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
import geopandas as gpd
from tqdm import tqdm
import numpy as np
from rasterio.windows import Window
import multiprocessing as mp
import pandas as pd

# ===== 输入与输出 =====
tif_file = "/storage/group/tvq5043/default/Zhuohong/NEON_data/Preprocess/MLBS/refined_globalID_IOU038_MinA65_0828.tif"
shp_file = "/storage/group/tvq5043/default/Zhuohong/NEON_data/Preprocess/MLBS/polygon_tree_crown_withID/refined_globalID_IOU038_MinA65_0828.shp"
min_area = 2*2  # 最小多边形面积（像素^2）
tile_size = 2000  # 分块大小
tmp_dir = "./tmp_tiles"  # 临时文件夹存 GeoJSON

os.makedirs(tmp_dir, exist_ok=True)

# ===== 处理单个分块的函数 =====
def process_tile(window, idx, tif_file, min_area, tmp_dir):
    out_path = os.path.join(tmp_dir, f"tile_{idx}.geojson")
    with rasterio.open(tif_file) as src:
        tile_data = src.read(1, window=window)
        if tile_data.dtype == 'uint32':
            tile_data = tile_data.astype('int32')
        tile_transform = src.window_transform(window)
        polygons = []
        for geom, value in shapes(tile_data, mask=tile_data > 0, transform=tile_transform):
            poly = shape(geom)
            if poly.area >= min_area:
                polygons.append({
                    "geometry": mapping(poly),
                    "properties": {"tree_id": int(value)}
                })
        if polygons:
            gpd.GeoDataFrame.from_features(polygons, crs=src.crs).to_file(out_path, driver="GeoJSON")
    return out_path

# ===== 处理带索引的分块，适合 multiprocessing =====
def process_tile_idx(args):
    idx, window = args
    return process_tile(window, idx, tif_file, min_area, tmp_dir)

# ===== 主流程 =====
def main():
    # TIFF 信息
    with rasterio.open(tif_file) as src:
        crs = src.crs
        height, width = src.height, src.width
        dtype = src.dtypes[0]
        print(f"TIFF 数据类型: {dtype}, 尺寸: {width}x{height} 像素")

    # 生成分块窗口和索引
    windows = [
        Window(col_off, row_off, min(tile_size, width - col_off), min(tile_size, height - row_off))
        for row_off in range(0, height, tile_size)
        for col_off in range(0, width, tile_size)
    ]
    windows_idx = list(enumerate(windows))

    # 多进程处理
    pool = mp.Pool(processes=4)  # 可根据系统调整
    tmp_files = []
    for out_file in tqdm(pool.imap_unordered(process_tile_idx, windows_idx),
                         total=len(windows_idx),
                         desc="处理分块", unit="tile"):
        tmp_files.append(out_file)
    pool.close()
    pool.join()

    # 合并所有临时 GeoJSON
    all_gdfs = []
    for tmp_file in tqdm(tmp_files, desc="合并临时文件"):
        if os.path.exists(tmp_file):
            gdf = gpd.read_file(tmp_file)
            all_gdfs.append(gdf)

    if all_gdfs:
        final_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs=crs)
        final_gdf.to_file(shp_file)
        print(f"✅ 成功将多边形写入 {shp_file}")
    else:
        print("⚠️ 没有生成任何多边形")

if __name__ == "__main__":
    mp.set_start_method("spawn")  # 避免 fork 方式问题
    main()
