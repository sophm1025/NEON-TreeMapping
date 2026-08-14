import os
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping
from tqdm import tqdm  # Progress bar module

# ========================
# Optional parameters
# ========================

# Whether to select only the first N bands (if None, use all bands)
SELECT_FIRST_N_BANDS = None  # e.g., set to 100 to use only the first 100 bands

# Input paths
shp_path = "/storage/group/tvq5043/default/Zhuohong/NEON_data/Preprocess/MLBS/Support_data/MLBS_circle.shp"
hsi_tiff_path = "/storage/group/tvq5043/default/Zhuohong/NEON_data/original_NEON_Data/MLBS2022/Refl002_MLBS_allbands_intersection.tif"
chm_tiff_path = "/storage/group/tvq5043/default/Zhuohong/NEON_data/original_NEON_Data/MLBS2022/CHM_MLBS_intersection.tif"  
output_folder = "/storage/group/tvq5043/default/Zhuohong/NEON_data/Preprocess/MLBS/training_sample_hsi-chm/"

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Read vector boundaries
gdf = gpd.read_file(shp_path)

# List to save CSV records
csv_records = []

# Open hyperspectral (HSI) and CHM images
with rasterio.open(hsi_tiff_path) as hsi_src, rasterio.open(chm_tiff_path) as chm_src:
    hsi_nodata = hsi_src.nodata  # Get HSI nodata value
    chm_nodata = chm_src.nodata  # Get CHM nodata value
    hsi_band_count = hsi_src.count

    # Ensure CHM has only one band
    if chm_src.count != 1:
        raise ValueError("CHM data should have only one band!")

    # Decide whether to select only the first N HSI bands
    if SELECT_FIRST_N_BANDS is not None:
        selected_bands = list(range(1, min(SELECT_FIRST_N_BANDS, hsi_band_count) + 1))  # 1-based index
    else:
        selected_bands = None  # Use all bands

    # Iterate with progress bar
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Clipping images"):
        geom = row.geometry
        individual_id = row['individual']
        properties = row.drop(labels='geometry').to_dict()

        # Clip HSI data
        if selected_bands:
            hsi_image, hsi_transform = mask(hsi_src, [mapping(geom)], crop=True, nodata=hsi_nodata, indexes=selected_bands)
        else:
            hsi_image, hsi_transform = mask(hsi_src, [mapping(geom)], crop=True, nodata=hsi_nodata)

        # Clip CHM data
        chm_image, chm_transform = mask(chm_src, [mapping(geom)], crop=True, nodata=chm_nodata)

        # Ensure CHM and HSI have the same spatial resolution and shape
        if hsi_image.shape[1:] != chm_image.shape[1:]:
            raise ValueError(f"HSI and CHM clipped shapes do not match: HSI {hsi_image.shape}, CHM {chm_image.shape}")

        if hsi_transform != chm_transform:
            print(f"Warning: {individual_id}'s HSI and CHM transform matrices differ, which may affect spatial alignment!")

        # Concatenate CHM as the first channel with HSI
        combined_image = np.concatenate([chm_image, hsi_image], axis=0)

        # Update metadata
        out_meta = hsi_src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": combined_image.shape[1],
            "width": combined_image.shape[2],
            "transform": hsi_transform,
            "nodata": hsi_nodata,  # Use HSI nodata value (adjust if needed)
            "compress": "lzw",
            "count": combined_image.shape[0]  # Update number of bands (CHM + HSI bands)
        })

        # Output TIFF path
        out_tif_path = os.path.join(output_folder, f"{individual_id}.tif")
        with rasterio.open(out_tif_path, "w", **out_meta) as dest:
            dest.write(combined_image)

        # Save attribute information
        record = {"image_name": f"{individual_id}.tif"}
        record.update(properties)
        csv_records.append(record)

# Save CSV records
csv_df = pd.DataFrame(csv_records)
csv_df.to_csv(os.path.join(output_folder, "attributes.csv"), index=False)

print("✅ Clipping and concatenation completed, CHM+HSI images and attribute table have been saved!")
