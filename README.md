# NeON tree mapping

## Overview
This repository provides scripts to download, preprocess, and map NeON tree crowns using CHM and RGB data.

---

## Step 1: Data download
- **GEE data**  
  Use **`GEE_download_ref002_chm.py`** to download canopy height model (CHM) data from Google Earth Engine.  
  > *Note: this script still requires further modification by Georgiy.*

- **NeON structure data**  
  Use **`compile_crown_area.R`** to download NeON structure data from the NeON database.  
  Provide the site name as the input parameter.
  
After downloading, the data will be organized as follows:
1. **Imagery**  
   HSI, RGB, and CHM data:  
   ```
   /storage/group/tvq5043/default/Zhuohong/NEON_data/original_NEON_Data/{SITE}{YEAR}
   ```
2. **Field-collected samples**  
   Combined crowns CSV:  
   ```
   /storage/group/tvq5043/default/Zhuohong/NEON_mapping/{SITE}/{SITE}_tree_data.csv
   ```
---
## Step 2: Data processing
1. **Generate rough tree-crown boundaries (CHM, 1 m)**  
   Run:
   ```bash
   python 01_create_treecrown_chm.py --site <SITE> --year <YEAR>
   ```
   *Default year is 2022.*  
   This produces an initial (rough) tree crown map for later refinement.

2. **Refine tree-crown boundaries (RGB, 0.1 m)**  
   *Train a Mask-RCNN model using RGB images and the rough crown map:
   ```bash
   python 02_refine_treecrown_DL_rgb.py --site <SITE> --year <YEAR>
   ```
   Predict refined tree crowns:
   ```bash
   python 03_01_predict_treecrown.py --site <SITE> --year <YEAR>
   ```
   The prediction tiles are about 3072*3072, while a site is about 14k*14k.
   Therefore, we need to merge the tiles back to a site.
   ```bash
   python 03_02_mergeback_to_site.py --site <SITE> --year <YEAR>
   ```
   Ultimately, we convert the refined tree crown from .tiff format to .gpkg format.
   ```bash
   python python 04_trasfer_treecrownTif_to_gpkg.py --site <SITE> --year <YEAR>
   ``` 
## Step 3: Generate training pool
:star2:I: I already grouped the following 1-3 into one script:
Just run    
  ```bash
   python 00_generate_training_sample.py --site <SITE> --year <YEAR>
   ```
:star2:If you want to know each detailed information, read the information below:
1. **Convert training samples from CSV to shapefile** 
   Generate point shapefiles for training samples:
   ```bash
   python 01_generate_point_shp_from_csv.py --site <SITE> --year <YEAR>
   ```
2. **Filter training sample points**  
  The vegetation structure data may contain a large number of trees in a single crown, which can be observed in overhead images.
  Therefore, we need to filter training points based on their height, crown size, and other factors.
   ```bash
   python 02_select_trainpoint.py --site <SITE> --year <YEAR>
   ```
3. **Generate training pool and plot spectral curves**  
   After filtering training points, we crop a 4*4 area around each point.
   ```bash
   python 03_genenrate_samplepool.py --site <SITE> --year <YEAR>
   ```
   Plot the taxon and spectral curve based on the sample pool (optional)
   ```bash
   python 04_taxon_spectral_plot.py --site <SITE> --year <YEAR>
   ```
## Step 4: Train tree species classification model  (vision transformer)
1. **Run Train script**  
   ```bash
   python 05_train.py --site <SITE> --year <YEAR>
   ```
## Step 5: Inference tree species map with the well-trained model
1. **Generate the center point of each tree crown among the site**  
   ```bash
   python 06_generate_center_point_from_gpkg.py --site <SITE> --year <YEAR>
   ```
2. **Run inference script**  
   ```bash
   python 07_inference_full_site.py --site <SITE> --year <YEAR>
   ```
## Step 6: Evaluate the accuracy   
---

## Notes
- Replace `<SITE>` with the NeON site name (e.g., `SERC`).
- Replace `<YEAR>` if needed; defaults to `2022`.
