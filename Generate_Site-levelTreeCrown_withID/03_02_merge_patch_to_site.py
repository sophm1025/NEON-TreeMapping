import os
import time
import logging
from collections import defaultdict

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import psutil

# ================= Configuration =================
RGB_TIF = "/storage/group/tvq5043/default/Zhuohong/NEON_data/RGB_MLBS_intersection.tif"
OUTPUT_MASK = "/storage/group/tvq5043/default/Zhuohong/NEON_data/Preprocess/MLBS/refined_globalID_IOU038_MinA65_0828.tif"
TEMP_PATCH_DIR = "/storage/group/tvq5043/default/Zhuohong/NEON_data/Preprocess/MLBS/temp_patches"

PATCH_SIZE = 800
OVERLAP = 128
IOU_THRESHOLD = 0.38          # Overlap threshold for merging
MIN_AREA = 100                 # Ignore very small objects

# Global ID tracking
global_id = 1
id_mapping = {}         # (row_idx, col_idx, local_id) -> global_id
overlap_pairs = []      # [(gid1, gid2), ...]

# Set up logging
logging.basicConfig(filename='merge.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# ========== Compute IOU ==========
def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

# ========== Merge Global IDs ==========
def merge_global_ids(overlap_pairs):
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    # Merge overlapping IDs
    for id1, id2 in overlap_pairs:
        union(id1, id2)

    # Generate new global IDs
    new_id = 1
    id_remap = {}
    for old_id in parent:
        root = find(old_id)
        if root not in id_remap:
            id_remap[root] = new_id
            new_id += 1
        id_remap[old_id] = id_remap[root]

    return id_remap

# ========== Main Program ==========
if __name__ == "__main__":
    start_time = time.time()
    logging.info("Start: Merging patch masks (with georeference)")
    print("Start: Merging patch masks")

    # Open input RGB TIFF to get reference CRS and size
    with rasterio.open(RGB_TIF) as rgb_src:
        width, height = rgb_src.width, rgb_src.height
        rgb_transform = rgb_src.transform
        rgb_crs = rgb_src.crs
        profile = rgb_src.profile.copy()
        profile.update({"count": 1, "dtype": "uint32", "BIGTIFF": "YES"})

    # Initialize output file
    with rasterio.open(OUTPUT_MASK, "w", **profile) as dst:
        dst.write(np.zeros((1, height, width), dtype=np.uint32))

    # Compute patch grid
    step = PATCH_SIZE - OVERLAP
    num_rows = (height - OVERLAP) // step + 1
    num_cols = (width - OVERLAP) // step + 1
    print(f"Merging {num_rows} x {num_cols} patches")

    # Phase 1: Collect overlap info and assign preliminary IDs
    buffers = {}  # (row_idx, col_idx) -> {'right': mask, 'bottom': mask}

    with rasterio.open(OUTPUT_MASK, "r+", **profile) as dst:
        for row_idx in tqdm(range(num_rows), desc="Phase 1: Process rows"):
            for col_idx in tqdm(range(num_cols), desc=f"Process columns (row {row_idx+1})", leave=False):
                patch_file = os.path.join(TEMP_PATCH_DIR, f"patch_{row_idx}_{col_idx}.tif")
                if not os.path.exists(patch_file):
                    continue

                with rasterio.open(patch_file) as patch_src:
                    mask_local = patch_src.read(1)
                    win_h, win_w = mask_local.shape
                    patch_crs = patch_src.crs

                if patch_crs != rgb_crs:
                    logging.warning(f"Patch CRS mismatch: {patch_file}")
                    continue

                row_off = row_idx * step
                col_off = col_idx * step
                window = Window(col_off, row_off, win_w, win_h)

                mask_out = np.zeros((win_h, win_w), dtype=np.uint32)
                unique_ids = np.unique(mask_local)

                for local_id in unique_ids:
                    if local_id == 0:
                        continue
                    local_mask = (mask_local == local_id).astype(np.uint8)
                    if local_mask.sum() < MIN_AREA:
                        continue

                    # Assign unique global ID
                    current_gid = global_id
                    global_id += 1
                    id_mapping[(row_idx, col_idx, local_id)] = current_gid

                    # Check left overlap
                    if col_idx > 0:
                        left_key = (row_idx, col_idx - 1)
                        if left_key in buffers and buffers[left_key]['right'] is not None:
                            left_overlap_mask = buffers[left_key]['right']
                            overlap_region = local_mask[:, :OVERLAP]
                            for uid in np.unique(left_overlap_mask):
                                if uid == 0:
                                    continue
                                adj_mask = (left_overlap_mask == uid).astype(np.uint8)
                                iou = compute_iou(overlap_region, adj_mask[:, -OVERLAP:])
                                if iou > IOU_THRESHOLD:
                                    overlap_pairs.append((current_gid, uid))

                    # Check top overlap
                    if row_idx > 0:
                        top_key = (row_idx - 1, col_idx)
                        if top_key in buffers and buffers[top_key]['bottom'] is not None:
                            top_overlap_mask = buffers[top_key]['bottom']
                            overlap_region = local_mask[:OVERLAP, :]
                            for uid in np.unique(top_overlap_mask):
                                if uid == 0:
                                    continue
                                adj_mask = (top_overlap_mask == uid).astype(np.uint8)
                                iou = compute_iou(overlap_region, adj_mask[-OVERLAP:, :])
                                if iou > IOU_THRESHOLD:
                                    overlap_pairs.append((current_gid, uid))

                    mask_out[local_mask > 0] = current_gid

                existing_mask = dst.read(1, window=window)
                combined_mask = np.where(mask_out > 0, mask_out, existing_mask)
                dst.write(combined_mask, 1, window=window)

                # Save right and bottom overlap for next patches
                buffers[(row_idx, col_idx)] = {
                    'right': mask_out[:, -OVERLAP:].copy() if col_idx < num_cols - 1 else None,
                    'bottom': mask_out[-OVERLAP:, :].copy() if row_idx < num_rows - 1 else None
                }

                del mask_local, mask_out

            # Clear previous row buffers
            for col_idx in range(num_cols):
                if (row_idx - 1, col_idx) in buffers:
                    del buffers[(row_idx - 1, col_idx)]

            mem_usage = psutil.Process().memory_info().rss / 1024**2
            elapsed_time = time.time() - start_time
            tqdm.write(f"Row {row_idx+1} done, memory {mem_usage:.2f} MB, time {elapsed_time:.2f}s")

    # Phase 2: Merge global IDs
    print("Phase 2: Merging global IDs...")
    id_remap = merge_global_ids(overlap_pairs)

    with rasterio.open(OUTPUT_MASK, "r+", **profile) as dst:
        for row_idx in tqdm(range(num_rows), desc="Phase 2: Update IDs"):
            for col_idx in range(num_cols):
                patch_file = os.path.join(TEMP_PATCH_DIR, f"patch_{row_idx}_{col_idx}.tif")
                if not os.path.exists(patch_file):
                    continue

                with rasterio.open(patch_file) as patch_src:
                    mask_local = patch_src.read(1)
                    win_h, win_w = mask_local.shape

                row_off = row_idx * step
                col_off = col_idx * step
                window = Window(col_off, row_off, win_w, win_h)

                mask_out = np.zeros((win_h, win_w), dtype=np.uint32)
                for local_id in np.unique(mask_local):
                    if local_id == 0:
                        continue
                    gid = id_mapping.get((row_idx, col_idx, local_id))
                    if gid and gid in id_remap:
                        mask_out[mask_local == local_id] = id_remap[gid]

                existing_mask = dst.read(1, window=window)
                combined_mask = np.where(mask_out > 0, mask_out, existing_mask)
                dst.write(combined_mask, 1, window=window)

                del mask_local, mask_out

    elapsed_time = time.time() - start_time
    print(f"✅ Done, merged raster saved to {OUTPUT_MASK}")
    print(f"Total runtime: {elapsed_time:.2f} s")
