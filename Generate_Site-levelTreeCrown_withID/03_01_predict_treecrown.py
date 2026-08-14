import os
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm
import psutil
import time
import logging

# ========== Configuration ==========
RGB_TIF = "/storage/group/tvq5043/default/Zhuohong/NEON_data/RGB_MLBS_intersection.tif"
MODEL_WEIGHTS = "/storage/group/tvq5043/default/Zhuohong/NEON_data/Preprocess/MLBS/model_epoch18.pth"
PATCH_SIZE = 800
OVERLAP = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_AREA = 10
BATCH_SIZE = 64  # Adjust according to GPU memory
TEMP_PATCH_DIR = "/storage/group/tvq5043/default/Zhuohong/NEON_data/Preprocess/MLBS/temp_patches"

# Set up logging
logging.basicConfig(filename='predict.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Create temporary directory
os.makedirs(TEMP_PATCH_DIR, exist_ok=True)

# ========== Main Program ==========
if __name__ == "__main__":
    start_time = time.time()
    logging.info("Step 1 start: Predict patch masks")

    # Load the model
    logging.info("Loading model...")
    print("Loading model...")
    model = maskrcnn_resnet50_fpn(num_classes=2, weights=None)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    logging.info(f"Model loaded, memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

    # Open the input image
    logging.info("Opening image...")
    print("Opening image...")
    src = rasterio.open(RGB_TIF)
    width, height = src.width, src.height
    logging.info(f"Image size: {width}x{height}")
    print(f"Image size: {width}x{height}")
    profile = src.profile.copy()
    profile.update({"count": 1, "dtype": "uint32", "BIGTIFF": "YES"})

    # Generate patch list
    step = PATCH_SIZE - OVERLAP
    patches = [(row_off, col_off, min(PATCH_SIZE, height - row_off), min(PATCH_SIZE, width - col_off))
               for row_off in range(0, height, step)
               for col_off in range(0, width, step)]
    logging.info(f"Total {len(patches)} patches")
    print(f"Total {len(patches)} patches")

    # Batch prediction and save patch masks
    print("Starting patch mask prediction and saving...")
    for i in tqdm(range(0, len(patches), BATCH_SIZE), desc="Prediction batches"):
        batch_patches = patches[i:i + BATCH_SIZE]
        batch_tensors = []
        batch_metadata = []

        # Prepare batch data
        for row_off, col_off, win_h, win_w in batch_patches:
            window = Window(col_off, row_off, win_w, win_h)
            # Read image patch
            patch_data = src.read(window=window) / 255.0
            padded = np.zeros((3, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            padded[:, :win_h, :win_w] = patch_data
            img_tensor = torch.from_numpy(padded).unsqueeze(0).to(DEVICE)
            batch_tensors.append(img_tensor)
            batch_metadata.append((row_off, col_off, win_h, win_w))

        # Batch inference
        batch_tensors = torch.cat(batch_tensors, dim=0)
        with torch.no_grad():
            outputs = model(batch_tensors)

        # Process each patch output and save as TIFF
        for (row_off, col_off, win_h, win_w), output in zip(batch_metadata, outputs):
            mask_out = np.zeros((win_h, win_w), dtype=np.uint32)
            local_id = 1  # Local ID within each patch starts from 1
            for m in output['masks']:
                m = m[0].cpu().numpy()
                m = (m > 0.5).astype(np.uint8)
                if m[:win_h, :win_w].sum() < MIN_AREA:
                    continue
                mask_out[m[:win_h, :win_w] > 0] = local_id
                local_id += 1

            # Save as TIFF
            row_idx = row_off // step
            col_idx = col_off // step
            patch_file = os.path.join(TEMP_PATCH_DIR, f"patch_{row_idx}_{col_idx}.tif")
            patch_profile = src.profile.copy()
            patch_profile.update({"height": win_h, "width": win_w, "count": 1, "dtype": "uint32"})
            with rasterio.open(patch_file, "w", **patch_profile) as patch_dst:
                patch_dst.write(mask_out, 1)

            # Free memory
            del mask_out

        # Free batch memory
        del batch_tensors, outputs
        torch.cuda.empty_cache()

        # Monitor memory usage
        mem_usage = psutil.Process().memory_info().rss / 1024**2
        elapsed_time = time.time() - start_time
        logging.info(f"Batch {i//BATCH_SIZE + 1} done, memory usage: {mem_usage:.2f} MB, time: {elapsed_time:.2f} s")

    src.close()
    elapsed_time = time.time() - start_time
    print(f"✅ Step 1 complete: All patch masks saved to {TEMP_PATCH_DIR}")
    print(f"Total runtime: {elapsed_time:.2f} s")
    logging.info(f"Step 1 complete, total runtime: {elapsed_time:.2f} s")
