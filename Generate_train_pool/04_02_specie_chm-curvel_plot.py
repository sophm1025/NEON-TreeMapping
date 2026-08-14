import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ========== Config ==========
tif_folder = r"/storage/group/tvq5043/default/Zhuohong/NEON_data/Preprocess/MLBS/training_sample_hsi-chm/"
output_png = r"/storage/group/tvq5043/default/Zhuohong/NEON_data/Preprocess/MLBS/Support_data/taxon_chm_mean.png"

# Collect results
data = []

# Traverse all tif files
for file in os.listdir(tif_folder):
    if file.endswith(".tif"):
        filepath = os.path.join(tif_folder, file)

        # Extract species name (filename format: "species_ID.tif")
        species = file.split("_")[0]

        # Open tif, read band 1
        with rasterio.open(filepath) as src:
            band1 = src.read(1).astype(float)

            # Remove nodata values
            if src.nodata is not None:
                band1 = band1[band1 != src.nodata]
            band1 = band1[np.isfinite(band1)]

            if band1.size > 0:
                mean_val = band1.mean()
                data.append({"species": species, "height_m": mean_val})

# Convert to DataFrame
df = pd.DataFrame(data)

# Count number of samples per species
species_counts = df["species"].value_counts().sort_index()

# ========== Plot ==========
fig, ax1 = plt.subplots(figsize=(10, 6))

# Boxplot on left y-axis
df.boxplot(column="height_m", by="species", grid=False, ax=ax1)
ax1.set_title("Tree Height Distribution by Species")
ax1.set_xlabel("Species")
ax1.set_ylabel("Tree Height (m)")
plt.suptitle("")  # remove automatic title
plt.xticks(rotation=45)

# Add right y-axis with sample counts
ax2 = ax1.twinx()
species_order = sorted(df["species"].unique())
x_positions = range(1, len(species_order) + 1)

# Match species order with counts
counts = [species_counts.get(sp, 0) for sp in species_order]

ax2.plot(x_positions, counts, "D", color="red", label="Sample count")  # red diamonds
ax2.set_ylabel("Number of samples")

# Save to PNG
plt.tight_layout()
plt.savefig(output_png, dpi=300)
plt.close()

print(f"Boxplot saved to: {output_png}")
