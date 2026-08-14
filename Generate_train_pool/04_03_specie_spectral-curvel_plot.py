import os
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil

# ========== 配置 ==========
folder_path = r"/storage/group/tvq5043/default/Zhuohong/NEON_data/Preprocess/MLBS/training_sample/"
output_csv = r"/storage/group/tvq5043/default/Zhuohong/NEON_data/Preprocess/MLBS/training_specie_spetral.csv"
filtered_csv = output_csv.replace(".csv", "_filtered.csv")
output_plot = r"/storage/group/tvq5043/default/Zhuohong/NEON_data/Preprocess/MLBS/training_specie_spetral.png"
band_csv = r"/storage/group/tvq5043/default/Zhuohong/NEON_data/Preprocess/MLBS/Support_data/NEON_wavelength.csv"

max_band = 426  # 只取前426波段
n_cols_legend = 6  # 图例每行显示数量
legend_fontsize = 12  # 图例文字大小

# ========== 读取波段信息 ==========
band_df = pd.read_csv(band_csv, encoding='utf-8')  # 如果报错可改 'gbk' 或 'latin1'
band_numbers = band_df['Band'].values[:max_band]
wavelengths = band_df['wavelength'].values[:max_band]

# ========== 遍历 tiff 文件，计算平均光谱 ==========
species_dict = {}
for file_name in os.listdir(folder_path):
    if file_name.endswith(".tif"):
        species_name = file_name.split("_")[0]
        file_path = os.path.join(folder_path, file_name)
        with rasterio.open(file_path) as src:
            data = src.read().astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            band_means = np.nanmean(data.reshape(data.shape[0], -1), axis=1)
            band_means = band_means[:max_band]  # 只取前426波段
        species_dict.setdefault(species_name, []).append(band_means)

# ========== 计算每个树种的平均光谱 ==========
species_means = {}
for species, arrays in species_dict.items():
    stacked = np.stack(arrays, axis=0)
    mean_spectrum = np.nanmean(stacked, axis=0)
    species_means[species] = mean_spectrum[:max_band]

# ========== 保存原始 CSV ==========
df = pd.DataFrame(species_means, index=band_numbers)
df.index.name = 'Band'
df.to_csv(output_csv)
print(f"CSV saved to {output_csv}")

# ========== 过滤异常值并保存新的 CSV ==========
df_filtered = df.copy()
df_filtered[(df_filtered < 0) | (df_filtered > 5000)] = 0
df_filtered.to_csv(filtered_csv)
print(f"Filtered CSV saved to {filtered_csv}")

# ========== 绘制光谱曲线 ==========
species_list = list(species_means.keys())
n_rows_legend = ceil(len(species_list)/n_cols_legend)

# 设置整体fig大小，给图例留出足够空间
fig_height = 6 + 0.6 * n_rows_legend
fig, ax = plt.subplots(figsize=(14, fig_height))

# 绘制折线
for species, spectrum in species_means.items():
    ax.plot(band_numbers, spectrum)

ax.set_xlabel("Band")
ax.set_ylabel("Average Reflectance")
ax.set_title("Average Spectral Curves per Species", pad=25)
ax.grid(True)

# 上轴显示波长，稀疏显示
step = max(1, len(band_numbers)//15)
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(band_numbers[::step])
ax2.set_xticklabels([f"{w:.0f}" for w in wavelengths[::step]])
ax2.set_xlabel("Wavelength (nm)")

# 下方图例，完全放在图下面
fig.legend(species_list, ncol=n_cols_legend, loc='lower center', bbox_to_anchor=(0.5, -0.15), fontsize=legend_fontsize)
plt.tight_layout()
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot saved to {output_plot}")
