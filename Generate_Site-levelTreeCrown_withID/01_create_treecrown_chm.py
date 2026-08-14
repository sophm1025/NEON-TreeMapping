import rasterio
from skimage import feature, segmentation, morphology
from scipy.ndimage import gaussian_filter
import numpy as np
from tqdm import tqdm

# 输入和输出路径
input_path = '/storage/group/tvq5043/default/Zhuohong/NEON_data/CHM_MLBS_intersection.tif'
output_path = '/storage/group/tvq5043/default/Zhuohong/NEON_data/Preprocess/MLBS/CHM_based_tree_crown_above3m.tif'

# 读取CHM并获取地理元数据
try:
    with rasterio.open(input_path) as src:
        chm = src.read(1)  # 读取第一个波段
        profile = src.profile  # 获取元数据（CRS、变换矩阵等）
except rasterio.errors.RasterioIOError as e:
    print(f"Error reading the file: {e}")
    exit(1)

# 将CHM中低于5米的区域设置为0（非树木）
chm[chm < 3] = 0

# 平滑处理
chm_smooth = gaussian_filter(chm, sigma=0.5)

# 检测树顶
peaks = feature.peak_local_max(chm_smooth, min_distance=int(2))

# 创建与CHM相同形状的markers数组
markers = np.zeros_like(chm_smooth, dtype=np.int32)
for i, peak in tqdm(enumerate(peaks, 1), desc="Processing peaks"):  # 从1开始编号树顶
    markers[peak[0], peak[1]] = i

# 分水岭分割，使用高度≥5米的mask
labels = segmentation.watershed(-chm_smooth, markers, mask=chm_smooth >= 5)

# 去除小区域
# labels = morphology.remove_small_objects(labels, min_size=50)

# 保存分割结果为TIF文件，保留地理坐标
profile.update(dtype=rasterio.int32, nodata=0)  # 更新元数据，设置数据类型为int32
with rasterio.open(output_path, 'w', **profile) as dst:
    dst.write(labels.astype(np.int32), 1)
    dst.nodata = 0  # 设置nodata值
