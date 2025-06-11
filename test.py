# import numpy as np

# # 替换为你的 .npy 文件路径
# npy_path = '/media/Data_2/R2RResult/starGanData/starGanData/746_B.npy'

# # 加载 numpy 数组
# data = np.load(npy_path)

# # 打印形状和通道数
# print(f"Shape: {data.shape}")
# if len(data.shape) == 3:
#     print(f"Channels: {data.shape[0]} (assumed format: [C, H, W])")
# elif len(data.shape) == 2:
#     print("Grayscale image (single channel)")
# else:
#     print("Unknown format")
import rawpy

raw = rawpy.imread("/media/Data_2/RAW2RAW/paired/iphone-x/dng/2021-06-03_20-35-48.dng")

black_level = raw.black_level_per_channel
white_level = raw.white_level

print(f"black level: {black_level}")
print(f"white level: {white_level}")