# create_noise_profile_npy.py
import torch
import os
import numpy as np
from tqdm import tqdm

PATCH_SIZE = 16
NUM_BINS = 100
IMG_DIR = '/media/Data_2/R2RResult/processed/unpaired/iphone-x'
OUTPUT_FILE = 'noise_profiles/iphone_profile.pt'
# IMG_DIR = '/media/Data_2/R2RResult/processed/unpaired/samsung-s9'
# OUTPUT_FILE = 'noise_profiles/samsung_profile.pt'
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

all_means = []
all_vars = []

# 只载入 .npy 文件
image_files = [os.path.join(IMG_DIR, f)
               for f in os.listdir(IMG_DIR)
               if f.endswith('.npy')]

def load_and_preprocess_npy(path):
    # 假设 np.load 后得到形状 [C, H, W]，范围已在 [0,1]
    data = np.load(path)
    return torch.from_numpy(data).float()  # -> [C, H, W]

for npy_path in tqdm(image_files):
    img_tensor = load_and_preprocess_npy(npy_path)
    # 以下与原脚本一致
    patches = img_tensor.unfold(1, PATCH_SIZE, PATCH_SIZE) \
                        .unfold(2, PATCH_SIZE, PATCH_SIZE)
    patches = patches.contiguous().view(img_tensor.size(0), -1, PATCH_SIZE * PATCH_SIZE)
    mean_per_patch = patches.mean(dim=-1)
    var_per_patch  = patches.var(dim=-1)
    all_means.append(mean_per_patch.mean(dim=0))
    all_vars.append(var_per_patch.mean(dim=0))

all_means = torch.cat(all_means)
all_vars  = torch.cat(all_vars)

bins = torch.linspace(0.0, 1.0, NUM_BINS + 1)
bin_sum_vars = torch.zeros(NUM_BINS)
bin_counts   = torch.zeros(NUM_BINS)

bin_indices = torch.bucketize(all_means, bins, right=True) - 1
for i, idx in enumerate(bin_indices):
    if 0 <= idx < NUM_BINS:
        bin_sum_vars[idx] += all_vars[i]
        bin_counts[idx]   += 1

master_histogram = bin_sum_vars / (bin_counts + 1e-8)
torch.save(master_histogram, OUTPUT_FILE)
print(f"Noise profile saved to {OUTPUT_FILE}")