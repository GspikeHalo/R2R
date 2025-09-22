import os
import re
import torch
import scipy.io
import argparse
from PIL import Image
import numpy as np
import sys
from torch import pixel_shuffle
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


POSTFIX_TO_CAM = {
    'A': 'huawei',
    'B': 'nikon',
    'C': 'iphone',
    'D': 'samsung',
}
CAMERAS = ['iphone', 'huawei', 'nikon', 'samsung']  # 可改成你想要的顺序

RAW_VAR_CANDIDATES = ['raw_rggb']  # 如有别名可加在这里，比如 'raw4', 'raw'

def split_large_image(**kwargs):
    image, type = kwargs['image'],kwargs['type']
    if type=='split':
        size = kwargs['patch_size']
        height, width = image.shape[:2]
        # 计算分割后的小图片数量
        num_cols = width // size
        num_rows = height // size
        small_images = []
        for row in range(num_rows):
            for col in range(num_cols):
                # 计算每个小图片的位置
                left = col * size
                top = row * size
                right = left + size
                bottom = top + size
                # 分割并保存小图片
                small_image = image[top:bottom,left:right, ...]
                small_images.append(small_image)
        return small_images
    elif type=='zoom':
        processed_image_size = kwargs['processed_image_size']
        return [cv2.resize(image,processed_image_size)]

def depth_to_space(feature_maps):
    if not torch.is_tensor(feature_maps):
        feature_maps = torch.tensor(feature_maps)
    feature_maps = pixel_shuffle(feature_maps, 2)
    feature_maps = np.array(feature_maps.to('cpu'))
    return feature_maps

def load_raw_rggb_from_mat(mat_path):
    """从 .mat 里取出 (H,W,4) 的 float32, [0,1] 的原始 RGGB（Chinese–English）"""
    data = scipy.io.loadmat(mat_path)
    for key in RAW_VAR_CANDIDATES:
        if key in data:
            arr = data[key]
            # 兼容 MATLAB cell / object 嵌套
            if hasattr(arr, 'dtype') and arr.dtype == object and arr.size == 1:
                arr = arr.item()
            arr = np.asarray(arr)
            if arr.ndim != 3 or arr.shape[2] != 4:
                raise ValueError(f"[ERROR] Variable '{key}' must be (H,W,4). Got {arr.shape} in {mat_path}")
            return arr.astype(np.float32)
    raise KeyError(f"[ERROR] None of {RAW_VAR_CANDIDATES} found in {mat_path}. Keys: {list(data.keys())}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['zoom', 'split'], default='split')  # 切块策略 patching strategy
    parser.add_argument('--processed_image_size', type=list, default=[512, 384])  # 仅在 type=zoom 时使用
    parser.add_argument('--patch_size', default=256, type=int)  # 四通道 RGGB 的 patch 尺寸
    parser.add_argument('--parent_path', default='/media/Data_2/R2RResult/r2r-odb/')
    parser.add_argument('--paired_data_path', default='paired')
    parser.add_argument('--unpaired_data_path', default='unpaired')
    parser.add_argument('--processed_data_path', default='processed')
    parser.add_argument('--raw_path', default='raw-rggb')  # 仅 unpaired 用；paired 固定为 'raw'
    args = parser.parse_args()

    # 用 join 更稳健（Chinese–English）
    args.paired_data_path = os.path.join(args.parent_path, args.paired_data_path)
    args.unpaired_data_path = os.path.join(args.parent_path, args.unpaired_data_path)
    args.processed_data_path = os.path.join(args.parent_path, args.processed_data_path)

    # 输出目录（processed）
    for cam in CAMERAS:
        ensure_dir(os.path.join(args.processed_data_path, 'test', cam))
        ensure_dir(os.path.join(args.processed_data_path, 'test', f'vis-{cam}'))
        ensure_dir(os.path.join(args.processed_data_path, 'unpaired', cam))
        ensure_dir(os.path.join(args.processed_data_path, 'unpaired', f'vis-{cam}'))

    # -----------------------------
    # Part 1: Paired
    # 读取 {parent}/paired/raw/*.mat 结构（与预处理输出一致）
    # -----------------------------
    paired_raw_dir = os.path.join(args.paired_data_path, 'raw')
    if os.path.isdir(paired_raw_dir):
        file_re = re.compile(r'^(?P<stem>.+)_(?P<code>[ABCD])\.mat$', re.IGNORECASE)
        groups = {}
        for fn in os.listdir(paired_raw_dir):
            if not fn.lower().endswith('.mat'):
                continue
            m = file_re.match(fn)
            if not m:
                continue
            stem = m.group('stem')
            code = m.group('code').upper()
            cam = POSTFIX_TO_CAM.get(code, None)
            if cam is None:
                continue
            groups.setdefault(stem, {})[cam] = os.path.join(paired_raw_dir, fn)

        print(f'[INFO] Found {len(groups)} test stems in {paired_raw_dir}')
        index = 0
        for stem, cam_files in sorted(groups.items()):
            s_index = index
            max_patches = 0
            for cam in CAMERAS:
                if cam not in cam_files:
                    continue
                mat_path = cam_files[cam]
                try:
                    camera_raw = load_raw_rggb_from_mat(mat_path)  # (H,W,4), [0,1]
                except Exception as e:
                    print(f'[ERROR] Read MAT failed: {mat_path} -> {e}')
                    continue

                # 切块 split patches
                camera_raw_splits = split_large_image(
                    image=camera_raw,
                    patch_size=args.patch_size,
                    type=args.type,
                    processed_image_size=args.processed_image_size
                )

                # 同一 stem 下各相机索引对齐（Chinese–English: index alignment per stem）
                index = s_index
                max_patches = max(max_patches, len(camera_raw_splits))

                for camera_raw_split in camera_raw_splits:
                    # (H,W,4) -> (4,H,W)
                    camera_raw_split = camera_raw_split.transpose(2, 0, 1)
                    np.save(os.path.join(args.processed_data_path, 'test', cam, f'{index}.npy'), camera_raw_split)

                    # 可视化 visualization
                    vis = depth_to_space(camera_raw_split).squeeze(0)  # 依你现有的 depth_to_space 约定
                    image = Image.fromarray((np.clip(vis, 0.0, 1.0) * 255).astype(np.uint8))
                    image.save(os.path.join(args.processed_data_path, 'test', f'vis-{cam}', f'{index}.png'))

                    index += 1

                print(f'[INFO][paired] stem={stem} cam={cam} patches={len(camera_raw_splits)}')

            # 跳到下一个 stem 的起始索引（Chinese–English）
            index = s_index + max_patches
    else:
        print(f'[WARN] test input dir not found: {paired_raw_dir}')

    # -----------------------------
    # Part 2: Unpaired
    # 读取 {parent}/unpaired/{camera}/raw-rggb/*.mat 结构（与预处理输出一致）
    # -----------------------------
    for cam in CAMERAS:
        in_dir = os.path.join(args.unpaired_data_path, cam, args.raw_path)  # 默认 raw-rggb
        if not os.path.isdir(in_dir):
            print(f'[WARN] unpaired input dir not found: {in_dir}')
            continue
        raws = [f for f in os.listdir(in_dir) if f.lower().endswith('.mat')]
        raws.sort()
        print(f'[INFO] Unpaired {cam}: {len(raws)} files')

        index = 0
        for raw in raws:
            in_path = os.path.join(in_dir, raw)
            try:
                camera_raw = load_raw_rggb_from_mat(in_path)  # (H,W,4), [0,1]
            except Exception as e:
                print(f'[ERROR] Read MAT failed: {in_path} -> {e}')
                continue

            camera_raw_splits = split_large_image(
                image=camera_raw,
                patch_size=args.patch_size,
                type=args.type,
                processed_image_size=args.processed_image_size
            )

            for camera_raw_split in camera_raw_splits:
                camera_raw_split = camera_raw_split.transpose(2, 0, 1)
                np.save(os.path.join(args.processed_data_path, 'unpaired', cam, f'{index}.npy'),
                        camera_raw_split)

                vis = depth_to_space(camera_raw_split).squeeze(0)
                image = Image.fromarray((np.clip(vis, 0.0, 1.0) * 255).astype(np.uint8))
                image.save(os.path.join(args.processed_data_path, 'unpaired', f'vis-{cam}', f'{index}.png'))

                index += 1

            print(f'[INFO][unpaired] cam={cam} file={raw} patches={len(camera_raw_splits)}')
