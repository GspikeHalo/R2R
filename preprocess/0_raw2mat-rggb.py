import os
import errno
from copy import deepcopy

import cv2
import numpy as np
import rawpy
from scipy.io import savemat

def apply_orient(arr, mode):
    """
    对 2D/3D(H,W,C) 数组做手动翻转/旋转 (manual flip/rotation).
    mode ∈ {'none','cw90','ccw90','rot180','hflip','vflip'}
      - 'cw90'   : 顺时针 90° (clockwise 90)
      - 'ccw90'  : 逆时针 90° (counter-clockwise 90)
      - 'rot180' : 旋转 180°
      - 'hflip'  : 水平翻转 (horizontal)
      - 'vflip'  : 垂直翻转 (vertical)
    """
    if mode in (None, 'none'):
        return arr
    if mode == 'cw90':
        return np.rot90(arr, -1, axes=(0, 1))
    if mode == 'ccw90':
        return np.rot90(arr,  1, axes=(0, 1))
    if mode == 'rot180':
        return np.rot90(arr,  2, axes=(0, 1))
    if mode == 'hflip':
        return np.flip(arr, axis=1)
    if mode == 'vflip':
        return np.flip(arr, axis=0)
    raise ValueError(f"Unknown orientation mode: {mode}")



def check_dir(path_):
    if not os.path.exists(path_):
        try:
            os.makedirs(path_)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

def imwrite(filename, image):
    """Save RGB float image in [0,1] to an 8-bit file."""
    image = np.clip(image, 0.0, 1.0) * 255.0  # 你已改为 255，这里保持
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image)

def _cfa_to_pos_order(cfa):
    """
    将 cfa 统一为按 RGGB 顺序的 2×2 位置索引数组(0..3)：
    - 位置式 position-based: 直接返回（4 个值是 {0,1,2,3} 的排列）
    - 颜色式 color-based: 假定 0=R, 1/3=G, 2=B；找出 R/G/G/B 所在位置并返回 [R, G1, G2, B]
    """
    cfa = np.asarray(cfa, dtype=int).ravel()
    if cfa.size != 4:
        raise ValueError(f"CFA must have 4 entries, got {cfa.size}: {cfa}")

    vals = np.unique(cfa)

    if set(vals.tolist()) == {0, 1, 2, 3} and np.all(np.sort(cfa) == np.array([0, 1, 2, 3])):
        return cfa

    if not set(vals.tolist()).issubset({0, 1, 2, 3}):
        raise ValueError(f"Unsupported CFA values: {vals}")

    r_pos = np.where(cfa == 0)[0]
    b_pos = np.where(cfa == 2)[0]
    g_pos = np.where((cfa == 1) | (cfa == 3))[0]

    if r_pos.size != 1 or b_pos.size != 1 or g_pos.size != 2:
        raise ValueError(f"Color-based CFA must contain 1×R, 2×G(1/3), 1×B. Got R:{r_pos}, G:{g_pos}, B:{b_pos}")

    g_pos = np.sort(g_pos)
    return np.array([int(r_pos[0]), int(g_pos[0]), int(g_pos[1]), int(b_pos[0])], dtype=int)


def pos_order_from_raw(raw):
    """
    从 rawpy 读到的 raw_pattern(2x2) 与 color_desc 推断 2×2 的位置索引 [R,G1,G2,B]。
    兼容 RYYB：把 'Y' 视作 G-like（与 'G' 同类）。
    """
    pat = raw.raw_pattern.copy()
    cdesc = raw.color_desc
    if isinstance(cdesc, bytes):
        cdesc = cdesc.decode()

    letters = [cdesc[pat[0,0]], cdesc[pat[0,1]], cdesc[pat[1,0]], cdesc[pat[1,1]]]

    r_pos = [i for i, L in enumerate(letters) if L == 'R']
    b_pos = [i for i, L in enumerate(letters) if L == 'B']
    g_pos = [i for i, L in enumerate(letters) if L in ('G', 'Y')]

    if len(r_pos) == 1 and len(b_pos) == 1 and len(g_pos) == 2:
        g_pos.sort()
        return np.array([r_pos[0], g_pos[0], g_pos[1], b_pos[0]], dtype=int)

    raise ValueError(f"Unsupported CFA for RGGB/RYYB-style packing: letters={letters}, color_desc={cdesc}")


def pack_rggb(raw_image, cfa):
    """
    将 Bayer 2D 马赛克 (Bayer mosaic) 打包为 4 通道 RGGB。
    返回形状：(H/2, W/2, 4) ，通道顺序 [R, G1, G2, B]
    """
    if raw_image.ndim != 2:
        raise ValueError(f"raw_image must be 2D, got shape {raw_image.shape}")

    h, w = raw_image.shape
    pos_order = _cfa_to_pos_order(cfa)  # 长度 4，元素 ∈ {0,1,2,3}
    idx = [(0, 0), (0, 1), (1, 0), (1, 1)]  # 2×2 位置到采样步进

    channels = []
    for pos in pos_order:
        rr, cc = idx[pos]
        ch = raw_image[rr:h:2, cc:w:2].copy()
        channels.append(ch)

    base_shape = channels[0].shape
    if not all(ch.shape == base_shape for ch in channels):
        raise RuntimeError(f"Packed channel shapes mismatch: {[ch.shape for ch in channels]}")

    return np.stack(channels, axis=-1)  # [R,G1,G2,B]


def from_rggb_to_rgb(rggb):
    """可视化：合并两路 G 并按 RGB 返回（不就地修改输入）。"""
    g = (rggb[..., 1] + rggb[..., 2]) / 2.0
    return np.stack([rggb[..., 0], g, rggb[..., 3]], axis=-1)

BASE_DIR   = '/media/Data_2/r2r_odb/'
RESULT_DIR = '/media/Data_2/R2RResult/r2r-odb'

pair_data = ['paired/', 'unpaired/']
cameras   = ['huawei/', 'nikon/']

postfix_map = {
    'huawei/': '_A',
    'nikon/':  '_B',
}

meta_data_huawei = {'white_level': 4095,  'black_level': 256,  'cfa_pattern': np.array([2, 3, 1, 0])}
meta_data_nikon  = {'white_level': 16383, 'black_level': 1008, 'cfa_pattern': np.array([0, 1, 3, 2])}

camera_meta = {
    'huawei/': meta_data_huawei,
    'nikon/':  meta_data_nikon,
}

ORIENT_PER_CAMERA = {
    'huawei/': 'none',   # 华为不翻转
    'nikon/':  'hflip',   # 尼康：默认顺时针90°；如方向不对，改成 'ccw90' / 'rot180' / 'hflip' / 'vflip'
}

for _pair in pair_data:

    def get_out_dirs(pair_tag, cam_tag):
        if pair_tag == 'paired/':
            out_rggb_dir = os.path.join(RESULT_DIR, 'paired', 'raw')
            out_vis_dir  = os.path.join(RESULT_DIR, 'paired', 'jpg')
        else:
            out_rggb_dir = os.path.join(RESULT_DIR, pair_tag, cam_tag, 'raw-rggb')
            out_vis_dir  = os.path.join(RESULT_DIR, pair_tag, cam_tag, 'vis')
        return out_rggb_dir, out_vis_dir

    for _cam in cameras:
        postfix = postfix_map[_cam]

        path_to_raw_data = os.path.join(BASE_DIR, _pair, _cam)
        in_dir = os.path.join(path_to_raw_data, 'raw')
        if not os.path.isdir(in_dir):
            print(f'[WARN] Skip: directory not found: {in_dir}')
            continue

        out_rggb_dir, out_vis_dir = get_out_dirs(_pair, _cam)
        check_dir(out_rggb_dir)
        check_dir(out_vis_dir)

        all_raw_img_paths = [
            os.path.join(in_dir, f) for f in os.listdir(in_dir)
            if f.lower().endswith(('.dng', '.nef'))
        ]
        all_raw_img_paths.sort()

        meta_data = camera_meta[_cam]

        for raw_img_path in all_raw_img_paths:
            try:
                with rawpy.imread(raw_img_path) as raw:
                    raw_bayer = raw.raw_image_visible.copy()

                    try:
                        pos_order = pos_order_from_raw(raw)
                    except Exception:
                        pos_order = deepcopy(meta_data['cfa_pattern'])

            except Exception as e:
                print(f'[ERROR] Failed to read: {raw_img_path} -> {e}')
                continue

            if raw_bayer.ndim != 2:
                print(f'[WARN] Not a 2D mosaic plane, skipped: {raw_img_path} (shape={raw_bayer.shape})')
                continue

            wl = float(meta_data['white_level'])
            bl = float(meta_data['black_level'])
            denom = max(wl - bl, 1.0)
            raw_bayer_norm = (raw_bayer.astype(np.float32) - bl) / denom
            raw_bayer_norm = np.clip(raw_bayer_norm, 0.0, 1.0)

            raw_rggb = pack_rggb(raw_bayer_norm, pos_order)
            raw_rggb = apply_orient(raw_rggb, ORIENT_PER_CAMERA.get(_cam, 'none'))

            stem = os.path.splitext(os.path.basename(raw_img_path))[0]

            savemat(os.path.join(out_rggb_dir, stem + postfix + '.mat'),
                    {"raw_rggb": raw_rggb})

            vis = (from_rggb_to_rgb(raw_rggb) * 0.9) ** (1 / 1.6)
            imwrite(os.path.join(out_vis_dir, stem + postfix + '.jpg'), vis)

            print(stem)
