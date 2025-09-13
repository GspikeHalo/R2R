import cv2
import numpy as np
import rawpy
from scipy.io import savemat
import os
import errno
from copy import deepcopy

def check_dir(path_):
    if not os.path.exists(path_):
        try:
            os.makedirs(path_)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

def pack_rggb(raw_image, cfa):
    """Pack a Bayer 2D plane into 4-channel RGGB according to the given CFA order."""
    height, width = raw_image.shape
    channels = []
    cfa = cfa.copy()
    cfa[cfa == 2] += 1
    cfa[2:][cfa[2:] == 1] += 1
    idx = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for c in cfa:
        raw_c = raw_image[idx[c][0]:height:2, idx[c][1]:width:2].copy()
        channels.append(raw_c)
    return np.stack(channels, axis=-1)

def imwrite(filename, image):
    """Save RGB float image in [0,1] to an 8-bit file."""
    image = np.clip(image, 0.0, 1.0) * 256.0
    image = image.astype(np.uint8)
    image = from_rgb2bgr(image)
    cv2.imwrite(filename, image)

def from_rgb2bgr(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

def kernelP(rggb):
    r, gr, gb, b = np.split(rggb, 4, axis=1)
    return np.concatenate(
        [rggb, rggb ** 2, r * gr, r * gb, r * b, gr * gb, gr * b, gb * b, r * gr * gb * b, np.ones_like(r)],
        axis=1
    )

def mapping(img, matrix):
    h, w, c = img.shape
    flat = np.reshape(img, (-1, 4))
    mapped = kernelP(flat) @ matrix
    return np.reshape(mapped, (h, w, c))

def from_rggb_to_rgb(rggb):
    """Simple visualization: merge G channels and reorder to RGB."""
    g = (rggb[:, :, 1] + rggb[:, :, 2]) / 2
    rggb[:, :, 1] = g
    rggb[:, :, 2] = rggb[:, :, 3]
    return rggb[:, :, :3]

BASE_DIR   = '/media/Data_2/r2r_odb/'
RESULT_DIR = '/media/Data_2/R2RResult/r2r-odb'

pair_data = ['paired/', 'unpaired/']
cameras   = ['huawei/', 'nikon/']

postfix_map = {
    'huawei/': '_A',
    'nikon/':  '_B',
}

meta_data_huawei = {'white_level': 4095,  'black_level': 256,
                    'cfa_pattern': np.array([2, 3, 1, 0])}
meta_data_nikon  = {'white_level': 16383, 'black_level': 1008,
                    'cfa_pattern': np.array([0, 1, 3, 2])}

for _pair in pair_data:
    for _cam in cameras:
        postfix = postfix_map[_cam]

        path_to_raw_data = os.path.join(BASE_DIR, _pair, _cam)
        out_rggb_dir = os.path.join(RESULT_DIR, _pair, _cam, 'raw-rggb/')
        out_vis_dir  = os.path.join(RESULT_DIR, _pair, _cam, 'vis/')

        check_dir(out_rggb_dir)
        check_dir(out_vis_dir)

        in_dir = os.path.join(path_to_raw_data, 'raw')
        if not os.path.isdir(in_dir):
            print(f'[WARN] Skip: directory not found: {in_dir}')
            continue

        all_raw_img_paths = [
            os.path.join(in_dir, f) for f in os.listdir(in_dir)
            if f.lower().endswith(('.dng', '.nef'))
        ]
        all_raw_img_paths.sort()

        meta_data = meta_data_huawei if _cam == 'huawei/' else meta_data_nikon

        for raw_img_path in all_raw_img_paths:
            try:
                with rawpy.imread(raw_img_path) as raw:
                    raw_bayer = raw.raw_image_visible.copy()
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

            raw_rggb = pack_rggb(raw_bayer_norm, deepcopy(meta_data['cfa_pattern']))

            stem = os.path.splitext(os.path.basename(raw_img_path))[0]

            savemat(os.path.join(out_rggb_dir, stem + postfix + '.mat'),
                    {"raw_rggb": raw_rggb})

            vis = (from_rggb_to_rgb(raw_rggb.copy()) * 0.9) ** (1 / 1.6)
            imwrite(os.path.join(out_vis_dir, stem + postfix + '.jpg'), vis)

            print(stem)
