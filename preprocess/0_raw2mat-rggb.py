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
    idx = [[0, 0], [0, 1], [1, 0], [1, 1]]

    cfa = np.array(cfa, dtype=int).copy()

    if int(cfa.max()) < 3:
        cfa[cfa == 2] += 1
        cfa[2:][cfa[2:] == 1] += 1
        cfa_pos = cfa
    else:
        cfa_pos = cfa
    channels = []
    for c in cfa_pos:
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

def rggb4_to_bayer(rggb4, cfa):
    assert rggb4.ndim == 3 and rggb4.shape[2] == 4, f"Expect (h,w,4), got {rggb4.shape}"
    h, w, _ = rggb4.shape
    cfa_t = cfa.copy()
    cfa_t[cfa_t == 2] += 1
    cfa_t[2:][cfa_t[2:] == 1] += 1
    idx = [[0, 0], [0, 1], [1, 0], [1, 1]]
    mosaic = np.empty((h * 2, w * 2), dtype=rggb4.dtype)
    for k, c in enumerate(cfa_t):
        rr, cc = idx[c]
        mosaic[rr::2, cc::2] = rggb4[:, :, k]
    return mosaic

BASE_DIR   = '/media/Data_2/r2r_odb/'
RESULT_DIR = '/media/Data_2/R2RResult/r2r-odb'

pair_data = ['paired/', 'unpaired/']
cameras   = ['huawei/', 'nikon/', 'iphone/', 'samsung/']

postfix_map = {
    'huawei/': '_A',
    'nikon/':  '_B',
    'iphone/': '_C',
    'samsung/': '_D',
}

meta_data_huawei = {'white_level': 4095,  'black_level': 256,
                    'cfa_pattern': np.array([2, 3, 1, 0])}
meta_data_nikon  = {'white_level': 16383, 'black_level': 1008,
                    'cfa_pattern': np.array([0, 1, 3, 2])}
meta_data_iphone   = {'white_level': 65535, 'black_level': 0,
                      'cfa_pattern': np.array([0, 1, 2, 1])}
meta_data_samsung  = {'white_level': 4095,  'black_level': 0,
                      'cfa_pattern': np.array([0, 1, 2, 1])}

camera_meta = {
    'huawei/':  meta_data_huawei,
    'nikon/':   meta_data_nikon,
    'iphone/':  meta_data_iphone,
    'samsung/': meta_data_samsung,
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
                    raw_data = raw.raw_image_visible.copy()
            except Exception as e:
                print(f'[ERROR] Failed to read: {raw_img_path} -> {e}')
                continue

            if raw_data.ndim == 3 and raw_data.shape[2] == 4:
                try:
                    raw_bayer = rggb4_to_bayer(raw_data, deepcopy(meta_data['cfa_pattern']))
                except Exception as e:
                    print(f'[ERROR] RGGB4->Bayer failed: {raw_img_path} -> {e}')
                    continue
            else:
                raw_bayer = raw_data

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
