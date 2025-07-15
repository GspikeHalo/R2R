#!/usr/bin/env python

import argparse
import collections
import os

import tqdm
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from uvcgan2.consts import MERGE_NONE, MERGE_PAIRED
from uvcgan2.eval.funcs import (
    load_eval_model_dset_from_cmdargs, tensor_to_image, slice_data_loader,
    get_eval_savedir, make_image_subdirs
)
from uvcgan2.utils.parsers import (
    add_standard_eval_parsers, add_plot_extension_parser
)

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description='Save model predictions as triplets and compute MAE/SSIM/PSNR'
    )
    add_standard_eval_parsers(parser)
    add_plot_extension_parser(parser)
    return parser.parse_args()

def _psnr_np(x: np.ndarray, y: np.ndarray, max_val: float = 1.0, eps: float = 1e-10) -> float:
    mse = np.mean((x - y) ** 2)
    return 10 * np.log10(max_val**2 / (mse + eps))

def rggb2rgb_np(arr: np.ndarray) -> np.ndarray:
    r, gr, gb, b = arr[0], arr[1], arr[2], arr[3]
    g = 0.5 * (gr + gb)
    return np.stack([r, g, b], axis=0)

def save_triplets_and_metrics(images, savedir, sample_counter, ext, domain, metrics):
    if domain == 0:
        src_t, fake_t, tgt_t = images.real_a, images.fake_b, images.real_b
        key = 'fwd'
    else:
        src_t, fake_t, tgt_t = images.real_b, images.fake_a, images.real_a
        key = 'rev'

    B = src_t.shape[0]
    for i in range(B):
        idx = sample_counter[domain]
        src = tensor_to_image(src_t[i])
        fake = tensor_to_image(fake_t[i])
        tgt = tensor_to_image(tgt_t[i])

        # MAE
        mae = np.mean(np.abs(fake - tgt))
        # PSNR
        psnr = _psnr_np(fake, tgt)
        # SSIM：确保 win_size 是不超过最小边、且为奇数
        h, w = tgt.shape[1], tgt.shape[2]
        max_win = min(h, w)
        win = min(7, max_win if max_win % 2 == 1 else max_win - 1)
        win = max(win, 3)  # 最小也要 3
        ssim = structural_similarity(
            tgt.transpose(1,2,0),
            fake.transpose(1,2,0),
            channel_axis=2,
            data_range=1.0,
            win_size=win
        )

        metrics[key]['mae_sum']  += mae
        metrics[key]['ssim_sum'] += ssim
        metrics[key]['psnr_sum'] += psnr
        metrics[key]['count']    += 1

        # 拼接三联图（RGB 可视化）
        src_rgb  = rggb2rgb_np(src).transpose(1,2,0)
        fake_rgb = rggb2rgb_np(fake).transpose(1,2,0)
        tgt_rgb  = rggb2rgb_np(tgt).transpose(1,2,0)
        trip = np.concatenate([
            (src_rgb * 255).round().astype(np.uint8),
            (fake_rgb * 255).round().astype(np.uint8),
            (tgt_rgb * 255).round().astype(np.uint8)
        ], axis=1)
        img = Image.fromarray(trip)

        subdir = os.path.join(savedir, f'domain_{domain}')
        os.makedirs(subdir, exist_ok=True)
        base = os.path.join(subdir, f'triplet_{idx}')
        for e in ext:
            img.save(base + f'.{e}')

        sample_counter[domain] += 1

def dump_paired_images(model, loader, n_eval, batch_size, savedir, ext, metrics):
    data_loader, steps = slice_data_loader(loader, batch_size, n_eval)
    sample_counter = collections.defaultdict(int)
    for idx, (batchA, batchB) in enumerate(tqdm.tqdm(data_loader, total=steps, desc='Paired Eval')):
        batchA = batchA.to(model.device, non_blocking=True)
        batchB = batchB.to(model.device, non_blocking=True)

        # —— A -> B ——
        # set real_a from batchA
        model.set_input(batchA, domain=0)
        # manually inject real_b from batchB so we have targets
        model.images.real_b = batchB
        model.forward_nograd()
        save_triplets_and_metrics(model.images, savedir, sample_counter, ext, 0, metrics)

        # —— B -> A ——
        model.set_input(batchB, domain=1)
        model.images.real_a = batchA
        model.forward_nograd()
        save_triplets_and_metrics(model.images, savedir, sample_counter, ext, 1, metrics)
def main():
    cmdargs = parse_cmdargs()

    # 强制成对模式
    args, model, loader, evaldir = load_eval_model_dset_from_cmdargs(
        cmdargs, merge_type=MERGE_PAIRED
    )
    args.config.data.merge_type = MERGE_PAIRED

    savedir = get_eval_savedir(
        evaldir, 'triplets', cmdargs.model_state, cmdargs.split
    )
    make_image_subdirs(model, savedir)

    metrics = {
        'fwd': {'mae_sum':0.0, 'ssim_sum':0.0, 'psnr_sum':0.0, 'count':0},
        'rev': {'mae_sum':0.0, 'ssim_sum':0.0, 'psnr_sum':0.0, 'count':0},
    }

    dump_paired_images(
        model, loader, cmdargs.n_eval, args.batch_size,
        savedir, cmdargs.ext, metrics
    )

    for key in ('fwd','rev'):
        cnt = metrics[key]['count']
        if cnt > 0:
            print(f"{key.upper()} avg MAE:  {metrics[key]['mae_sum']/cnt:.6f}")
            print(f"{key.upper()} avg SSIM: {metrics[key]['ssim_sum']/cnt:.6f}")
            print(f"{key.upper()} avg PSNR: {metrics[key]['psnr_sum']/cnt:.6f}")
        else:
            print(f"No {key} samples.")

if __name__ == '__main__':
    main()


# def parse_cmdargs():
#     parser = argparse.ArgumentParser(
#         description='Save model predictions as triplets and compute MAE/SSIM/PSNR'
#     )
#     add_standard_eval_parsers(parser)
#     add_plot_extension_parser(parser)
#     return parser.parse_args()
#
# def _psnr_np(x: np.ndarray, y: np.ndarray, max_val: float = 1.0, eps: float = 1e-10) -> float:
#     """Compute PSNR between two NumPy arrays in [0,1]."""
#     mse = np.mean((x - y) ** 2)
#     return 10 * np.log10(max_val**2 / (mse + eps))
#
# def rggb2rgb_np(arr: np.ndarray) -> np.ndarray:
#     """
#     Convert a 4-channel RGGB image (C,H,W) numpy array in [0,1]
#     to a 3-channel RGB array in [0,1].
#     """
#     # arr shape: (4, H, W)
#     r, gr, gb, b = arr[0], arr[1], arr[2], arr[3]
#     g = 0.5 * (gr + gb)
#     rgb = np.stack([r, g, b], axis=0)
#     return rgb
#
# def save_triplets_and_metrics(images, savedir, sample_counter, ext, domain, metrics):
#     """
#     For each sample in the batch:
#     - assemble triplet (src | fake | tgt) and save
#     - compute MAE, SSIM, PSNR between fake and tgt, accumulate into metrics
#     domain==0: forward (A->B), domain==1: reverse (B->A)
#     """
#     # pick tensors
#     if domain == 0:
#         src_t, fake_t, tgt_t = images.real_a, images.fake_b, images.real_b
#         key = 'fwd'
#     else:
#         src_t, fake_t, tgt_t = images.real_b, images.fake_a, images.real_a
#         key = 'rev'
#
#     B = src_t.shape[0]
#     for i in range(B):
#         idx = sample_counter[domain]
#
#         # convert to numpy [0,1]
#         src = tensor_to_image(src_t[i])   # shape (4,H,W), float in [0,1]
#         fake = tensor_to_image(fake_t[i])
#         tgt = tensor_to_image(tgt_t[i])
#
#         # convert RGGB→RGB for triplet
#         src_rgb  = rggb2rgb_np(src)
#         fake_rgb = rggb2rgb_np(fake)
#         tgt_rgb  = rggb2rgb_np(tgt)
#
#         # compute metrics on 4-channel raw
#         # flatten all channels & pixels
#         mae = np.mean(np.abs(fake - tgt))
#         ssim = structural_similarity(
#             tgt.transpose(1,2,0), fake.transpose(1,2,0),
#             multichannel=True, data_range=1.0
#         )
#         psnr = _psnr_np(fake, tgt, max_val=1.0)
#
#         metrics[key]['mae_sum']  += mae
#         metrics[key]['ssim_sum'] += ssim
#         metrics[key]['psnr_sum'] += psnr
#         metrics[key]['count']    += 1
#
#         # assemble triplet image (H, W*3, 3), convert to uint8
#         trip = np.concatenate([
#             (src_rgb.transpose(1,2,0) * 255).round().astype(np.uint8),
#             (fake_rgb.transpose(1,2,0) * 255).round().astype(np.uint8),
#             (tgt_rgb.transpose(1,2,0) * 255).round().astype(np.uint8),
#         ], axis=1)
#         img = Image.fromarray(trip)
#
#         # save
#         subdir = os.path.join(savedir, f'domain_{domain}')
#         os.makedirs(subdir, exist_ok=True)
#         base = os.path.join(subdir, f'triplet_{idx}')
#         for e in ext:
#             img.save(base + '.' + e)
#
#         sample_counter[domain] += 1
#
# def dump_single_domain_images(
#     model, data_it, domain, n_eval, batch_size, savedir, sample_counter, ext, metrics
# ):
#     data_it, steps = slice_data_loader(data_it, batch_size, n_eval)
#     desc = f'Domain {domain}'
#     for batch in tqdm.tqdm(data_it, desc=desc, total=steps):
#         model.set_input(batch, domain=domain)
#         model.forward_nograd()
#         save_triplets_and_metrics(
#             model.images, savedir, sample_counter,
#             ext, domain, metrics
#         )
#
# def dump_paired_images(model, loader, n_eval, batch_size, savedir, ext, metrics):
#     loader, steps = slice_data_loader(loader, batch_size, n_eval)
#     for idx, (batchA, batchB) in enumerate(tqdm.tqdm(loader, total=steps)):
#         # 1) move to GPU
#         for k in batchA: batchA[k] = batchA[k].to(model.device)
#         for k in batchB: batchB[k] = batchB[k].to(model.device)
#         # 2) A→B
#         model.set_input(batchA, domain=0)
#         model.forward_nograd()
#         # …这里调用你已有的保存+指标计算函数，记得传入 model.images…
#         # 3) B→A
#         model.set_input(batchB, domain=1)
#         model.forward_nograd()
#         # …再一次保存+计算…
#
# def dump_images(model, data_list, n_eval, batch_size, savedir, ext, metrics):
#     make_image_subdirs(model, savedir)
#     sample_counter = collections.defaultdict(int)
#     if isinstance(ext, str):
#         ext = [ext]
#
#     for domain, data_it in enumerate(data_list):
#         dump_single_domain_images(
#             model, data_it, domain, n_eval, batch_size,
#             savedir, sample_counter, ext, metrics
#         )
#
# def main():
#     cmdargs = parse_cmdargs()
#
#     args, model, data_loader, evaldir = load_eval_model_dset_from_cmdargs(
#         cmdargs, merge_type=MERGE_NONE
#     )
#
#     savedir = get_eval_savedir(
#         evaldir, 'triplets', cmdargs.model_state, cmdargs.split
#     )
#
#     # init metrics
#     metrics = {
#         'fwd': {'mae_sum':0.0, 'ssim_sum':0.0, 'psnr_sum':0.0, 'count':0},
#         'rev': {'mae_sum':0.0, 'ssim_sum':0.0, 'psnr_sum':0.0, 'count':0},
#     }
#
#     dump_paired_images(
#         model, data_loader,
#         cmdargs.n_eval, args.batch_size,
#         savedir, cmdargs.ext, metrics
#     )
#
#     # print averages
#     for key in ('fwd','rev'):
#         cnt = metrics[key]['count']
#         if cnt > 0:
#             print(f"{key.upper()} avg MAE:  {metrics[key]['mae_sum']/cnt:.6f}")
#             print(f"{key.upper()} avg SSIM: {metrics[key]['ssim_sum']/cnt:.6f}")
#             print(f"{key.upper()} avg PSNR: {metrics[key]['psnr_sum']/cnt:.6f}")
#         else:
#             print(f"No {key} samples.")
#
# if __name__ == '__main__':
#     main()


# def parse_cmdargs():
#     parser = argparse.ArgumentParser(
#         description = 'Save model predictions as images'
#     )
#     add_standard_eval_parsers(parser)
#     add_plot_extension_parser(parser)
#
#     return parser.parse_args()
#
# def save_images(model, savedir, sample_counter, ext):
#     for (name, torch_image) in model.images.items():
#         if torch_image is None:
#             continue
#
#         for index in range(torch_image.shape[0]):
#             sample_index = sample_counter[name]
#
#             image = tensor_to_image(torch_image[index])
#             image = np.round(255 * image).astype(np.uint8)
#             image = Image.fromarray(image)
#
#             path  = os.path.join(savedir, name, f'sample_{sample_index}')
#             for e in ext:
#                 image.save(path + '.' + e)
#
#             sample_counter[name] += 1
#
# def dump_single_domain_images(
#     model, data_it, domain, n_eval, batch_size, savedir, sample_counter, ext
# ):
#     # pylint: disable=too-many-arguments
#     data_it, steps = slice_data_loader(data_it, batch_size, n_eval)
#     desc = f'Translating domain {domain}'
#
#     for batch in tqdm.tqdm(data_it, desc = desc, total = steps):
#         model.set_input(batch, domain = domain)
#         model.forward_nograd()
#
#         save_images(model, savedir, sample_counter, ext)
#
# def dump_images(model, data_list, n_eval, batch_size, savedir, ext):
#     # pylint: disable=too-many-arguments
#     make_image_subdirs(model, savedir)
#
#     sample_counter = collections.defaultdict(int)
#     if isinstance(ext, str):
#         ext = [ ext, ]
#
#     for domain, data_it in enumerate(data_list):
#         dump_single_domain_images(
#             model, data_it, domain, n_eval, batch_size, savedir,
#             sample_counter, ext
#         )
#
# def main():
#     cmdargs = parse_cmdargs()
#
#     args, model, data_list, evaldir = load_eval_model_dset_from_cmdargs(
#         cmdargs, merge_type = MERGE_NONE
#     )
#
#     if not isinstance(data_list, (list, tuple)):
#         data_list = [ data_list, ]
#
#     savedir = get_eval_savedir(
#         evaldir, 'images', cmdargs.model_state, cmdargs.split
#     )
#
#     dump_images(
#         model, data_list, cmdargs.n_eval, args.batch_size, savedir,
#         cmdargs.ext
#     )
#
# if __name__ == '__main__':
#     main()

