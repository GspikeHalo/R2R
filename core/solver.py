"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os, re
from os.path import join as ospj
import time
import datetime
from munch import Munch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM

from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision import transforms

import wandb

class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets, self.nets_ema = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        self.noise_losses = None
        if getattr(args, 'lambda_noise', 0.0) > 0 and getattr(args, 'noise_profile_paths', ''):
            paths = [p for p in args.noise_profile_paths.split(',') if p]
            self.noise_losses = nn.ModuleList([
                NoiseHistogramLoss(
                    p,
                    patch_size=args.noise_patch,
                    stride=args.noise_stride,
                    keep_ratio=args.noise_keep_ratio,
                    use_mad=args.noise_use_mad,
                    device=self.device
                ) for p in paths
            ])

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), data_parallel=True, **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]

            self.best_ckptios = [
                CheckpointIO(os.path.join(args.checkpoint_dir, 'best_nets.ckpt'), data_parallel=True, **self.nets),
                CheckpointIO(os.path.join(args.checkpoint_dir, 'best_nets_ema.ckpt'), data_parallel=True, **self.nets_ema),
                CheckpointIO(os.path.join(args.checkpoint_dir, 'best_optims.ckpt'), **self.optims),
            ]

            self.best_mae = float('inf')
            self.ssim_train = SSIM(data_range=1.0, channel=4, size_average=True).to(self.device)
        else:
            if self.args.best_model:
                ema_template = os.path.join(self.args.checkpoint_dir, 'best_nets_ema.ckpt')
                print("Eval mode: will load BEST EMA checkpoint")
            else:
                ema_template = ospj(self.args.checkpoint_dir, '{:06d}_nets_ema.ckpt')
                print("Eval mode: will load STEP EMA checkpoint")
            self.ckptios = [CheckpointIO(ema_template, data_parallel=True, **self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_children():
            if 'ema' not in name:
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

        self.ref_transform = transforms.Compose([
            transforms.Resize([self.args.img_size, self.args.img_size]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5],
                                 std =[0.5, 0.5, 0.5, 0.5]),
        ])

        self.val_ref_pool_cmeans = None  # {domain: [(abs_path, mean_vec[4]), ...]}

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def rggb2rgb(self, img):
        r, gr, gb, b = img[0], img[1], img[2], img[3]
        return torch.stack([r, 0.5 * (gr + gb), b], 0)

    def _to01(self, x_m11: torch.Tensor) -> torch.Tensor:
        return x_m11.mul(0.5).add(0.5).clamp(0, 1)

    def _channel_means(self, x_m11: torch.Tensor) -> torch.Tensor:
        """
        输入：[-1,1]，形状 [4,H,W] 或 [1,4,H,W]
        输出：通道均值向量 [4] in [0,1]（R, GR, GB, B）
        """
        if x_m11.dim() == 4 and x_m11.size(0) == 1:
            x_m11 = x_m11[0]
        x01 = self._to01(x_m11)
        return x01.view(4, -1).mean(dim=1).cpu()

    def _build_val_ref_pool_by_cmeans(self, root_dir: str):
        """
        为每个 domain 建立 [(abs_path, mean_vec[4]), ...]（使用 ref_transform 后计算）
        """
        root_p = self.args.val_img_dir if root_dir is None else root_dir
        root_p = os.path.abspath(root_p)
        pool = {}
        for d in sorted(os.listdir(root_p)):
            ddir = os.path.join(root_p, d)
            if not os.path.isdir(ddir):
                continue
            items = []
            for fn in sorted(os.listdir(ddir)):
                if not fn.endswith('.npy'):
                    continue
                pth = os.path.join(ddir, fn)
                arr = np.load(pth)                     # (4,H,W)
                ten = torch.from_numpy(arr).float()
                ten = self.ref_transform(ten)          # [-1,1]
                mv = self._channel_means(ten)          # [4] cpu
                items.append((os.path.abspath(pth), mv))
            pool[d] = items
        self.val_ref_pool_cmeans = pool

    def _select_ref_by_cmeans(self, domain: str, target_vec: torch.Tensor,
                              exclude_path: str) -> torch.Tensor:
        """
        在指定 domain 内按 ||mean_vec - target_vec||2 选取**单张**参考，排除 exclude_path。
        返回形状 [4,H,W]，已做 ref_transform，范围 [-1,1]。
        """
        assert self.val_ref_pool_cmeans is not None, "call _build_val_ref_pool_by_cmeans first."
        cands = [(p, m) for (p, m) in self.val_ref_pool_cmeans.get(domain, []) if p != exclude_path]
        if not cands:
            raise RuntimeError(f"No candidates in pool for domain={domain}")
        tv = target_vec.cpu()
        p_sel, _ = min(cands, key=lambda t: torch.linalg.norm(t[1] - tv).item())
        arr = np.load(p_sel)
        ten = torch.from_numpy(arr).float()
        ten = self.ref_transform(ten)
        return ten

    def _noise_loss_batch(self, x_fake, y_trg):
        if (self.noise_losses is None) or (self.args.lambda_noise <= 0):
            return x_fake.new_zeros([])
        x_lin = x_fake.mul(0.5).add(0.5).clamp(0, 1)
        if len(self.noise_losses) == 1:
            return self.noise_losses[0](x_lin)
        tot = x_fake.new_zeros([])
        used = 0
        for d in y_trg.unique():
            m = (y_trg == d)
            if m.any():
                tot = tot + self.noise_losses[d.item()](x_lin[m])
                used += 1
        return tot / max(used, 1)

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, loaders.ref, 'train')
        fetcher_val = InputFetcher(loaders.val, None, 'val')
        _ = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            x_real, y_org = inputs.x_src, inputs.y_src
            x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref

            d_loss, d_losses_ref = compute_d_loss(
                nets, args, x_real, y_org, y_trg, x_ref=x_ref)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            g_loss, g_losses_ref = compute_g_loss(
                nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2],
                noise_loss_fn=self._noise_loss_batch,
                ssim_fn=getattr(self, 'ssim_train', None)  # SSIM on [-1,1]
            )
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.style_encoder.step()

            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses_ref, g_losses_ref],
                                        ['D/ref_', 'G/ref_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

                if args.use_wandb:
                    wandb.log(all_losses, step=i+1)

            # generate images for debugging
            if (i+1) % args.sample_every == 0:
                step = i+1
                os.makedirs(args.sample_dir, exist_ok=True)
                print(f"\n=== Iter {step}: sampling ALL fixed val images ===")

                imgs_to_log = []

                with torch.no_grad():
                    nets_ema.generator.eval()
                    nets_ema.style_encoder.eval()

                    sample_fetcher = InputFetcher(loaders.val, loaders.ref, 'train')

                    for _ in range(len(loaders.val)):
                        batch = next(sample_fetcher)
                        x_fixed = batch.x_src.to(self.device)  # [B,4,H,W]
                        x_ref_all = batch.x_ref.to(self.device)  # [B,4,H,W]
                        y_ref_all = batch.y_ref.to(self.device)  # [B]
                        B = x_fixed.size(0)

                        for domain in range(args.num_domains):
                            mask = (y_ref_all == domain)
                            if mask.sum() == 0:
                                continue
                            x_ref_pool = x_ref_all[mask]
                            Kp = x_ref_pool.size(0)
                            if Kp < B:
                                idx = torch.randint(low=0, high=Kp, size=(B,), device=x_ref_pool.device)
                            elif Kp > B:
                                idx = torch.randperm(Kp, device=x_ref_pool.device)[:B]
                            else:
                                idx = torch.arange(B, device=x_ref_pool.device)
                            x_ref_d = x_ref_pool[idx]
                            c_t = torch.full((B,), domain, dtype=torch.long, device=x_fixed.device)
                            s_t = nets_ema.style_encoder(x_ref_d, c_t)
                            x_fake = nets_ema.generator(x_fixed, s_t)
                            x_fake_den = self.denorm(x_fake)

                            if args.use_wandb:
                                for b in range(B):
                                    fake_rgb = self.rggb2rgb(x_fake_den[b])
                                    caption = f"step{step} | batch_idx{b} | target_domain{domain}"
                                    imgs_to_log.append(wandb.Image(fake_rgb, caption=caption))

                if args.use_wandb and imgs_to_log:
                    wandb.log({"val/fixed_samples": imgs_to_log}, step=step)

            # save model checkpoints
            if (i + 1) % args.save_every == 0:
                step = i + 1
                for fn in os.listdir(self.args.checkpoint_dir):
                    if re.match(r'^\d+_.*\.ckpt$', fn):
                        os.remove(os.path.join(self.args.checkpoint_dir, fn))
                self._save_checkpoint(step=step)
                print(f"⇒ Saved checkpoint for iter {step}, old step files have been removed.")

            # # compute FID and LPIPS if necessary
            if (i+1) % args.eval_every == 0:
                print(f"\n===Iter {i+1}: running test() ===")
                current_mae = self.test(step=i+1)
                if current_mae < self.best_mae:
                    self.best_mae = current_mae
                    print(f"New best MAE {current_mae:.4f}, saving best checkpoints…")
                    for ckptio in self.best_ckptios:
                        ckptio.save(step=i + 1)
                print(f"---Done test at iter (i+1) ===\n")

    @torch.no_grad()
    def test(self, step=None):
        from itertools import combinations
        if step is None:
            step = self.args.resume_iter

        self._load_checkpoint(step)
        self.generator_ema.eval()
        self.style_encoder_ema.eval()

        self._build_val_ref_pool_by_cmeans(self.args.val_img_dir)

        os.makedirs(self.args.result_dir, exist_ok=True)
        trip_dir = os.path.join(self.args.result_dir, "triptychs")
        fake_root = os.path.join(self.args.result_dir, "fakes_by_target")
        os.makedirs(trip_dir, exist_ok=True)
        os.makedirs(fake_root, exist_ok=True)

        domains = sorted(
            d for d in os.listdir(self.args.val_img_dir)
            if os.path.isdir(os.path.join(self.args.val_img_dir, d))
        )
        domain2idx = {d: i for i, d in enumerate(domains)}
        pairs = list(combinations(domains, 2))

        def _psnr(x, y, max_val=1.0, eps=1e-10):
            mse = ((x - y) ** 2).mean(dim=[1, 2, 3])
            return 10 * torch.log10(max_val ** 2 / (mse + eps))

        def _sym_kl_hist(x, y, bins=256, eps=1e-8):
            B, C, H, W = x.shape
            out = x.new_zeros(B)
            # per-image / per-channel histogram KL
            for i in range(B):
                kls = []
                for c in range(C):
                    # flatten channel
                    xc = x[i, c].contiguous().view(-1)
                    yc = y[i, c].contiguous().view(-1)
                    # histograms on [0,1]
                    px = torch.histc(xc, bins=bins, min=0.0, max=1.0)
                    qx = torch.histc(yc, bins=bins, min=0.0, max=1.0)
                    # normalize to distributions
                    psum = px.sum()
                    qsum = qx.sum()
                    px = px / (psum + eps)
                    qx = qx / (qsum + eps)
                    # add epsilon for stability
                    px_ = px + eps
                    qx_ = qx + eps
                    # KL(P||Q) and KL(Q||P)
                    kld_pq = torch.sum(px_ * (torch.log(px_) - torch.log(qx_)))
                    kld_qp = torch.sum(qx_ * (torch.log(qx_) - torch.log(px_)))
                    kls.append(0.5 * (kld_pq + kld_qp))
                out[i] = torch.stack(kls).mean()
            return out  # [B]

        ssim_fn = SSIM(data_range=1.0, channel=4, size_average=False).to(self.device)
        kl_bins = int(getattr(self.args, 'kl_bins', 256))

        tot = {}
        for a, b in pairs:
            tot[f"{a}->{b}"] = {"mae": 0.0, "psnr": 0.0, "ssim": 0.0, "kl": 0.0, "count": 0}
            tot[f"{b}->{a}"] = {"mae": 0.0, "psnr": 0.0, "ssim": 0.0, "kl": 0.0, "count": 0}

        for batch_i, (imgs_dict, filenames) in enumerate(tqdm(self.test_loader, desc="Testing")):
            for d in domains:
                imgs_dict[d] = imgs_dict[d].to(self.device)

            B = next(iter(imgs_dict.values())).size(0)
            if B <= 1:
                continue

            fnames = list(filenames) if isinstance(filenames, (list, tuple)) else [filenames] * B

            for src, tgt in pairs:
                x_src = imgs_dict[src]
                x_tgt = imgs_dict[tgt]
                y_src = torch.full((B,), domain2idx[src], device=self.device, dtype=torch.long)
                y_tgt = torch.full((B,), domain2idx[tgt], device=self.device, dtype=torch.long)

                # ---- forward: src -> tgt，仅单参考（通道均值最近邻）
                refs = []
                for b in range(B):
                    tvec = self._channel_means(x_tgt[b])
                    exclude = os.path.abspath(os.path.join(self.args.val_img_dir, tgt, fnames[b]))
                    ref = self._select_ref_by_cmeans(tgt, tvec, exclude)   # [4,H,W]
                    refs.append(ref)
                x_ref = torch.stack(refs, dim=0).to(self.device)            # [B,4,H,W]
                s_t = self.style_encoder_ema(x_ref, y_tgt)                  # [B,S]
                x_fake = self.generator_ema(x_src, s_t, y_org=y_src, c_t=y_tgt)

                x_fake_den = self.denorm(x_fake)
                x_tgt_den  = self.denorm(x_tgt)

                mae_f  = torch.abs(x_fake_den - x_tgt_den).view(B, -1).mean(dim=1).sum().item()
                psnr_f = _psnr(x_fake_den, x_tgt_den).sum().item()
                ssim_f = ssim_fn(x_fake_den, x_tgt_den).sum().item()
                kl_f   = _sym_kl_hist(x_fake_den, x_tgt_den, bins=kl_bins).sum().item()

                key_fwd = f"{src}->{tgt}"
                tot[key_fwd]["mae"]   += mae_f
                tot[key_fwd]["psnr"]  += psnr_f
                tot[key_fwd]["ssim"]  += ssim_f
                tot[key_fwd]["kl"]    += kl_f
                tot[key_fwd]["count"] += B

                if not self.args.use_wandb:
                    b0 = 0
                    src_rgb  = self.rggb2rgb(self.denorm(x_src)[b0])
                    fake_rgb = self.rggb2rgb(x_fake_den[b0])
                    tgt_rgb  = self.rggb2rgb(x_tgt_den[b0])
                    ref_rgb  = self.rggb2rgb(self.denorm(x_ref)[b0])
                    panel = torch.stack([src_rgb, ref_rgb, fake_rgb, tgt_rgb], 0)
                    save_image(panel, os.path.join(trip_dir, f"{src}2{tgt}_batch{batch_i}.png"), nrow=4)
                    os.makedirs(os.path.join(fake_root, tgt), exist_ok=True)
                    save_image(fake_rgb, os.path.join(fake_root, tgt, f"{src}2{tgt}_batch{batch_i}_fake.png"))

                # ---- reverse: tgt -> src，仅单参考
                refs = []
                for b in range(B):
                    tvec = self._channel_means(x_src[b])
                    exclude = os.path.abspath(os.path.join(self.args.val_img_dir, src, fnames[b]))
                    ref = self._select_ref_by_cmeans(src, tvec, exclude)
                    refs.append(ref)
                x_ref_r = torch.stack(refs, dim=0).to(self.device)
                s_s = self.style_encoder_ema(x_ref_r, y_src)
                x_fake_rev = self.generator_ema(x_tgt, s_s, y_org=y_tgt, c_t=y_src)

                x_fake_rev_den = self.denorm(x_fake_rev)
                x_src_den = self.denorm(x_src)

                mae_r  = torch.abs(x_fake_rev_den - x_src_den).view(B, -1).mean(dim=1).sum().item()
                psnr_r = _psnr(x_fake_rev_den, x_src_den).sum().item()
                ssim_r = ssim_fn(x_fake_rev_den, x_src_den).sum().item()
                kl_r   = _sym_kl_hist(x_fake_rev_den, x_src_den, bins=kl_bins).sum().item()

                key_rev = f"{tgt}->{src}"
                tot[key_rev]["mae"]   += mae_r
                tot[key_rev]["psnr"]  += psnr_r
                tot[key_rev]["ssim"]  += ssim_r
                tot[key_rev]["kl"]    += kl_r
                tot[key_rev]["count"] += B

                if not self.args.use_wandb:
                    b0 = 0
                    tgt_rgb      = self.rggb2rgb(x_tgt_den[b0])
                    fake_rev_rgb = self.rggb2rgb(x_fake_rev_den[b0])
                    src_rgb      = self.rggb2rgb(x_src_den[b0])
                    ref_rev_rgb  = self.rggb2rgb(self.denorm(x_ref_r)[b0])
                    panel = torch.stack([tgt_rgb, ref_rev_rgb, fake_rev_rgb, src_rgb], 0)
                    save_image(panel, os.path.join(trip_dir, f"{tgt}2{src}_batch{batch_i}.png"), nrow=4)
                    os.makedirs(os.path.join(fake_root, src), exist_ok=True)
                    save_image(fake_rev_rgb, os.path.join(fake_root, src, f"{tgt}2{src}_batch{batch_i}_fake.png"))

        for key, v in tot.items():
            cnt = max(1, v["count"])
            print(f"{key}  MAE:{v['mae']/cnt:.4f} PSNR:{v['psnr']/cnt:.2f} "
                f"SSIM:{v['ssim']/cnt:.4f} KL:{v['kl']/cnt:.6f}")

        total_mae  = sum(v["mae"]  for v in tot.values())
        total_psnr = sum(v["psnr"] for v in tot.values())
        total_ssim = sum(v["ssim"] for v in tot.values())
        total_kl   = sum(v["kl"]   for v in tot.values())
        total_cnt  = sum(v["count"] for v in tot.values())

        avg_mae  = total_mae / total_cnt if total_cnt > 0 else float('nan')
        avg_psnr = total_psnr / total_cnt if total_cnt > 0 else float('nan')
        avg_ssim = total_ssim / total_cnt if total_cnt > 0 else float('nan')
        avg_kl   = total_kl   / total_cnt if total_cnt > 0 else float('nan')

        print(f"Avg all  MAE:{avg_mae:.4f} "
              f"PSNR:{avg_psnr:.2f} "
              f"SSIM:{avg_ssim:.4f} "
              f"KL:{avg_kl:.6f}")

        # -- WandB logging of evaluation metrics --
        if self.args.use_wandb:
            metrics = {}
            for key, v in tot.items():
                cnt = v["count"]
                if cnt == 0:
                    continue
                metrics[f"{key}/MAE"] = v["mae"] / cnt
                metrics[f"{key}/PSNR"] = v["psnr"] / cnt
                metrics[f"{key}/SSIM"] = v["ssim"] / cnt
                metrics[f"{key}/KL"] = v["kl"] / cnt
            metrics["Avg/MAE"] = avg_mae
            metrics["Avg/PSNR"] = avg_psnr
            metrics["Avg/SSIM"] = avg_ssim
            metrics["Avg/KL"] = avg_kl
            wandb.log(metrics, step=step)

        return avg_mae

def compute_d_loss(nets, args, x_real, y_org, y_trg, x_ref):
    assert x_ref is not None

    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    with torch.no_grad():
        s_trg = nets.style_encoder(x_ref, y_trg)
        x_fake = nets.generator(x_real, s_trg, y_org=y_org, c_t=y_trg)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())

def compute_g_loss(nets, args, x_real, y_org, y_trg, x_refs, noise_loss_fn=None, ssim_fn=None):
    x_ref, x_ref2 = x_refs

    # adversarial loss
    s_trg = nets.style_encoder(x_ref, y_trg)
    x_fake = nets.generator(x_real, s_trg, y_org=y_org, c_t=y_trg)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    s_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2 = nets.generator(x_real, s_trg2, y_org=y_org, c_t=y_trg)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, y_org=y_trg, c_t=y_org)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))
    if (ssim_fn is not None) and getattr(args, 'lambda_cyc_ssim', 0.0) > 0:
        loss_cyc_ssim = 1.0 - ssim_fn(x_rec, x_real)
    else:
        loss_cyc_ssim = x_real.new_zeros([])

    # identity (L1) + SSIM on [-1,1]
    lambda_id = getattr(args, 'lambda_id', 0.0)
    if lambda_id > 0:
        x_id = nets.generator(x_real, s_org, y_org=y_org, c_t=y_org)
        loss_id = torch.mean(torch.abs(x_id - x_real))
        if (ssim_fn is not None) and getattr(args, 'lambda_id_ssim', 0.0) > 0:
            loss_id_ssim = 1.0 - ssim_fn(x_id, x_real)
        else:
            loss_id_ssim = x_real.new_zeros([])
    else:
        x_id = None
        loss_id = x_real.new_zeros([])
        loss_id_ssim = x_real.new_zeros([])

    loss_noise = x_real.new_zeros([])
    if (noise_loss_fn is not None) and (getattr(args, 'lambda_noise', 0.0) > 0):
        loss_noise = noise_loss_fn(x_fake, y_trg)

    loss = (loss_adv
            + args.lambda_sty * loss_sty
            - args.lambda_ds * loss_ds
            + args.lambda_cyc * loss_cyc
            + getattr(args, 'lambda_noise', 0.0) * loss_noise
            + lambda_id * loss_id
            + getattr(args, 'lambda_cyc_ssim', 0.0) * loss_cyc_ssim
            + getattr(args, 'lambda_id_ssim', 0.0) * loss_id_ssim )

    return loss, Munch(
        adv=loss_adv.item(),
        sty=loss_sty.item(),
        ds=loss_ds.item(),
        cyc=loss_cyc.item(),
        noise=loss_noise.item(),
        id=loss_id.item(),
        cyc_ssim=(loss_cyc_ssim.item() if torch.is_tensor(loss_cyc_ssim) else 0.0),
        id_ssim=(loss_id_ssim.item() if torch.is_tensor(loss_id_ssim) else 0.0),
    )

def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

class NoiseHistogramLoss(nn.Module):
    def __init__(self, profile_path, patch_size=16, stride=None, keep_ratio=None, use_mad=True, device='cuda', eps=1e-8):
        super().__init__()
        prof = torch.load(profile_path, map_location=device)
        if isinstance(prof, dict) and 'profiles' in prof and 'bins' in prof:
            centers = prof['bins'].to(device).float()
            target = prof['profiles'].to(device).float()
            if keep_ratio is None:
                keep_ratio = float(prof.get('meta', {}).get('keep_ratio', 0.3))
        else:
            target = torch.as_tensor(prof, device=device).float().unsqueeze(0)
            centers = torch.linspace(0, 1, target.shape[-1], device=device)
            if keep_ratio is None:
                keep_ratio = 1.0
        edges = torch.empty(centers.numel() + 1, device=device)
        edges[0] = 0.0
        edges[-1] = 1.0
        if centers.numel() > 1:
            edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
        else:
            edges[1:-1] = 1.0
        self.patch_size = patch_size
        self.stride = patch_size if stride is None else stride
        self.keep_ratio = keep_ratio
        self.use_mad = use_mad
        self.eps = eps
        self.register_buffer('bin_edges', edges)
        self.register_buffer('target_profiles', target)
        self.register_buffer('bin_centers', centers)
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=device).view(1,1,3,3)
        ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=device).view(1,1,3,3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def _sobel_mag(self, x):
        b, c, h, w = x.shape
        gx = torch.nn.functional.conv2d(x, self.kx.repeat(c,1,1,1), padding=1, groups=c)
        gy = torch.nn.functional.conv2d(x, self.ky.repeat(c,1,1,1), padding=1, groups=c)
        return torch.sqrt(gx*gx + gy*gy)

    def _unfold(self, x):
        p = x.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride)
        b, c, nh, nw, ph, pw = p.shape
        return p.contiguous().view(b, c, nh*nw, ph*pw)

    def _patch_var(self, patches):
        if self.use_mad:
            med = patches.median(dim=-1, keepdim=True).values
            mad = (patches - med).abs().median(dim=-1).values
            sigma = mad / 0.67448975
            return sigma * sigma
        else:
            return patches.var(dim=-1, correction=0)

    def forward(self, x):
        B, C, H, W = x.shape
        Ct, Nb = self.target_profiles.shape
        if Ct == 1 and C > 1:
            tgt = self.target_profiles.expand(C, Nb)
        elif Ct != C:
            tgt = self.target_profiles.mean(dim=0, keepdim=True).expand(C, Nb)
        else:
            tgt = self.target_profiles
        patches = self._unfold(x.float())
        means = patches.mean(dim=-1)
        vars_ = self._patch_var(patches)
        g = self._sobel_mag(x.float())
        gmean = self._unfold(g).mean(dim=-1)
        if self.keep_ratio < 1.0:
            q = torch.quantile(gmean, self.keep_ratio, dim=-1, keepdim=True)
            mask = gmean <= q
        else:
            mask = torch.ones_like(gmean, dtype=torch.bool)
        means = torch.where(mask, means, torch.full_like(means, -1.0))
        vars_ = torch.where(mask, vars_, torch.zeros_like(vars_))
        sum_vars = x.new_zeros(B, C, Nb)
        counts = x.new_zeros(B, C, Nb)
        idx = torch.bucketize(means, self.bin_edges, right=True) - 1
        for b in range(B):
            for c in range(C):
                valid = (idx[b, c] >= 0) & (idx[b, c] < Nb) & mask[b, c]
                if valid.any():
                    ii = idx[b, c, valid].long()
                    sum_vars[b, c].index_add_(0, ii, vars_[b, c, valid])
                    counts[b, c].index_add_(0, ii, torch.ones(ii.numel(), device=x.device, dtype=sum_vars.dtype))
        hist = sum_vars / (counts + self.eps)
        bin_mask = counts > 0
        diff = (hist - tgt.unsqueeze(0)).abs() * bin_mask
        per_sc = diff.sum(dim=-1) / bin_mask.sum(dim=-1).clamp_min(1.0)
        loss = per_sc.mean()
        return loss