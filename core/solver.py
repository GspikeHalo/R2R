"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
import time
import datetime
from munch import Munch

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
        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), data_parallel=True, **self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_children():
            if 'ema' not in name:
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

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

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, loaders.ref, 'train')
        fetcher_val = InputFetcher(loaders.val, None, 'val')
        inputs_val = next(fetcher_val)

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
                nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2])
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
                print(f"\n=== Iter {step}: sampling fixed val batch ===")

                x_fixed = inputs_val.x_src  # [B,4,H,W]
                imgs_to_log = []

                with torch.no_grad():
                    nets_ema.generator.eval()
                    nets_ema.style_encoder.eval()
                    for domain in range(args.num_domains):
                        c_t = torch.full(
                            (x_fixed.size(0),),
                            domain,
                            dtype=torch.long,
                            device=x_fixed.device
                        )
                        s_t = nets_ema.style_encoder(x_fixed, c_t)

                        x_fake = nets_ema.generator(x_fixed, s_t)  # [B,4,H,W]
                        x_fake_den = self.denorm(x_fake)

                        for idx in range(x_fake_den.size(0)):
                            fake_rgb = self.rggb2rgb(x_fake_den[idx])  # [3,H,W]
                            if args.use_wandb:
                                caption = (
                                    f"step{step} | "
                                    f"idx{idx} | "
                                    f"target_domain{domain}"
                                )
                                imgs_to_log.append(
                                    wandb.Image(fake_rgb, caption=caption)
                                )

                if args.use_wandb:
                    wandb.log({"val/fixed_samples": imgs_to_log}, step=step)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)

            # # compute FID and LPIPS if necessary
            if (i+1) % args.eval_every == 0:
                # calculate_metrics(nets_ema, args, i+1, mode='latent')
                # calculate_metrics(nets_ema, args, i+1, mode='reference')
                print(f"\n===Iter {i+1}: running test() ===")
                self.test(step=i+1)
                print(f"---Done test at iter (i+1) ===\n")

    @torch.no_grad()
    def test(self, step=None):
        if step is None:
            step = self.args.resume_iter

        self._load_checkpoint(step)
        self.generator_ema.eval()
        self.style_encoder_ema.eval()

        domains = sorted(os.listdir(self.args.val_img_dir))
        domain2idx = {d: i for i, d in enumerate(domains)}  # {iphone:0}
        pairs = [(s, t) for s in domains for t in domains if s != t]

        tot = {
            f"{s}->{t}": {"mae": 0.0, "psnr": 0.0, "ssim": 0.0, "count": 0}
            for s, t in pairs
        }
        tot_imgs = 0

        def _psnr(x, y, max_val=1.0, eps=1e-10):
            mse = ((x - y) ** 2).mean(dim=[1, 2, 3])
            return 10 * torch.log10(max_val ** 2 / (mse + eps))

        ssim_fn = SSIM(data_range=1.0, channel=4, size_average=False).to(self.device)

        for batch_i, (imgs_dict, filenames) in enumerate(
                tqdm(self.test_loader, desc="Testing")):
            B = next(iter(imgs_dict.values())).size(0)
            tot_imgs += B

            for d in domains:
                imgs_dict[d] = imgs_dict[d].to(self.device)

            for src, tgt in pairs:
                x_src = imgs_dict[src]
                x_tgt = imgs_dict[tgt]

                y_tgt = torch.full((B,), domain2idx[tgt],
                                   device=self.device, dtype=torch.long)

                s_t = self.style_encoder_ema(x_tgt, y_tgt)
                x_fake = self.generator_ema(x_src, s_t)
                x_fake_den = self.denorm(x_fake)
                x_tgt_den = self.denorm(x_tgt)

                mae = torch.abs(x_fake_den - x_tgt_den).view(B, -1) \
                    .mean(dim=1).sum().item()
                psnr = _psnr(x_fake_den, x_tgt_den).sum().item()
                ssim = ssim_fn(x_fake_den, x_tgt_den).sum().item()

                key = f"{src}->{tgt}"
                tot[key]["mae"] += mae
                tot[key]["psnr"] += psnr
                tot[key]["ssim"] += ssim
                tot[key]["count"] += B

                if not self.args.use_wandb:
                    trip = torch.stack([
                        self.rggb2rgb(self.denorm(x_src)[0]),
                        self.rggb2rgb(x_fake_den[0]),
                        self.rggb2rgb(x_tgt_den[0])
                    ], dim=0)
                    save_image(
                        trip,
                        f"{self.args.result_dir}/{src}2{tgt}_batch{batch_i}.png",
                        nrow=3
                    )

        # print per–pair metrics
        for key, v in tot.items():
            cnt = v["count"]
            print(f"{key}  MAE:{v['mae'] / cnt:.4f} "
                  f"PSNR:{v['psnr'] / cnt:.2f} "
                  f"SSIM:{v['ssim'] / cnt:.4f}")

        # print overall micro-average across all pairs
        total_mae = sum(v["mae"] for v in tot.values())
        total_psnr = sum(v["psnr"] for v in tot.values())
        total_ssim = sum(v["ssim"] for v in tot.values())
        total_count = sum(v["count"] for v in tot.values())

        avg_mae = total_mae / total_count
        avg_psnr = total_psnr / total_count
        avg_ssim = total_ssim / total_count

        print(f"Avg all  MAE:{avg_mae:.4f} "
              f"PSNR:{avg_psnr:.2f} "
              f"SSIM:{avg_ssim:.4f}")

        # -- WandB logging of evaluation metrics --
        if self.args.use_wandb:
            metrics = {}
            for key, v in tot.items():
                cnt = v["count"]
                metrics[f"{key}/MAE"] = v["mae"] / cnt
                metrics[f"{key}/PSNR"] = v["psnr"] / cnt
                metrics[f"{key}/SSIM"] = v["ssim"] / cnt
            # also log the micro-averages
            metrics["Avg/MAE"] = avg_mae
            metrics["Avg/PSNR"] = avg_psnr
            metrics["Avg/SSIM"] = avg_ssim
            wandb.log(metrics, step=step)

def compute_d_loss(nets, args, x_real, y_org, y_trg, x_ref):
    assert x_ref is not None

    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    with torch.no_grad():
        s_trg = nets.style_encoder(x_ref, y_trg)
        x_fake = nets.generator(x_real, s_trg)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())

def compute_g_loss(nets, args, x_real, y_org, y_trg, x_refs):
    x_ref, x_ref2 = x_refs

    # adversarial loss
    s_trg = nets.style_encoder(x_ref, y_trg)
    x_fake = nets.generator(x_real, s_trg)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    s_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2 = nets.generator(x_real, s_trg2)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    loss = loss_adv + args.lambda_sty * loss_sty \
        - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc
    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       cyc=loss_cyc.item())


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
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg