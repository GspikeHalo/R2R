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
from metrics.eval import calculate_metrics
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
                if net == 'fan':
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
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
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
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

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, loaders.ref, args.latent_dim, 'train')
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
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

            masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

            d_loss, d_losses_ref = compute_d_loss(
                nets, args, x_real, y_org, y_trg, x_ref=x_ref, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            g_loss, g_losses_ref = compute_g_loss(
                nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()

            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            # moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
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

            # # generate images for debugging
            if (i+1) % args.sample_every == 0:
                step = i+1
                os.makedirs(args.sample_dir, exist_ok=True)
                print(f"\n=== Iter {step}: sampling fixed val batch ===")

                x_fixed = inputs_val.x_src    # Shape [B,4,H,W]
                # 我们按每个目标域都做一次翻译
                imgs_to_log = []
                captions   = []
                with torch.no_grad():
                    nets_ema.generator.eval()
                    nets_ema.style_encoder.eval()
                    for domain in range(min(args.num_domains, 5)):  # 比如只看前5个域
                        c_t = torch.full((x_fixed.size(0),), domain,
                                        dtype=torch.long,
                                        device=x_fixed.device)
                        # 1) 用 EMA 的 style_encoder 生成风格向量
                        s_t = nets_ema.style_encoder(x_fixed, c_t)
                        # 2) 用 EMA 的 generator 做翻译
                        x_fake = nets_ema.generator(x_fixed, s_t)  # [B,4,H,W]
                        # 3) 取 batch 中第一个样本，做 RGGB→RGB，映射到 [0,1]
                        raw = x_fake[0]  # [4,H,W]
                        r, gr, gb, b = raw[0], raw[1], raw[2], raw[3]
                        g = 0.5 * (gr + gb)
                        rgb = torch.stack([r, g, b], dim=0)      # [3,H,W]
                        rgb = (rgb + 1) * 0.5                   # → [0,1]
                        arr = (rgb.permute(1,2,0).cpu().numpy() * 255).astype('uint8')
                        # 4) 转 PIL.Image 并累积
                        from PIL import Image
                        pil = Image.fromarray(arr)
                        imgs_to_log.append(pil)
                        captions.append(f"iter{step}_dom{domain}")

                # 5) 上传到 WandB
                if args.use_wandb:
                    wandb.log({
                        "val/fixed_samples": [
                            wandb.Image(img, caption=cap)
                            for img, cap in zip(imgs_to_log, captions)
                        ]
                    }, step=step)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)

            # # # compute FID and LPIPS if necessary
            # if (i+1) % args.eval_every == 0:
            #     # calculate_metrics(nets_ema, args, i+1, mode='latent')
            #     # calculate_metrics(nets_ema, args, i+1, mode='reference')
            #     print(f"\n===Iter {i+1}: running test() ===")
            #     self.test(step=i+1)
            #     print(f"---Done test at iter (i+1) ===\n")

    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))
        ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))

        fname = ospj(args.result_dir, 'reference.jpg')
        print('Working on {}...'.format(fname))
        utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)

        fname = ospj(args.result_dir, 'video_ref.mp4')
        print('Working on {}...'.format(fname))
        utils.video_ref(nets_ema, args, src.x, ref.x, ref.y, fname)

    @torch.no_grad()
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
        calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    @torch.no_grad()
    def test(self, step=None):
        """对成对数据进行正向 (O→T) 和 反向 (T→O) 的 MAE / PSNR / SSIM 评估，并保存三联图。"""
        # 1) 恢复 EMA 模型
        if step is None:
            step = self.args.resume_iter

        self._load_checkpoint(step)
        self.generator_ema.eval()
        self.style_encoder_ema.eval()

        # 2) 指标累加
        tot_mae_f = tot_psnr_f = tot_ssim_f = 0.0
        tot_mae_r = tot_psnr_r = tot_ssim_r = 0.0
        tot_imgs = 0

        # PSNR 计算
        def _psnr(x, y, max_val=1.0, eps=1e-10):
            mse = ((x - y)**2).mean(dim=[1,2,3])
            return 10 * torch.log10(max_val**2 / (mse + eps))

        # SSIM 函数（4 通道）
        ssim_fn = SSIM(data_range=1.0, channel=4, size_average=False).to(self.device)

        # RGGB → RGB 用于可视化
        def rggb2rgb(img):
            r, gr, gb, b = img[0], img[1], img[2], img[3]
            g = 0.5 * (gr + gb)
            return torch.stack([r, g, b], dim=0)

        # 3) 构建 domain→idx 映射（与 PairedNpyDataset 使用的子目录一致）
        domains = sorted(os.listdir(self.args.val_img_dir))
        domain2idx = {d:i for i,d in enumerate(domains)}

        # 4) 遍历 paired loader
        for x_o, x_t, filenames, domain_o_list, domain_t_list in tqdm(self.mydata_loader, desc='Testing'):
            B = x_o.size(0)
            tot_imgs += B

            x_o = x_o.to(self.device)
            x_t = x_t.to(self.device)

            # 4.1) 构造标签向量
            idx_o = domain2idx[domain_o_list[0]]  # PairedNpyDataset 保证同 batch 全一致
            idx_t = domain2idx[domain_t_list[0]]
            y_o = torch.full((B,), idx_o, device=self.device, dtype=torch.long)
            y_t = torch.full((B,), idx_t, device=self.device, dtype=torch.long)

            # —— Forward: O→T ——
            s_t        = self.style_encoder_ema(x_t, y_t)
            x_pred     = self.generator_ema(x_o, s_t)
            x_pred_den = self.denorm(x_pred)
            x_t_den    = self.denorm(x_t)

            mae_f  = torch.abs(x_pred_den - x_t_den).view(B, -1).mean(dim=1).sum().item()
            psnr_f = _psnr(x_pred_den, x_t_den).sum().item()
            ssim_f = ssim_fn(x_pred_den, x_t_den).sum().item()
            tot_mae_f  += mae_f
            tot_psnr_f += psnr_f
            tot_ssim_f += ssim_f

            # —— Reverse: T→O ——
            s_o     = self.style_encoder_ema(x_o, y_o)
            x_rev   = self.generator_ema(x_t, s_o)
            x_rev_den = self.denorm(x_rev)
            x_o_den   = self.denorm(x_o)

            mae_r  = torch.abs(x_rev_den - x_o_den).view(B, -1).mean(dim=1).sum().item()
            psnr_r = _psnr(x_rev_den, x_o_den).sum().item()
            ssim_r = ssim_fn(x_rev_den, x_o_den).sum().item()
            tot_mae_r  += mae_r
            tot_psnr_r += psnr_r
            tot_ssim_r += ssim_r

            # 5) 可视化三联图（每批次最多 5 张）
            V = min(5, B)
            for k in range(V):
                trip_f = torch.stack([
                    rggb2rgb(x_o_den[k]),
                    rggb2rgb(x_pred_den[k]),
                    rggb2rgb(x_t_den[k])
                ], dim=0)
                save_image(trip_f,
                           f"{self.args.result_dir}/batch_{k:03d}_fwd.png",
                           nrow=3)

                trip_r = torch.stack([
                    rggb2rgb(x_t_den[k]),
                    rggb2rgb(x_rev_den[k]),
                    rggb2rgb(x_o_den[k])
                ], dim=0)
                save_image(trip_r,
                           f"{self.args.result_dir}/batch_{k:03d}_rev.png",
                           nrow=3)

        # 6) 打印平均指标
        print(f'Forward  MAE:{tot_mae_f/tot_imgs:.4f}, '
              f'PSNR:{tot_psnr_f/tot_imgs:.2f}, '
              f'SSIM:{tot_ssim_f/tot_imgs:.4f}')
        print(f'Reverse  MAE:{tot_mae_r/tot_imgs:.4f}, '
              f'PSNR:{tot_psnr_r/tot_imgs:.2f}, '
              f'SSIM:{tot_ssim_r/tot_imgs:.4f}')


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)

        x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())

def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)

    x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=masks)
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