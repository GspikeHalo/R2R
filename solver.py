from model import Generator
from generator import Conformer
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from pytorch_msssim import SSIM
from tqdm import tqdm
import wandb


# 修改one hot为1， 0， -1
class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, mydata_loader, test_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.mydata_loader = mydata_loader
        self.test_loader = test_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lambda_id = config.lambda_id

        # Training configurations.
        # self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step
        self.eval_step = config.eval_step

        self.crop_size = 256
        self.attr2idx = {}
        self.image_dir = config.mydata_image_dir
        self.attr_path = config.mydata_attr_path

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        # g_conv_dim 第一层卷积通道数， g_repeat_num 残差块重复次数
        # self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        self.G = Conformer()
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        c_trg_list = []

        for i in range(c_dim):
            # Create an all-zero tensor of same shape as c_org
            c_trg = torch.zeros_like(c_org)
            c_trg[:, i] = 1  # Set only the i-th domain as 1
            c_trg_list.append(c_trg.to(self.device))

        return c_trg_list

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

    def rggb_to_rgb_tensor(self, x_rggb: torch.Tensor) -> torch.Tensor:
        # x_rggb.shape == (B, 4, H, W), 通道顺序 [R, Gr, Gb, B]
        r = x_rggb[:, 0, :, :]
        gr = x_rggb[:, 1, :, :]
        gb = x_rggb[:, 2, :, :]
        b = x_rggb[:, 3, :, :]
        g = 0.5 * (gr + gb)
        return torch.stack([r, g, b], dim=1)

    def get_fixed_samples(self, filenames):
        """手动读取固定的图片及其标签，返回Tensor"""
        imgs = []
        labels = []

        # 构建 attr2idx，确保一致
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        attr2idx = {attr_name: i for i, attr_name in enumerate(all_attr_names)}

        for fname in filenames:
            img_path = os.path.join(self.image_dir, fname)
            img = np.load(img_path).astype(np.float32)

            C, H, W = img.shape
            if H >= self.crop_size and W >= self.crop_size:
                cy = (H - self.crop_size) // 2
                cx = (W - self.crop_size) // 2
                img = img[:, cy:cy + self.crop_size, cx:cx + self.crop_size]

            if H != self.image_size or W != self.image_size:
                img_tensor = torch.from_numpy(img).unsqueeze(0)
                img_tensor = F.interpolate(img_tensor, size=(self.image_size, self.image_size),
                                           mode='bilinear', align_corners=False)[0]
            else:
                img_tensor = torch.from_numpy(img)

            img_tensor = img_tensor * 2.0 - 1.0
            imgs.append(img_tensor)

            # 读取标签
            label = []
            attr_line = self.filename_to_attr(fname)
            for attr_name in self.selected_attrs:
                idx = attr2idx[attr_name]
                label.append(1.0 if attr_line[idx] == '1' else 0.0)
            labels.append(label)

        imgs = torch.stack(imgs)
        labels = torch.FloatTensor(labels)
        return imgs.to(self.device), labels.to(self.device)

    def filename_to_attr(self, filename):
        with open(self.attr_path, 'r') as f:
            lines = f.readlines()[2:]  # 跳过头两行
            for line in lines:
                if line.startswith(filename):
                    return line.strip().split()[1:]
        raise ValueError(f"未找到 {filename} 对应的标签行")

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        data_loader = self.mydata_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        # x_fixed, c_org = next(data_iter)
        # x_fixed = x_fixed.to(self.device)
        # c_fixed_list = self.create_labels(c_org, self.c_dim, self.selected_attrs)
        fixed_filenames = [
            '3054_A.npy', '5889_A.npy', '2285_A.npy', '3574_A.npy',
            '2798_A.npy', '6143_A.npy', '2540_A.npy', '4346_A.npy',
            '2797_B.npy', '5890_B.npy', '3316_B.npy', '3053_B.npy',
            '279_B.npy', '588_B.npy', '3055_B.npy', '3310_B.npy'
        ]

        x_fixed, c_org = self.get_fixed_samples(fixed_filenames)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr
        best_mae = float('inf')
        best_iter = 0

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)
            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            c_org = label_org.clone()
            c_trg = label_trg.clone()

            x_real = x_real.to(self.device)  # Input images.
            c_org = c_org.to(self.device)  # Original domain labels.
            c_trg = c_trg.to(self.device)  # Target domain labels.
            label_org = label_org.to(self.device)  # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)  # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org)

            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()  # D在real图像的得分，应趋于稳定，在固定范围内波动 -1 左右
            loss['D/loss_fake'] = d_loss_fake.item()  # D在fake图像的得分，应尽可能小，在固定范围内波动 +1 左右
            loss['D/loss_cls'] = d_loss_cls.item()  # D的属性分类损失，应尽可能小 < 0.1
            loss['D/loss_gp'] = d_loss_gp.item()  # 接近0

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg)

                x_id = self.G(x_real, c_org)
                g_loss_id = torch.mean(torch.abs(x_real - x_id))

                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls + self.lambda_id * g_loss_id
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()  # G的对抗损失，下降趋于平稳 约等于0
                loss['G/loss_rec'] = g_loss_rec.item()  # 循环一致，下降
                loss['G/loss_cls'] = g_loss_cls.item()  # 分类损失，下降
                loss['G/loss_id'] = g_loss_id.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i + 1)
                wandb.log(loss, step=i + 1)

            if (i + 1) % self.sample_step == 0:
                x_fake_fixed = [
                    self.G(x_fixed, c_t) for c_t in c_fixed_list
                ]

                x_fake_rgb = [self.rggb_to_rgb_tensor(img) for img in x_fake_fixed]

                # 3) log 到 WandB
                wandb.log({
                    "fixed_generated": [
                        wandb.Image(img, caption=str(c_t.tolist()))
                        for img, c_t in zip(x_fake_rgb, c_fixed_list)
                    ]
                }, step=i + 1)

            if (i + 1) % self.eval_step == 0:
                mae_f, psnr_f, ssim_f, mae_r, psnr_r, ssim_r = self.evaluate()
                avg_mae = (mae_f + mae_r) / 2.0
                avg_psnr = (psnr_f + psnr_r) / 2.0
                avg_ssim = (ssim_f + ssim_r) / 2.0

                # print(f"[Iter {i+1}] Eval MAE={mae:.4f}, PSNR={psnr:.2f}, SSIM={ssim:.4f}")
                wandb.log({
                    "eval/avg_MAE": avg_mae,
                    "eval/avg_PSNR": avg_psnr,
                    "eval/avg_SSIM": avg_ssim,
                    "eval/forward_MAE": mae_f,
                    "eval/forward_PSNR": psnr_f,
                    "eval/forward_SSIM": ssim_f,
                    "eval/reverse_MAE": mae_r,
                    "eval/reverse_PSNR": psnr_r,
                    "eval/reverse_SSIM": ssim_r,
                }, step=i + 1)

                # 如果当前 MAE 更低，就更新 best 记录并保存 ckpt
                if avg_mae < best_mae:
                    best_mae = avg_mae
                    best_iter = i + 1
                    G_path = os.path.join(self.model_save_dir, f'best-G-iter{best_iter}.ckpt')
                    D_path = os.path.join(self.model_save_dir, f'best-D-iter{best_iter}.ckpt')
                    torch.save(self.G.state_dict(), G_path)
                    torch.save(self.D.state_dict(), D_path)
                    print(f"  → New best MAE {best_mae:.4f} at iter {best_iter}, model saved.")
                    # if (i + 1) % self.eval_step == 0:
            #     mae, psnr, ssim = self.evaluate()
            #     print(f"[Iter {i+1}] Eval MAE={mae:.4f}, PSNR={psnr:.2f}, SSIM={ssim:.4f}")
            #     wandb.log({
            #         "eval/MAE": mae,
            #         "eval/PSNR": psnr,
            #         "eval/SSIM": ssim
            #     }, step=i + 1)

            #     # 如果当前 MAE 更低，就更新 best 记录并保存 ckpt
            #     if mae < best_mae:
            #         best_mae = mae
            #         best_iter = i + 1
            #         G_path = os.path.join(self.model_save_dir, f'best-G-iter{best_iter}.ckpt')
            #         D_path = os.path.join(self.model_save_dir, f'best-D-iter{best_iter}.ckpt')
            #         torch.save(self.G.state_dict(), G_path)
            #         torch.save(self.D.state_dict(), D_path)
            #         print(f"  → New best MAE {best_mae:.4f} at iter {best_iter}, model saved.")

            # # Save model checkpoints.
            # if (i + 1) % self.model_save_step == 0:
            #     G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i + 1))
            #     D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i + 1))
            #     torch.save(self.G.state_dict(), G_path)
            #     torch.save(self.D.state_dict(), D_path)
            #     print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def evaluate(self):
        """返回 (mae_f, psnr_f, ssim_f, mae_r, psnr_r, ssim_r)。"""
        self.G.eval()
        total_mae_f = total_psnr_f = total_ssim_f = 0.0
        total_mae_r = total_psnr_r = total_ssim_r = 0.0
        total_imgs = 0

        def _psnr_per_image(x, y, max_val=1.0, eps=1e-10):
            mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
            return 10 * torch.log10(max_val ** 2 / (mse + eps))

        ssim_fn = SSIM(data_range=1.0, channel=4, size_average=False).to(self.device)

        with torch.no_grad():
            for x_o, x_t, c_o, c_t, _, _ in tqdm(self.test_loader, desc='Evaluating'):
                B = x_o.size(0)
                total_imgs += B
                x_o, x_t = x_o.to(self.device), x_t.to(self.device)
                c_o, c_t = c_o.to(self.device), c_t.to(self.device)

                # 正向：原域 → 目标域
                x_pred = self.G(x_o, c_t)
                x_pred_den, x_t_den = self.denorm(x_pred), self.denorm(x_t)
                per_mae_f = torch.abs(x_pred_den - x_t_den).view(B, -1).mean(1)
                total_mae_f += per_mae_f.sum().item()
                total_psnr_f += _psnr_per_image(x_pred_den, x_t_den).sum().item()
                total_ssim_f += ssim_fn(x_pred_den, x_t_den).sum().item()

                # 反向：目标域 → 原域
                x_rev = self.G(x_t, c_o)
                x_rev_den, x_o_den = self.denorm(x_rev), self.denorm(x_o)
                per_mae_r = torch.abs(x_rev_den - x_o_den).view(B, -1).mean(1)
                total_mae_r += per_mae_r.sum().item()
                total_psnr_r += _psnr_per_image(x_rev_den, x_o_den).sum().item()
                total_ssim_r += ssim_fn(x_rev_den, x_o_den).sum().item()

        mae_f = total_mae_f / total_imgs
        psnr_f = total_psnr_f / total_imgs
        ssim_f = total_ssim_f / total_imgs
        mae_r = total_mae_r / total_imgs
        psnr_r = total_psnr_r / total_imgs
        ssim_r = total_ssim_r / total_imgs

        print(f"[Eval over {total_imgs} imgs] "
              f"Fwd MAE:{mae_f:.4f} PSNR:{psnr_f:.2f} SSIM:{ssim_f:.4f} | "
              f"Rev MAE:{mae_r:.4f} PSNR:{psnr_r:.2f} SSIM:{ssim_r:.4f}")
        self.G.train()

        return mae_f, psnr_f, ssim_f, mae_r, psnr_r, ssim_r

    # def evaluate(self):
    #     """Evaluate the generator on the test set, returning (MAE, PSNR, SSIM)."""
    #     # 评估当前 in-memory 的 G
    #     self.G.eval()
    #     total_mae   = 0.0
    #     total_psnr  = 0.0
    #     total_ssim  = 0.0
    #     total_imgs  = 0

    #     def _psnr_per_image(x, y, max_val=1.0, eps=1e-10):
    #         mse = torch.mean((x - y)**2, dim=[1,2,3])
    #         return 10 * torch.log10(max_val**2 / (mse + eps))

    #     ssim_fn = SSIM(data_range=1.0, channel=4, size_average=False).to(self.device)

    #     with torch.no_grad():
    #         for x_o, x_t, c_o, c_t, p0, p1 in tqdm(self.test_loader, desc='Evaluating'):
    #             B = x_o.size(0)
    #             total_imgs += B

    #             x_o = x_o.to(self.device)
    #             x_t = x_t.to(self.device)
    #             c_t = c_t.to(self.device)

    #             x_pred     = self.G(x_o, c_t)
    #             x_pred_den = self.denorm(x_pred)
    #             x_t_den    = self.denorm(x_t)

    #             # MAE
    #             abs_diff = torch.abs(x_pred_den - x_t_den)
    #             per_mae  = abs_diff.view(B, -1).mean(dim=1)
    #             total_mae  += per_mae.sum().item()

    #             # PSNR
    #             per_psnr   = _psnr_per_image(x_pred_den, x_t_den)
    #             total_psnr += per_psnr.sum().item()

    #             # SSIM
    #             per_ssim   = ssim_fn(x_pred_den, x_t_den)
    #             total_ssim += per_ssim.sum().item()

    #     mae  = total_mae  / total_imgs
    #     psnr = total_psnr / total_imgs
    #     ssim = total_ssim / total_imgs

    #     print(f'==== Evaluation results over {total_imgs} samples ====')
    #     print(f'MAE : {mae:.4f}')
    #     print(f'PSNR: {psnr:.2f} dB')
    #     print(f'SSIM: {ssim:.4f}')
    #     self.G.train()
    #     return mae, psnr, ssim

    def test(self):
        self.restore_model(self.test_iters)
        self.G.eval()

        # 正向／反向累加变量
        total_mae_f, total_psnr_f, total_ssim_f = 0.0, 0.0, 0.0
        total_mae_r, total_psnr_r, total_ssim_r = 0.0, 0.0, 0.0
        total_imgs = 0

        def _psnr_per_image(x, y, max_val=1.0, eps=1e-10):
            mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
            return 10 * torch.log10(max_val ** 2 / (mse + eps))

        ssim_fn = SSIM(data_range=1.0, channel=4, size_average=False).to(self.device)
        num_visuals = 5  # 每个 batch 最多可视化的样本数

        def rggb_to_rgb_tensor(img: torch.Tensor) -> torch.Tensor:
            r, gr, gb, b = img[0], img[1], img[2], img[3]
            g = 0.5 * (gr + gb)
            return torch.stack([r, g, b], dim=0)

        with torch.no_grad():
            for i, (x_o, x_t, c_o, c_t, p0, p1) in enumerate(tqdm(self.mydata_loader, desc='Testing')):
                B = x_o.size(0)
                total_imgs += B

                x_o = x_o.to(self.device)
                x_t = x_t.to(self.device)
                c_o = c_o.to(self.device)
                c_t = c_t.to(self.device)

                # —— 正向：原域 → 目标域 —— #
                x_pred = self.G(x_o, c_t)
                x_pred_den = self.denorm(x_pred)
                x_t_den = self.denorm(x_t)

                per_mae_f = torch.abs(x_pred_den - x_t_den).view(B, -1).mean(1)
                per_psnr_f = _psnr_per_image(x_pred_den, x_t_den)
                per_ssim_f = ssim_fn(x_pred_den, x_t_den)
                total_mae_f += per_mae_f.sum().item()
                total_psnr_f += per_psnr_f.sum().item()
                total_ssim_f += per_ssim_f.sum().item()

                # —— 反向：目标域 → 原域 —— #
                x_rev = self.G(x_t, c_o)
                x_rev_den = self.denorm(x_rev)
                x_o_den = self.denorm(x_o)

                per_mae_r = torch.abs(x_rev_den - x_o_den).view(B, -1).mean(1)
                per_psnr_r = _psnr_per_image(x_rev_den, x_o_den)
                per_ssim_r = ssim_fn(x_rev_den, x_o_den)
                total_mae_r += per_mae_r.sum().item()
                total_psnr_r += per_psnr_r.sum().item()
                total_ssim_r += per_ssim_r.sum().item()

                # —— 可视化：分别保存正向与反向三联图 —— #
                V = min(num_visuals, B)
                for k in range(V):
                    # 正向三联：原图、预测→目标、真实目标
                    trip_f = torch.stack([
                        rggb_to_rgb_tensor(x_o_den[k]),
                        rggb_to_rgb_tensor(x_pred_den[k]),
                        rggb_to_rgb_tensor(x_t_den[k])
                    ], dim=0)  # [3,3,H,W]
                    save_image(
                        trip_f,
                        f"{self.result_dir}/batch_{i:03d}_fwd_sample_{k:02d}.png",
                        nrow=3, normalize=False
                    )

                    # 反向三联：目标图、预测→原、真实原
                    trip_r = torch.stack([
                        rggb_to_rgb_tensor(x_t_den[k]),
                        rggb_to_rgb_tensor(x_rev_den[k]),
                        rggb_to_rgb_tensor(x_o_den[k])
                    ], dim=0)
                    save_image(
                        trip_r,
                        f"{self.result_dir}/batch_{i:03d}_rev_sample_{k:02d}.png",
                        nrow=3, normalize=False
                    )

        # —— 计算平均指标 —— #
        mae_f = total_mae_f / total_imgs
        psnr_f = total_psnr_f / total_imgs
        ssim_f = total_ssim_f / total_imgs
        mae_r = total_mae_r / total_imgs
        psnr_r = total_psnr_r / total_imgs
        ssim_r = total_ssim_r / total_imgs

        print(f'==== Test over {total_imgs} samples ====')
        print(f'Forward  (O→T) MAE:{mae_f:.4f}, PSNR:{psnr_f:.2f}, SSIM:{ssim_f:.4f}')
        print(f'Reverse  (T→O) MAE:{mae_r:.4f}, PSNR:{psnr_r:.2f}, SSIM:{ssim_r:.4f}')
    # def test(self):
    #     self.restore_model(self.test_iters)
    #     self.G.eval()

    #     # 新版累加变量：按图像级累加
    #     total_mae   = 0.0
    #     total_psnr  = 0.0
    #     total_ssim  = 0.0
    #     total_imgs  = 0

    #     def _psnr_per_image(x, y, max_val=1.0, eps=1e-10):
    #         # 返回 shape=(B,) 的每张图 PSNR
    #         mse = torch.mean((x - y)**2, dim=[1,2,3])
    #         return 10 * torch.log10(max_val**2 / (mse + eps))

    #     ssim_fn = SSIM(data_range=1.0, channel=4, size_average=False).to(self.device)
    #     num_visuals = 5  # 每个 batch 最多可视化的样本数

    #     # 辅助：把 [4,H,W] 的 RGGB 转成 [3,H,W] 的 RGB
    #     def rggb_to_rgb_tensor(img: torch.Tensor) -> torch.Tensor:
    #         r, gr, gb, b = img[0], img[1], img[2], img[3]
    #         g = 0.5 * (gr + gb)
    #         return torch.stack([r, g, b], dim=0)

    #     with torch.no_grad():
    #         for i, (x_o, x_t, c_o, c_t, p0, p1) in enumerate(tqdm(self.mydata_loader, desc='Testing')):
    #             B = x_o.size(0)
    #             total_imgs += B

    #             x_o = x_o.to(self.device)
    #             x_t = x_t.to(self.device)
    #             c_t = c_t.to(self.device)

    #             x_pred     = self.G(x_o, c_t)
    #             x_pred_den = self.denorm(x_pred)
    #             x_t_den    = self.denorm(x_t)
    #             x_o_den    = self.denorm(x_o)

    #             # —— 指标累加（按图像级） —— #
    #             # MAE：每张图的 mean(|pred - target|)
    #             abs_diff = torch.abs(x_pred_den - x_t_den)
    #             per_mae  = abs_diff.view(B, -1).mean(dim=1)
    #             total_mae   += per_mae.sum().item()

    #             # PSNR：每张图
    #             per_psnr     = _psnr_per_image(x_pred_den, x_t_den)
    #             total_psnr  += per_psnr.sum().item()

    #             # SSIM：每张图
    #             per_ssim     = ssim_fn(x_pred_den, x_t_den)
    #             total_ssim  += per_ssim.sum().item()

    #             V = min(num_visuals, B)
    #             for k in range(V):
    #                 o_img4 = x_o_den[k]    # [4, H, W]
    #                 p_img4 = x_pred_den[k] # [4, H, W]
    #                 t_img4 = x_t_den[k]    # [4, H, W]

    #                 o_img3 = rggb_to_rgb_tensor(o_img4)
    #                 p_img3 = rggb_to_rgb_tensor(p_img4)
    #                 t_img3 = rggb_to_rgb_tensor(t_img4)

    #                 triplet = torch.stack([o_img3, p_img3, t_img3], dim=0)  # [3,3,H,W]
    #                 save_image(
    #                     triplet,
    #                     f"{self.result_dir}/batch_{i:03d}_sample_{k:02d}.png",
    #                     nrow=3,
    #                     normalize=False,
    #                 )

    #     # —— 对所有图像取平均 —— #
    #     mae  = total_mae   / total_imgs
    #     psnr = total_psnr  / total_imgs
    #     ssim = total_ssim  / total_imgs

    #     print(f'==== Test results over {total_imgs} samples ====')
    #     print(f'MAE : {mae:.4f}')
    #     print(f'PSNR: {psnr:.2f} dB')
    #     print(f'SSIM: {ssim:.4f}')
