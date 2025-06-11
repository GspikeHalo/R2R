from model import Generator
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

    def __init__(self, celeba_loader, rafd_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.mydata_loader = celeba_loader
        # self.rafd_loader = rafd_loader

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

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        # g_conv_dim 第一层卷积通道数， g_repeat_num 残差块重复次数
        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
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


    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        data_loader = self.mydata_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

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
            loss['D/loss_real'] = d_loss_real.item()    # D在real图像的得分，应趋于稳定，在固定范围内波动 -1 左右
            loss['D/loss_fake'] = d_loss_fake.item()    # D在fake图像的得分，应尽可能小，在固定范围内波动 +1 左右
            loss['D/loss_cls'] = d_loss_cls.item()      # D的属性分类损失，应尽可能小 < 0.1
            loss['D/loss_gp'] = d_loss_gp.item()        # 接近0

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg)


                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()    # G的对抗损失，下降趋于平稳 约等于0
                loss['G/loss_rec'] = g_loss_rec.item()      # 循环一致，下降
                loss['G/loss_cls'] = g_loss_cls.item()      # 分类损失，下降

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
                wandb.log({
                    "fixed_generated": [
                        wandb.Image(img, caption=str(c_t.tolist()))
                        for img, c_t in zip(x_fake_fixed, c_fixed_list)
                    ]
                }, step=i + 1)

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i + 1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


    def test(self):
        """Evaluate StarGAN on paired data: compute MAE, PSNR, SSIM."""
        # 1. 恢复模型 & 切 eval
        self.restore_model(self.test_iters)
        self.G.eval()

        # 2. 准备累加变量
        total_abs    = 0.0      # 用于 MAE（sum of absolute diff）
        total_pixels = 0        # 用于 MAE（sum of像素数）
        total_psnr   = 0.0      # 累加每张图的 PSNR
        total_ssim   = 0.0      # 累加每张图的 SSIM
        total_imgs   = 0        # 累加图片张数

        # 3. 每张图的 PSNR 计算（返回长度为 B 的张量）
        def _psnr_per_image(x, y, max_val=1.0, eps=1e-10):
            # x,y ∈ [0,1], shape [B,C,H,W]
            mse = torch.mean((x - y)**2, dim=[1,2,3])     # [B]
            psnr = 10 * torch.log10(max_val**2 / (mse + eps))
            return psnr                              # [B]

        # 4. SSIM 也改成逐图输出
        #    size_average=False 会让它返回 [B] 而不是一个标量
        ssim_fn = SSIM(data_range=1.0, channel=4, size_average=False).to(self.device)

        # 5. 遍历测试集
        with torch.no_grad():
            for x_o, x_t, c_o, c_t, p0, p1 in tqdm(self.mydata_loader, desc='Testing'):
                B = x_o.size(0)
                total_imgs += B

                x_o = x_o.to(self.device)    # [B,4,H,W]
                x_t = x_t.to(self.device)    # [B,4,H,W]
                c_t = c_t.to(self.device)    # [B,c_dim]

                # forward
                x_pred = self.G(x_o, c_t)

                # denorm → [0,1]
                x_pred_den = self.denorm(x_pred)
                x_t_den    = self.denorm(x_t)

                # —— MAE 部分 —— 
                diff = torch.abs(x_pred_den - x_t_den)     # [B,4,H,W]
                total_abs    += diff.sum().item()          # 累加所有元素绝对值之和
                total_pixels += diff.numel()               # 累加像素总数

                # —— PSNR 部分 —— 
                psnr_vals  = _psnr_per_image(x_pred_den, x_t_den)  # [B]
                total_psnr += psnr_vals.sum().item()              # 累加每张图的 PSNR

                # —— SSIM 部分 —— 
                ssim_vals  = ssim_fn(x_pred_den, x_t_den)         # [B]
                total_ssim += ssim_vals.sum().item()              # 累加每张图的 SSIM

        # 6. 最终平均
        mae  = total_abs    / total_pixels
        psnr = total_psnr   / total_imgs
        ssim = total_ssim   / total_imgs

        print(f'==== Test results over {total_imgs} samples ====')
        print(f'MAE : {mae:.4f}')
        print(f'PSNR: {psnr:.2f} dB')
        print(f'SSIM: {ssim:.4f}')
