#######################################################################################################################################
################## Code based on https://github.com/pengzhiliang/Conformer/blob/main/conformer.py #####################################
#######################################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, trunc_normal_
from utils import pad_image
import time
import torch.utils.benchmark as benchmark

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, film_params=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, film_params=None):
        x1 = self.norm1(x)
        if film_params is not None:
            gamma, beta = film_params
            x1 = gamma.unsqueeze(1) * x1 + beta.unsqueeze(1)
        x = x + self.drop_path(self.attn(x1))

        # MLP sub-layer
        x2 = self.norm2(x)
        if film_params is not None:
            gamma, beta = film_params
            x2 = gamma.unsqueeze(1) * x2 + beta.unsqueeze(1)
        x = x + self.drop_path(self.mlp(x2))

        return x



class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1,
                               bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True, film_params=None):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if film_params is not None:
            gamma, beta = film_params
            B = x.shape[0]
            x = gamma.view(B, -1, 1, 1) * x + beta.view(B, -1, 1, 1)

        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        x = self.conv_project(x)  # [N, C, H, W]

        x = self.sample_pooling(x).flatten(2).transpose(1, 2)  # [N, patch_nums,embed_dim]
        x = self.ln(x)
        x = self.act(x)

        return x


class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), ):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x.transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))


class Med_ConvBlock(nn.Module):
    """ special case for Convblock with down sampling,
    """

    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=True)

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride,
                                   groups=groups)

        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True,
                                          groups=groups)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)

        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.ModuleList(self.med_block)

        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)

        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x, x_t, film_params=None, film_trans=None):
        x, x2 = self.cnn_block(x, film_params=film_params)

        _, _, H, W = x2.shape

        x_st = self.squeeze_block(x2, x_t)

        x_t = self.trans_block(x_st + x_t, film_params=film_trans)

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t

class FiLMGenerator(nn.Module):
    def __init__(self, c_dim, film_channels, hidden_dim=128):
        super().__init__()
        self.nets = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(c_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 2 * C)
            ) for name, C in film_channels.items()
        })
    def forward(self, z):
        params = {}
        for name, net in self.nets.items():
            gamma_beta = net(z)  # [B, 2*C]
            gamma, beta = gamma_beta.chunk(2, dim=1)
            params[name] = (gamma, beta)
        return params

# class FiLMTransformer(nn.Module):
#     """
#     Cross-Attention–based FiLM parameter generator with:
#       1) conditional + global queries,
#       2) convolutional downsampling for richer KV features,
#       3) robust handling of both 3D tokens and 4D feature maps.
#     """
#     def __init__(
#         self,
#         embed_dim: int,
#         c_dim: int,
#         num_pairs: int = 1,
#         num_heads: int = 4,
#         mlp_ratio: float = 4.0
#     ):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.c_dim = c_dim
#         self.num_pairs = num_pairs
#
#         # 1) Project condition → [B, num_pairs * embed_dim]
#         self.cond_proj = nn.Linear(c_dim, embed_dim * num_pairs)
#         # Learnable global queries bias
#         self.global_q = nn.Parameter(torch.randn(1, num_pairs, embed_dim))
#
#         # 2) Convolutional downsampling for KV: [B, D, H, W] → [B, D, H, W]
#         self.kv_downsample = nn.Sequential(
#             nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(embed_dim),
#             nn.GELU(),
#             nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(embed_dim),
#             nn.GELU(),
#             nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(embed_dim),
#         )
#
#         # 3) Project feature tokens → keys & values
#         self.key_proj = nn.Linear(embed_dim, embed_dim)
#         self.val_proj = nn.Linear(embed_dim, embed_dim)
#
#         # 4) Cross-attention
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             batch_first=True
#         )
#
#         # 5) MLP to map attended vectors → 2*embed_dim
#         hidden_dim = int(embed_dim * mlp_ratio)
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, 2 * embed_dim)
#         )
#
#     def forward(
#         self,
#         feat_embs: torch.Tensor,           # [B, L, D] tokens or [B, D, H, W] feature map
#         target_label: torch.Tensor         # [B, c_dim]
#     ):
#         """
#         Args:
#           feat_embs:    3D tokens [B, L, D] or 4D map [B, D, H, W]
#           target_label: [B, c_dim]
#         Returns:
#           gammas, betas each [B, num_pairs, D]
#         """
#         # ==== Prepare tokens ====
#         if feat_embs.ndim == 4:
#             # conv downsample then flatten
#             x = self.kv_downsample(feat_embs)  # [B, D, H', W']
#             B, D, H2, W2 = x.shape
#             feat_tokens = x.flatten(2).transpose(1, 2)  # [B, N, D]
#         elif feat_embs.ndim == 3:
#             feat_tokens = feat_embs
#             B, _, D = feat_tokens.shape
#         else:
#             raise ValueError(f"Unsupported feat_embs ndim={feat_embs.ndim}")
#
#         # ==== Dim checks ====
#         assert D == self.embed_dim, f"embed_dim mismatch: got {D}"
#         assert target_label.shape[1] == self.c_dim
#
#         # ==== Build queries ====
#         cond_q = self.cond_proj(target_label).view(B, self.num_pairs, D)
#         Q = cond_q + self.global_q  # [B, num_pairs, D]
#
#         # ==== Generate Key/Value ====
#         K = self.key_proj(feat_tokens)  # [B, N, D]
#         V = self.val_proj(feat_tokens)
#
#         # ==== Cross-attention ====
#         attn_out, _ = self.cross_attn(query=Q, key=K, value=V)  # [B, num_pairs, D]
#
#         # ==== MLP and split ====
#         flat = attn_out.reshape(B * self.num_pairs, D)
#         out  = self.mlp(flat).view(B, self.num_pairs, 2 * D)
#         gammas, betas = out.chunk(2, dim=2)  # each [B, num_pairs, D]
#
#         return gammas, betas
class FiLMTransformerMulti(nn.Module):
    def __init__(self, embed_dim, c_dim, num_pairs, num_heads, mlp_ratio):
        super().__init__()
        self.embed_dim = embed_dim
        self.c_dim     = c_dim
        self.num_pairs = num_pairs

        self.cond_proj = nn.Linear(c_dim, embed_dim * num_pairs)
        self.global_q  = nn.Parameter(torch.randn(1, num_pairs, embed_dim))

        self.kv_downsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
        )

        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.val_proj = nn.Linear(embed_dim, embed_dim)  # ← 准备好 val_proj

        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_pairs)
        ])

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 2 * embed_dim),
            )
            for _ in range(num_pairs)
        ])

    def forward(self, feat_embs, target_label):
        B = target_label.size(0)

        x = self.kv_downsample(feat_embs)
        feat_tokens = x.flatten(2).transpose(1, 2)

        D = self.embed_dim
        cond_q = self.cond_proj(target_label).view(B, self.num_pairs, D)
        Q_all  = cond_q + self.global_q

        K = self.key_proj(feat_tokens)
        V = self.val_proj(feat_tokens)

        gammas = []
        betas  = []
        for i in range(self.num_pairs):
            Q_i = Q_all[:, i:i+1, :]
            attn_out, _ = self.cross_attn[i](Q_i, K, V)
            flat       = attn_out.squeeze(1)
            g_b        = self.mlp[i](flat)
            gamma_i, beta_i = g_b.chunk(2, dim=-1)
            gammas.append(gamma_i.unsqueeze(1))
            betas .append(beta_i.unsqueeze(1))

        gammas = torch.cat(gammas, dim=1)
        betas  = torch.cat(betas,  dim=1)
        return gammas, betas

class Conformer(nn.Module):

    def __init__(self, patch_size=16, raw_channels=4, c_dim=2, base_channel=32, channel_ratio=2, num_med_block=0,
                 embed_dim=256, depth=6, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., poly=2, share_model=None,transformer_branch_weight=None,down_scale_times=1,freeze_generator=False,poly_add_const=False,global_mapping=False):

        # Transformer
        super().__init__()
        self.down_scale_times = down_scale_times
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0
        stage_depth = depth // 3

        self.stage1_start = 2
        self.stage1_end = self.stage1_start + stage_depth - 1  # inclusive

        self.stage2_start = self.stage1_end + 1
        self.stage2_end = self.stage2_start + stage_depth - 1

        self.stage3_start = self.stage2_end + 1
        self.stage3_end = self.stage3_start + stage_depth - 1

        self.fin_stage = self.stage3_end + 1

        self.raw_channels = raw_channels
        self.c_dim = c_dim
        # self.in_channels = raw_channels + c_dim
        self.in_channels = raw_channels
        self.patch_size = patch_size

        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # film_channels = {
        #     'stem':       64,
        #     'trans_norm': embed_dim,
        # }

        # Classifier head
        self.trans_norm = nn.LayerNorm(embed_dim)

        self.pooling = nn.AdaptiveAvgPool2d(1)


        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 1 / 4 [56, 56]

        # 1 stage
        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)
        self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )

        # 2~4 stage
        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block
                            )
                            )

        stage_2_channel = int(base_channel * channel_ratio * 2)
        # 5~8 stage
        init_stage = fin_stage  # 5
        fin_stage = fin_stage + depth // 3  # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block
                            )
                            )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        # 9~12 stage
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 3  # 13
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block, last_fusion=last_fusion
                            )
                            )
        self.out1 = nn.Sequential(nn.Linear(in_features=embed_dim,out_features=embed_dim),
                                  nn.Linear(in_features=embed_dim,out_features=56))
        self.fin_stage = fin_stage
        self.apply(self._init_weights)

        self.poly_add_const = poly_add_const
        if share_model is not None:
            for name, share_module in share_model.named_modules():
                if 'out' in name or 'light' in name:
                    continue
                if isinstance(share_module,(nn.Linear,nn.Conv2d)):
                    module = self
                    if '.' in name:
                        name = name.split('.')
                    else:
                        name = [name]
                    for n in name[:-1]:
                        module = getattr(module,n)
                    setattr(module,name[-1],share_module)
                    if freeze_generator:
                        module.requires_grad_(False)

        self.film_xattn = FiLMTransformerMulti(
            embed_dim=embed_dim,
            c_dim=c_dim,
            num_pairs=2,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )

        # self.film_gen = FiLMGenerator(c_dim, film_channels)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)


    def forward(self, x, target_label):
        B,C,H,W = x.shape
        x_poly = x

        x_pad = pad_image(x, 32*self.down_scale_times)
        x_down = F.interpolate(x_pad,mode='bilinear',scale_factor=1.0/self.down_scale_times)

        # stem stage [N, 3, 224, 224] -> [N, 64, 56, 56]
        x = self.conv1(x_down)        # [B, 64, H/2, W/2]
        x = self.bn1(x)               # 归一化
        x = self.act1(x)
        x_base = self.maxpool(x)      # [B, 64, H/4, W/4]

        # 1 stage
        x_feat = self.conv_1(x_base, return_x_2=False)

        # [batch_size, patch_nums,embed_dim]
        feat_map = self.trans_patch_conv(x_base)
        gammas, betas = self.film_xattn(feat_map, target_label)
        gamma_t1, beta_t1 = gammas[:,0], betas[:,0]

        x_t = feat_map.flatten(2).transpose(1, 2)
        # [batch_size, patch_nums+1,embed_dim]
        x_t = self.trans_1(x_t, film_params=(gamma_t1, beta_t1))

        # 2 ~ final
        for i in range(2, self.fin_stage):
            if i == self.stage2_start:
                x_feat, x_t = getattr(self, f"conv_trans_{i}")(x_feat, x_t)
            # Apply stage 3 FiLM at the beginning of stage 3
            elif i == self.stage3_start:
                x_feat, x_t = getattr(self, f"conv_trans_{i}")(x_feat, x_t)
            else:
                x_feat, x_t = getattr(self, f"conv_trans_{i}")(x_feat, x_t)


        x_t = self.trans_norm(x_t)
        gamma_t2, beta_t2 = gammas[:,1], betas[:,1]
        x_t = gamma_t2.unsqueeze(1)*x_t + beta_t2.unsqueeze(1)
        B, patch_num, _ = x_t.size()
        weight_trans = self.out1(x_t)
        image = self.polynomial_transform(x_poly, weight_trans.view(B, patch_num, self.raw_channels, -1)) + x_poly

        return image

    def polynomial_transform(self, x:torch.Tensor, weights:torch.Tensor):
        # weight:[batch_size,patch_nums,out_channel,poly_num_per_channel]
        patch_size = self.patch_size * self.down_scale_times
        B,C,H,W = x.shape
        _,patch_nums,_,poly_num_per_channel = weights.shape
        patch_num_w, patch_num_h = W//patch_size, H//patch_size
        image_patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # B,patch_nums,C,patch_size,patch_size
        image_patches = image_patches.permute(0,2,3,1,4,5).contiguous()
        image_patches = image_patches.view(B,-1,C,patch_size,patch_size)
        B, patch_nums,C, _, _ = image_patches.shape
        image_patches = image_patches.permute(0,2,3,4,1).reshape(B,C,patch_size*patch_size,patch_nums).permute(0, 2, 3, 1)
        # [batch_size,L,patch_nums,poly_num_per_channel]
        poly_image_patches = torch.empty([B, patch_size * patch_size, patch_nums,poly_num_per_channel],device=torch.get_device(x))

        poly_image_patches[:, :, :, :4] = image_patches * image_patches
        poly_image_patches[:, :, :, 4:7] = image_patches[:, :, :, :-1] * image_patches[:, :, :, 1:]
        poly_image_patches[:, :, :, 7:9] = image_patches[:, :, :, :-2] * image_patches[:, :, :, 2:]
        poly_image_patches[:, :, :, 9] = image_patches[:, :, :, -3] * image_patches[:, :, :, 3]
        poly_image_patches[:, :, :, 10:14] = image_patches
        if self.poly_add_const:
            poly_image_patches[:, :, :, 14] = torch.zeros_like(image_patches[:,:,:,0]).to(torch.get_device(image_patches))

        weights = weights.reshape(B,patch_num_h,patch_num_w,self.raw_channels,poly_num_per_channel)
        # [batch_size, raw_channels*poly_num_per_channel,patch_num_h,patch_num_w]
        weights = weights.reshape(B,patch_num_h,patch_num_w,-1).permute(0,3,1,2)
        # [batch_size, raw_channels*poly_num_per_channel,patch_num_h*patch_size=H,patch_num_w*patch_size=W]
        weights = F.interpolate(weights, scale_factor=patch_size, mode='bilinear', align_corners=True)

        weights = weights.view(B, self.raw_channels,poly_num_per_channel,H,W)
        weights = weights.view(B, self.raw_channels, poly_num_per_channel, patch_num_h, patch_size, patch_num_w, patch_size)
        weights = weights.permute(0,1,2,3,5,4,6).contiguous().view(B,self.raw_channels,poly_num_per_channel,patch_num_h*patch_num_w,patch_size*patch_size)

        # [batch_size,L=patch_size*patch_size,patch_nums=patch_num_h*patch_num_w,poly_num_per_channel,out_channels=raw_channels]
        weights = weights.permute(0,4,3,2,1).contiguous()
        # [batch_size,L,patch_nums,out_channels]
        poly_image_patches = torch.matmul(poly_image_patches.unsqueeze(-2), weights).squeeze(-2)
        image_patches = poly_image_patches.view(B, patch_size, patch_size, patch_nums, self.raw_channels)
        image_patches = image_patches.view(B,patch_size,patch_size,H//patch_size,W//patch_size,C)
        image = image_patches.permute(0,5,3,1,4,2).contiguous().view(B,C,H,W)
        return image
