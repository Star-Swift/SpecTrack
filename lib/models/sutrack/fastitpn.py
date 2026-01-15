# --------------------------------------------------------
# Fast-iTPN: Integrally Pre-Trained Transformer Pyramid Network with Token Migration
# Github source: https://github.com/sunsmarterjie/iTPN/tree/main/fast_itpn
# Copyright (c) 2023 University of Chinese Academy of Sciences
# Licensed under The MIT License [see LICENSE for details]
# By Yunjie Tian
# Based on EVA02, timm and deit code bases
# https://github.com/baaivision/EVA/tree/master/EVA-02
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# --------------------------------------------------------'
from functools import partial
import warnings
import math
import torch
import torch.nn as nn
from timm.models.registry import register_model
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import to_2tuple, drop_path, trunc_normal_

from torch import Tensor, Size
from typing import Union, List

from einops import rearrange
from einops.layers.torch import Rearrange
from torch.distributions.normal import Normal
import numbers


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


_shape_t = Union[int, List[int], Size]


##########################################################################
## MoCE (Mixture of Complexity Experts) 相关模块
##########################################################################

class MySequential(nn.Sequential):
    """自定义Sequential：支持双输入（特征+共享特征）的序列传播"""
    def forward(self, x1, x2):
        for layer in self:
            x1 = layer(x1, x2)
        return x1


class SparseDispatcher(object):
    """稀疏调度器：实现特征在专家间的分发与合并"""
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(-1).unsqueeze(-1))
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), expert_out[-1].size(3),
                            requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class MoCELayerNorm(nn.Module):
    """MoCE专用LayerNorm，支持4D输入"""
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(MoCELayerNorm, self).__init__()
        self.dim = dim
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class SharedAttention(nn.Module):
    """共享注意力：用于生成共享特征"""
    def __init__(self, dim, num_heads=8, bias=True):
        super(SharedAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class FFTAttention(nn.Module):
    """频域注意力：在频域计算注意力，高效捕捉全局关联"""
    def __init__(self, dim: int, kernel_size: int = 3, patch_size: int = 4, **kwargs):
        super(FFTAttention, self).__init__()
        self.patch_size = patch_size

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=dim * 2)
        self.norm = MoCELayerNorm(dim, "WithBias")
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)

    def pad_and_rearrange(self, x):
        b, c, h, w = x.shape
        pad_h = (self.patch_size - (h % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (w % self.patch_size)) % self.patch_size
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        x = rearrange(x, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=self.patch_size, p2=self.patch_size)
        return x

    def rearrange_to_original(self, x, x_shape):
        h, w = x_shape
        x = rearrange(x, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size)
        x = x[:, :, :h, :w]
        return x

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)

        q = self.pad_and_rearrange(q)
        k = self.pad_and_rearrange(k)

        q_fft = torch.fft.rfft2(q.float())
        k_fft = torch.fft.rfft2(k.float())
        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))

        out = self.rearrange_to_original(out, (h, w))
        out = self.norm(out)
        out = out * v
        out = self.proj_out(out)
        return out


class ModExpert(nn.Module):
    """调制专家：单个专家单元，含投影+核心功能+调制+残差"""
    def __init__(self, dim: int, rank: int, func: nn.Module, depth: int, patch_size: int, kernel_size: int):
        super(ModExpert, self).__init__()
        self.depth = depth
        self.proj = nn.ModuleList([
            nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(dim, rank, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(rank, dim, kernel_size=1, padding=0, bias=False)
        ])
        self.body = func(rank, kernel_size=kernel_size, patch_size=patch_size)

    def process(self, x, shared):
        shortcut = x
        x = self.proj[0](x)
        x = self.body(x) * F.silu(self.proj[1](shared))
        x = self.proj[2](x)
        return x + shortcut

    def feat_extract(self, feats, shared):
        for _ in range(self.depth):
            feats = self.process(feats, shared)
        return feats

    def forward(self, x, shared):
        if x.size(0) == 0:
            return x
        return self.feat_extract(x, shared)


class HighPassConv2d(nn.Module):
    """高通滤波器：提取高频特征"""
    def __init__(self, c, freeze=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, bias=False, groups=c)
        kernel = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], dtype=torch.float32)
        self.conv.weight.data = kernel.repeat(c, 1, 1, 1)
        if freeze:
            self.conv.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)


class FrequencyEmbedding(nn.Module):
    """频率嵌入：提取高频特征作为路由依据"""
    def __init__(self, dim):
        super(FrequencyEmbedding, self).__init__()
        self.high_conv = nn.Sequential(HighPassConv2d(dim, freeze=True), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x):
        x = self.high_conv(x)
        x = x.mean(dim=(-2, -1))
        x = self.mlp(x)
        return x


class RoutingFunction(nn.Module):
    """路由函数：生成门控分数，选择激活的专家"""
    def __init__(self, dim, freq_dim, num_experts, k, complexity, use_complexity_bias=True, complexity_scale="max"):
        super(RoutingFunction, self).__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c 1 1 -> b c'),
            nn.Linear(dim, num_experts, bias=False)
        )
        self.freq_gate = nn.Linear(freq_dim, num_experts, bias=False)
        if complexity_scale == "min":
            complexity = complexity / complexity.min()
        elif complexity_scale == "max":
            complexity = complexity / complexity.max()
        self.register_buffer('complexity', complexity)

        self.k = k
        self.num_experts = num_experts
        self.noise_std = (1.0 / num_experts) * 1.0
        self.use_complexity_bias = use_complexity_bias

    def forward(self, x, freq_emb):
        logits = self.gate(x) + self.freq_gate(freq_emb)
        aux_loss = 0
        if self.training:
            aux_loss = self.importance_loss(logits.softmax(dim=-1))

        noise = torch.randn_like(logits) * self.noise_std
        noisy_logits = logits + noise
        gating_scores = noisy_logits.softmax(dim=-1)
        top_k_values, top_k_indices = torch.topk(gating_scores, self.k, dim=-1)

        if self.training:
            loss_load = self.load_loss(logits, noisy_logits, self.noise_std)
            aux_loss = 0.5 * aux_loss + 0.5 * loss_load

        gates = torch.zeros_like(logits).scatter_(1, top_k_indices, top_k_values)
        return gates, top_k_indices, top_k_values, aux_loss

    def importance_loss(self, gating_scores):
        importance = gating_scores.sum(dim=0)
        if self.use_complexity_bias:
            importance = importance * self.complexity
        imp_mean = importance.mean()
        imp_std = importance.std()
        return (imp_std / (imp_mean + 1e-8)) ** 2

    def load_loss(self, logits, logits_noisy, noise_std):
        thresholds = torch.topk(logits_noisy, self.k, dim=-1).indices[:, -1]
        threshold_per_item = torch.sum(F.one_hot(thresholds, self.num_experts) * logits_noisy, dim=-1)
        noise_required_to_win = threshold_per_item.unsqueeze(-1) - logits
        noise_required_to_win /= noise_std
        normal_dist = Normal(0, 1)
        p = 1. - normal_dist.cdf(noise_required_to_win)
        p_mean = p.mean(dim=0)
        p_mean_std = p_mean.std()
        p_mean_mean = p_mean.mean()
        return (p_mean_std / (p_mean_mean + 1e-8)) ** 2


class MoCEAdapter(nn.Module):
    """
    复杂度混合专家适配层（MoCE Adapter）
    核心创新：
        1. 多复杂度专家：专家间深度、补丁尺寸、核大小不同
        2. 动态路由：结合空间+频域特征生成门控
        3. 稀疏激活：仅激活top-k专家
    """
    def __init__(self, dim: int, rank: int = 64, num_experts: int = 4, top_k: int = 2,
                 expert_layer: nn.Module = FFTAttention, stage_depth: int = 1,
                 depth_type: str = "constant", freq_dim: int = None,
                 with_complexity: bool = True, complexity_scale: str = "max"):
        super().__init__()

        self.top_k = top_k
        self.num_experts = num_experts
        self.aux_loss = None
        freq_dim = freq_dim or dim

        patch_sizes = [2 ** (i + 2) for i in range(num_experts)]
        kernel_sizes = [3 + (2 * i) for i in range(num_experts)]

        if depth_type == "constant":
            depths = [stage_depth for _ in range(num_experts)]
        elif depth_type == "lin":
            depths = [stage_depth + i for i in range(num_experts)]
        else:
            depths = [stage_depth for _ in range(num_experts)]

        ranks = [rank for _ in range(num_experts)]

        self.freq_embed = FrequencyEmbedding(dim)
        self.shared_attn = SharedAttention(dim, num_heads=max(1, dim // 64))

        self.experts = nn.ModuleList([
            MySequential(ModExpert(dim, rank=r, func=expert_layer, depth=d, patch_size=p, kernel_size=k))
            for d, r, p, k in zip(depths, ranks, patch_sizes, kernel_sizes)
        ])

        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False)
        expert_complexity = torch.tensor([sum(p.numel() for p in expert.parameters()) for expert in self.experts])
        self.routing = RoutingFunction(dim, freq_dim, num_experts=num_experts, k=top_k,
                                        complexity=expert_complexity, use_complexity_bias=with_complexity,
                                        complexity_scale=complexity_scale)

    def forward(self, x, H, W):
        """
        输入: x [B, L, C] (序列格式), H, W 为空间维度
        输出: x [B, L, C]
        """
        B, L, C = x.shape
        
        
        x_4d = x.transpose(1, 2).reshape(B, C, H, W)

        freq_emb = self.freq_embed(x_4d)
        shared = self.shared_attn(x_4d)

        gates, top_k_indices, top_k_values, aux_loss = self.routing(x_4d, freq_emb)
        self.aux_loss = aux_loss

        if self.training:
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(x_4d)
            expert_shared = dispatcher.dispatch(shared)
            expert_outputs = [self.experts[i](expert_inputs[i], expert_shared[i]) for i in range(self.num_experts)]
            out = dispatcher.combine(expert_outputs, multiply_by_gates=True)
        else:
            selected_experts = [self.experts[i] for i in top_k_indices.squeeze(0)]
            expert_outputs = torch.stack([exp(x_4d, shared) for exp in selected_experts], dim=1)
            gates_selected = gates.gather(1, top_k_indices)
            weighted = gates_selected.unsqueeze(2).unsqueeze(3).unsqueeze(4) * expert_outputs
            out = weighted.sum(dim=1)

        out = self.proj_out(out)
        out = out.flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, L, C]
        return out


##########################################################################
## 原有代码继续
##########################################################################

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 norm_layer=nn.LayerNorm, subln=False
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.ffn_ln(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 norm_layer=nn.LayerNorm, subln=False
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()

        self.ffn_ln = norm_layer(hidden_features) if subln else None

        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.ffn_ln is not None:
            x = x.permute(0, 2, 3, 1)
            x = self.ffn_ln(x)
            x = x.permute(0, 3, 1, 2)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.,
                 norm_layer=nn.LayerNorm, subln=False
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x


class ConvSwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.,
                 norm_layer=nn.LayerNorm, subln=False
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Conv2d(in_features, hidden_features, 1)
        self.w2 = nn.Conv2d(in_features, hidden_features, 1)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Conv2d(hidden_features, out_features, 1)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x1 = self.w1(x).flatten(2).transpose(1, 2)
        x2 = self.w2(x).flatten(2).transpose(1, 2)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden).transpose(1, 2).view(B, C, H, W)
        x = self.w3(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=None,
            attn_head_dim=None, use_decoupled_rel_pos_bias=False, deepnorm=False, subln=False
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.deepnorm = deepnorm
        self.subln = subln
        if self.deepnorm or self.subln:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=False)
        else:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.rel_pos_bias = None
        self.qk_float = True

        self.window_size = None
        self.relative_position_bias_table = None

        if window_size:
            if use_decoupled_rel_pos_bias:
                self.rel_pos_bias = DecoupledRelativePositionBias(window_size=window_size, num_heads=num_heads)
            else:
                self.window_size = window_size
                self.num_relative_distance = (2 * window_size[0] - 1) * (
                        2 * window_size[1] - 1) + 3  # (2*14-1) * (2*14-1) + 3
                self.relative_position_bias_table = nn.Parameter(
                    torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
                # cls to token & token 2 cls & cls to cls

                # get pair-wise relative position index for each token inside the window
                coords_h = torch.arange(window_size[0])
                coords_w = torch.arange(window_size[1])
                coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
                coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
                relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
                relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
                relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
                relative_coords[:, :, 1] += window_size[1] - 1
                relative_coords[:, :, 0] *= 2 * window_size[1] - 1
                relative_position_index = \
                    torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
                relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
                relative_position_index[0, 0:] = self.num_relative_distance - 3
                relative_position_index[0:, 0] = self.num_relative_distance - 2
                relative_position_index[0, 0] = self.num_relative_distance - 1

                self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        B, N, C = x.shape

        if self.deepnorm or self.subln:
            q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
            k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
            v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)

            q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # B, num_heads, N, C
            k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
            v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        else:
            qkv_bias = None
            if self.q_bias is not None:
                qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # 3, B, num_heads, N, C
            q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        if self.qk_float:
            attn = (q.float() @ k.float().transpose(-2, -1))
        else:
            attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0).type_as(attn)

        if self.rel_pos_bias is not None:
            attn = attn + self.rel_pos_bias().type_as(attn)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias.type_as(attn)
        if attn_mask is not None:
            attn_mask = attn_mask.bool()
            attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, norm_layer=nn.LayerNorm, window_size=None, attn_head_dim=None,
                 use_decoupled_rel_pos_bias=False,
                 depth=None,
                 postnorm=False,
                 deepnorm=False,
                 subln=False,
                 swiglu=False,
                 naiveswiglu=False,
                 use_moce=False,
                 moce_rank=64,
                 moce_num_experts=4,
                 moce_top_k=2,
                 num_patches_search=196,  # 新增：search patches 数量
                 ):
        super().__init__()

        with_attn = num_heads > 0

        self.norm1 = norm_layer(dim) if with_attn else None
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size,
            use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias, attn_head_dim=attn_head_dim,
            deepnorm=deepnorm,
            subln=subln
        ) if with_attn else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        if naiveswiglu:
            self.mlp = SwiGLU(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                subln=subln,
                norm_layer=norm_layer,
            )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                subln=subln,
                norm_layer=norm_layer
            )

        # MoCE适配层
        self.use_moce = use_moce and with_attn
        self.num_patches_search = num_patches_search
        if self.use_moce:
            self.moce = MoCEAdapter(dim=dim, rank=moce_rank, num_experts=moce_num_experts, top_k=moce_top_k)
            self.moce_scale = nn.Parameter(torch.zeros(1))  # 可学习缩放因子，初始为0

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),
                                        requires_grad=True) if self.attn is not None else None
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.deepnorm = deepnorm
        if self.deepnorm:
            self.alpha = math.pow(2.0 * depth, 0.25)

        self.postnorm = postnorm

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        if self.gamma_2 is None:
            if self.postnorm:
                if self.attn is not None:
                    x = x + self.drop_path(
                        self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            elif self.deepnorm:
                if self.attn is not None:
                    residual = x
                    x = self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
                    x = self.drop_path(x)
                    x = residual * self.alpha + x
                    x = self.norm1(x)

                residual = x
                x = self.mlp(x)
                x = self.drop_path(x)
                x = residual * self.alpha + x
                x = self.norm2(x)
            else:
                if self.attn is not None:
                    x = x + self.drop_path(
                        self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                # MoCE适配层 - 只对 search tokens 应用
                if self.use_moce:
                    B, N, C = x.shape
                    # 计算 search 区域的空间维度
                    H_search = W_search = int(math.sqrt(self.num_patches_search))
                    # 检查是否有 cls token (如果 N > num_patches_search + template_patches)
                    has_cls = (N > self.num_patches_search) and ((N - 1) % self.num_patches_search != self.num_patches_search)
                    
                    if has_cls:
                        cls_token = x[:, :1, :]
                        x_rest = x[:, 1:, :]
                        x_search = x_rest[:, :self.num_patches_search, :]
                        x_other = x_rest[:, self.num_patches_search:, :]
                    else:
                        cls_token = None
                        x_search = x[:, :self.num_patches_search, :]
                        x_other = x[:, self.num_patches_search:, :]
                    
                    # 只对 search tokens 应用 MoCE
                    moce_out = self.moce(x_search, H_search, W_search)
                    x_search = x_search + self.moce_scale * moce_out
                    
                    # 重新组合
                    if has_cls:
                        x = torch.cat([cls_token, x_search, x_other], dim=1)
                    else:
                        x = torch.cat([x_search, x_other], dim=1)
                
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.postnorm:
                if self.attn is not None:
                    x = x + self.drop_path(
                        self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
            else:
                if self.attn is not None:
                    x = x + self.drop_path(
                        self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                # MoCE适配层 - 只对 search tokens 应用
                if self.use_moce:
                    B, N, C = x.shape
                    H_search = W_search = int(math.sqrt(self.num_patches_search))
                    has_cls = (N > self.num_patches_search) and ((N - 1) % self.num_patches_search != self.num_patches_search)
                    
                    if has_cls:
                        cls_token = x[:, :1, :]
                        x_rest = x[:, 1:, :]
                        x_search = x_rest[:, :self.num_patches_search, :]
                        x_other = x_rest[:, self.num_patches_search:, :]
                    else:
                        cls_token = None
                        x_search = x[:, :self.num_patches_search, :]
                        x_other = x[:, self.num_patches_search:, :]
                    
                    moce_out = self.moce(x_search, H_search, W_search)
                    x_search = x_search + self.moce_scale * moce_out
                    
                    if has_cls:
                        x = torch.cat([cls_token, x_search, x_other], dim=1)
                    else:
                        x = torch.cat([x_search, x_other], dim=1)
                
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class ConvMlpBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop_path=0., init_values=None, norm_layer=nn.LayerNorm,
                 depth=None,
                 postnorm=False,
                 deepnorm=False,
                 subln=False,
                 swiglu=False,
                 naiveswiglu=False,
                 ):
        super().__init__()

        self.attn = None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        if swiglu:
            self.mlp = xops.SwiGLU(
                in_features=dim,
                hidden_features=mlp_hidden_dim
            )  # hidden_features: 2/3
        elif naiveswiglu:
            self.mlp = ConvSwiGLU(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                subln=subln,
                norm_layer=norm_layer,
            )
        else:
            self.mlp = ConvMlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                subln=subln,
                norm_layer=norm_layer
            )

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),
                                        requires_grad=True) if self.attn is not None else None
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.deepnorm = deepnorm
        if self.deepnorm:
            self.alpha = math.pow(2.0 * depth, 0.25)

        self.postnorm = postnorm

    def forward(self, x):
        if self.gamma_2 is None:
            if self.postnorm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            elif self.deepnorm:
                residual = x
                x = self.mlp(x)
                x = self.drop_path(x)
                x = residual * self.alpha + x
                x = self.norm2(x)
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
        else:
            if self.postnorm:
                x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
            else:
                m = self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
                x = x + self.drop_path(self.gamma_2 * m)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, inner_patches=4, in_chans=16, embed_dim=128, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.inner_patches = inner_patches
        self.patches_resolution = self.patch_shape = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        conv_size = [size // inner_patches for size in patch_size]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=conv_size, stride=conv_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        patches_resolution = (H // self.patch_size[0], W // self.patch_size[1])
        num_patches = patches_resolution[0] * patches_resolution[1]
        x = self.proj(x).view(
            B, -1,
            patches_resolution[0], self.inner_patches,
            patches_resolution[1], self.inner_patches,
        ).permute(0, 2, 4, 3, 5, 1).reshape(B, num_patches, self.inner_patches, self.inner_patches, -1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvPatchEmbed(nn.Module):
    def __init__(self, search_size=224,template_size=112, patch_size=16, inner_patches=4, in_chans=16, embed_dim=128, norm_layer=None,
                 stop_grad_conv1=False):
        super().__init__()
        search_size = to_2tuple(search_size)
        template_size = to_2tuple(template_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution_search = [search_size[0] // patch_size[0], search_size[1] // patch_size[1]]
        patches_resolution_template = [template_size[0] // patch_size[0], template_size[1] // patch_size[1]]
        self.search_size = search_size
        self.template_size = template_size
        self.patch_size = patch_size
        self.stop_grad_conv1 = stop_grad_conv1
        self.inner_patches = inner_patches
        self.patches_resolution_search = self.patch_shape_search = patches_resolution_search
        self.num_patches_search = patches_resolution_search[0] * patches_resolution_search[1]
        self.patches_resolution_template = self.patch_shape_template = patches_resolution_template
        self.num_patches_template = patches_resolution_template[0] * patches_resolution_template[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        conv_size = [size // inner_patches for size in patch_size]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=conv_size, stride=conv_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x, bool_masked_pos=None, mask_token=None):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.stop_grad_conv1:
            x = x.detach() * 0.9 + x * 0.1

        if bool_masked_pos is not None:
            x = torch.nn.functional.unfold(x, kernel_size=4, stride=4, padding=0).transpose(1, 2)

            seq_len = x.shape[1]
            mask_token = mask_token.expand(B, seq_len, -1)
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

            x = torch.nn.functional.fold(x.transpose(1, 2), output_size=(H // 4, W // 4), kernel_size=4, padding=0,
                                         stride=4)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerge(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)
        self.mlp = None

    def forward(self, x):
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class ConvPatchMerge(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim)
        self.reduction = nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2, padding=0)
        self.mlp = None

    def forward(self, x):
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.reduction(x)
        return x


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


def _mask_1d_rel_pos_index(seq_len):
    index = torch.arange(seq_len)
    return index.view(1, seq_len) - index.view(seq_len, 1) + seq_len - 1


def _add_cls_to_index_matrix(index, num_tokens, offset):
    index = index.contiguous().view(num_tokens, num_tokens)
    new_index = torch.zeros(size=(num_tokens + 1, num_tokens + 1), dtype=index.dtype)
    new_index[1:, 1:] = index
    new_index[0, 0:] = offset
    new_index[0:, 0] = offset + 1
    new_index[0, 0] = offset + 2
    return new_index


class DecoupledRelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] + 2, 2 * window_size[1] + 2)

        num_tokens = window_size[0] * window_size[1]

        self.relative_position_bias_for_high = nn.Parameter(torch.zeros(self.num_relative_distance[0], num_heads))
        self.relative_position_bias_for_width = nn.Parameter(torch.zeros(self.num_relative_distance[1], num_heads))
        # cls to token & token 2 cls & cls to cls

        h_index = _mask_1d_rel_pos_index(window_size[0]).view(
            window_size[0], 1, window_size[0], 1).expand(-1, window_size[1], -1, window_size[1])
        h_index = _add_cls_to_index_matrix(h_index, num_tokens, 2 * window_size[0] - 1)
        self.register_buffer("relative_position_high_index", h_index)

        w_index = _mask_1d_rel_pos_index(window_size[1]).view(
            1, window_size[1], 1, window_size[1]).expand(window_size[0], -1, window_size[0], -1)
        w_index = _add_cls_to_index_matrix(w_index, num_tokens, 2 * window_size[1] - 1)

        self.register_buffer("relative_position_width_index", w_index)

    def forward(self):
        relative_position_bias = \
            F.embedding(input=self.relative_position_high_index, weight=self.relative_position_bias_for_high) + \
            F.embedding(input=self.relative_position_width_index, weight=self.relative_position_bias_for_width)
        return relative_position_bias.permute(2, 0, 1).contiguous()


class Fast_iTPN(nn.Module):
    def __init__(self, search_size=224,template_size=112, patch_size=16, in_chans=16, embed_dim=512, depth_stage1=3, depth_stage2=3, depth=24,
                 num_heads=8, bridge_mlp_ratio=3., mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.0, init_values=None, attn_head_dim=None, norm_layer=nn.LayerNorm,
                 patch_norm=False, num_classes=1000, use_mean_pooling=False,
                 init_scale=0.01,
                 cls_token=False,
                 grad_ckpt=False,
                 stop_grad_conv1=False,
                 use_abs_pos_emb=True,
                 use_rel_pos_bias=False,
                 use_shared_rel_pos_bias=False,
                 use_shared_decoupled_rel_pos_bias=False,
                 convmlp=False,
                 postnorm=False,
                 deepnorm=False,
                 subln=False,
                 swiglu=False,
                 naiveswiglu=False,
                 token_type_indicate=False,
                 use_moce=False,
                 moce_rank=64,
                 moce_num_experts=4,
                 moce_top_k=2,
                 moce_start_layer=0,
                 **kwargs):
        super().__init__()
        self.search_size = search_size
        self.template_size = template_size
        self.token_type_indicate = token_type_indicate
        self.mlp_ratio = mlp_ratio
        self.grad_ckpt = grad_ckpt
        self.num_main_blocks = depth
        self.depth_stage1 = depth_stage1
        self.depth_stage2 = depth_stage2
        self.depth = depth
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim
        self.convmlp = convmlp
        self.stop_grad_conv1 = stop_grad_conv1
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_shared_rel_pos_bias = use_shared_rel_pos_bias
        self.use_shared_decoupled_rel_pos_bias = use_shared_decoupled_rel_pos_bias
        self.use_decoupled_rel_pos_bias = False

        # MoCE配置
        self.use_moce = use_moce
        self.moce_rank = moce_rank
        self.moce_num_experts = moce_num_experts
        self.moce_top_k = moce_top_k
        self.moce_start_layer = moce_start_layer

        mlvl_dims = {'4': embed_dim // 4, '8': embed_dim // 2, '16': embed_dim}
        # split image into non-overlapping patches
        if convmlp:
            self.patch_embed = ConvPatchEmbed(
                search_size=search_size,template_size=template_size, patch_size=patch_size, in_chans=in_chans, embed_dim=mlvl_dims['4'],
                stop_grad_conv1=stop_grad_conv1,
                norm_layer=norm_layer if patch_norm else None)
        else:
            self.patch_embed = PatchEmbed(
                img_size=search_size, patch_size=patch_size, in_chans=in_chans, embed_dim=mlvl_dims['4'],
                norm_layer=norm_layer if patch_norm else None)
        self.num_patches_search = self.patch_embed.num_patches_search
        self.num_patches_template = self.patch_embed.num_patches_template
        if cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches_search+self.num_patches_template, embed_dim))
        else:
            self.pos_embed = None
        # indicate for tracking
        if self.token_type_indicate:
            self.template_background_token = nn.Parameter(torch.zeros(embed_dim))
            self.template_foreground_token = nn.Parameter(torch.zeros(embed_dim))
            self.search_token = nn.Parameter(torch.zeros(embed_dim))


        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape_search, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        if use_shared_decoupled_rel_pos_bias:
            assert self.rel_pos_bias is None
            self.rel_pos_bias = DecoupledRelativePositionBias(window_size=self.patch_embed.patch_shape_search,
                                                              num_heads=num_heads)

        self.subln = subln
        self.swiglu = swiglu
        self.naiveswiglu = naiveswiglu

        self.build_blocks(
            depths=[depth_stage1, depth_stage2, depth],
            dims=mlvl_dims,
            num_heads=num_heads,
            bridge_mlp_ratio=bridge_mlp_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            window_size=self.patch_embed.patch_shape_search if use_rel_pos_bias else None,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            attn_head_dim=attn_head_dim,
            postnorm=postnorm,
            deepnorm=deepnorm,
            subln=subln,
            swiglu=swiglu,
            naiveswiglu=naiveswiglu,
            convmlp=convmlp,
        )

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)

        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)

        self.apply(self._init_weights)

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def build_blocks(self,
                     depths=[3, 3, 24],
                     dims={'4': 128 // 4, '8': 256, '16': 512},
                     num_heads=8,
                     bridge_mlp_ratio=3.,
                     mlp_ratio=4.0,
                     qkv_bias=True,
                     qk_scale=None,
                     window_size=None,
                     drop=0.,
                     attn_drop=0.,
                     drop_path_rate=0.,
                     norm_layer=nn.LayerNorm,
                     init_values=0.,
                     attn_head_dim=None,
                     postnorm=False,
                     deepnorm=False,
                     subln=False,
                     swiglu=False,
                     naiveswiglu=False,
                     convmlp=False,
                     ):
        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, depths[0] + depths[1] + depths[2]))

        self.blocks = nn.ModuleList()

        if convmlp:
            self.blocks.extend([
                ConvMlpBlock(
                    dim=dims['4'],
                    mlp_ratio=bridge_mlp_ratio,
                    drop_path=next(dpr),
                    norm_layer=norm_layer,
                    init_values=0.,
                    depth=depths[-1],
                    postnorm=postnorm,
                    deepnorm=deepnorm,
                    subln=subln,
                    swiglu=False,
                    naiveswiglu=False,
                ) for _ in range(depths[0])
            ])
            self.blocks.append(ConvPatchMerge(dims['4'], norm_layer))
            self.blocks.extend([
                ConvMlpBlock(
                    dim=dims['8'],
                    mlp_ratio=bridge_mlp_ratio,
                    drop_path=next(dpr),
                    norm_layer=norm_layer,
                    init_values=0.,
                    depth=depths[-1],
                    postnorm=postnorm,
                    deepnorm=deepnorm,
                    subln=subln,
                    swiglu=False,
                    naiveswiglu=False,
                ) for _ in range(depths[1])
            ])
            self.blocks.append(ConvPatchMerge(dims['8'], norm_layer))
        else:
            self.blocks.extend([
                Block(
                    dim=dims['4'],
                    num_heads=0,
                    mlp_ratio=bridge_mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=next(dpr),
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=window_size,
                    depth=depths[-1],
                    postnorm=postnorm,
                    deepnorm=deepnorm,
                    subln=subln,
                    swiglu=swiglu,
                    naiveswiglu=naiveswiglu,
                ) for _ in range(depths[0])
            ])
            self.blocks.append(PatchMerge(dims['4'], norm_layer))
            self.blocks.extend([
                Block(
                    dim=dims['8'],
                    num_heads=0,
                    mlp_ratio=bridge_mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=next(dpr),
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=window_size,
                    depth=depths[-1],
                    postnorm=postnorm,
                    deepnorm=deepnorm,
                    subln=subln,
                    swiglu=swiglu,
                    naiveswiglu=naiveswiglu,
                ) for _ in range(depths[1])
            ])
            self.blocks.append(PatchMerge(dims['8'], norm_layer))

        ######### stage 3 with MoCE ########
        for layer_idx in range(depths[2]):
            use_moce_this_layer = self.use_moce and (layer_idx >= self.moce_start_layer)
            self.blocks.append(
                Block(
                    dim=dims['16'],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=next(dpr),
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=window_size,
                    attn_head_dim=attn_head_dim,
                    depth=depths[-1],
                    postnorm=postnorm,
                    deepnorm=deepnorm,
                    subln=subln,
                    swiglu=swiglu,
                    naiveswiglu=naiveswiglu,
                    use_moce=use_moce_this_layer,
                    moce_rank=self.moce_rank,
                    moce_num_experts=self.moce_num_experts,
                    moce_top_k=self.moce_top_k,
                    num_patches_search=self.num_patches_search,  # 传递 search patches 数量
                )
            )

    def get_moce_aux_loss(self):
        """获取所有MoCE层的辅助损失"""
        aux_loss = 0
        count = 0
        for blk in self.blocks:
            if hasattr(blk, 'use_moce') and blk.use_moce and hasattr(blk, 'moce'):
                if blk.moce.aux_loss is not None:
                    aux_loss = aux_loss + blk.moce.aux_loss
                    count += 1
        return aux_loss / max(count, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cls_token is not None:
            return {'pos_embed', 'cls_token'}
        return {'pos_embed'}

    def get_classifer(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def create_mask(self, image, image_anno):
        height = image.size(2)
        width = image.size(3)

        # Extract bounding box coordinates
        x0 = (image_anno[:, 0] * width).unsqueeze(1)
        y0 = (image_anno[:, 1] * height).unsqueeze(1)
        w = (image_anno[:, 2] * width).unsqueeze(1)
        h = (image_anno[:, 3] * height).unsqueeze(1)

        # Generate pixel indices
        x_indices = torch.arange(width, device=image.device)
        y_indices = torch.arange(height, device=image.device)

        # Create masks for x and y coordinates within the bounding boxes
        x_mask = ((x_indices >= x0) & (x_indices < x0 + w)).float()
        y_mask = ((y_indices >= y0) & (y_indices < y0 + h)).float()

        # Combine x and y masks to get final mask
        mask = x_mask.unsqueeze(1) * y_mask.unsqueeze(2) # (b,h,w)

        return mask

    def prepare_tokens_with_masks(self, template_list, search_list, template_anno_list, text_src, task_index):
        B = search_list[0].size(0)

        num_template = len(template_list)
        num_search = len(search_list)

        z = torch.stack(template_list, dim=1)  # (b,n,c,h,w)
        z = z.view(-1, *z.size()[2:])  # (bn,c,h,w)
        x = torch.stack(search_list, dim=1)  # (b,n,c,h,w)
        x = x.view(-1, *x.size()[2:])  # (bn,c,h,w)
        z_anno = torch.stack(template_anno_list, dim=1)  # (b,n,4)
        z_anno = z_anno.view(-1, *z_anno.size()[2:])  # (bn,4)
        if self.token_type_indicate:
            # generate a foreground mask
            z_indicate_mask = self.create_mask(z, z_anno)
            z_indicate_mask = z_indicate_mask.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size) # to match the patch embedding
            z_indicate_mask = z_indicate_mask.mean(dim=(3,4)).flatten(1) # elements are in [0,1], float, near to 1 indicates near to foreground, near to 0 indicates near to background

        if self.token_type_indicate:
            # generate the indicate_embeddings for z
            template_background_token = self.template_background_token.unsqueeze(0).unsqueeze(1).expand(z_indicate_mask.size(0), z_indicate_mask.size(1), self.embed_dim)
            template_foreground_token = self.template_foreground_token.unsqueeze(0).unsqueeze(1).expand(z_indicate_mask.size(0), z_indicate_mask.size(1), self.embed_dim)
            weighted_foreground = template_foreground_token * z_indicate_mask.unsqueeze(-1)
            weighted_background = template_background_token * (1 - z_indicate_mask.unsqueeze(-1))
            z_indicate = weighted_foreground + weighted_background


        z = self.patch_embed(z)
        x = self.patch_embed(x)
        # forward stage1&2
        if not self.convmlp and self.stop_grad_conv1:
            x = x.detach() * 0.9 + x * 0.1

        for blk in self.blocks[:-self.num_main_blocks]:
            z = checkpoint.checkpoint(blk, z) if self.grad_ckpt else blk(z)  # bn,c,h,w
            x = checkpoint.checkpoint(blk, x) if self.grad_ckpt else blk(x)  # bn,c,h,w

        x = x.flatten(2).transpose(1, 2)  # bn,l,c
        z = z.flatten(2).transpose(1, 2)

        if self.pos_embed is not None:
            x = x + self.pos_embed[:, :self.num_patches_search, :]
            z = z + self.pos_embed[:, self.num_patches_search:, :]

        if self.token_type_indicate:
            # generate the indicate_embeddings for x
            x_indicate = self.search_token.unsqueeze(0).unsqueeze(1).expand(x.size(0), x.size(1), self.embed_dim)
            # add indicate_embeddings to z and x
            x = x + x_indicate
            z = z + z_indicate


        z = z.view(-1, num_template, z.size(-2), z.size(-1))  # b,n,l,c
        z = z.reshape(z.size(0), -1, z.size(-1))  # b,l,c
        x = x.view(-1, num_search, x.size(-2), x.size(-1))
        x = x.reshape(x.size(0), -1, x.size(-1))

        if text_src is not None:
            xz = torch.cat([x, z, text_src], dim=1)
        else:
            xz = torch.cat([x, z], dim=1)

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            xz = torch.cat([cls_tokens, xz], dim=1)

        return xz

    def forward_features(self, template_list, search_list,template_anno_list, text_src, task_index):
        xz = self.prepare_tokens_with_masks(template_list, search_list, template_anno_list, text_src, task_index)
        xz = self.pos_drop(xz)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks[-self.num_main_blocks:]:
            xz = checkpoint.checkpoint(blk, xz, rel_pos_bias) if self.grad_ckpt else blk(xz, rel_pos_bias)

        xz = self.norm(xz)

        if self.fc_norm is not None:
            return self.fc_norm(xz)
        else:
            return xz

    def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
        xz = self.forward_features(template_list, search_list, template_anno_list, text_src, task_index)
        # x = self.head(x)
        out = [xz]
        return out


def load_pretrained(model, checkpoint, pos_type, patchembed_init):
    if "module" in checkpoint.keys():
        # adjust position encoding
        state_dict = checkpoint["module"]
    elif "model" in checkpoint.keys():
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    pe = state_dict['pos_embed'].float()
    b_pe, hw_pe, c_pe = pe.shape
    side_pe = int(math.sqrt(hw_pe))
    side_num_patches_search = int(math.sqrt(model.num_patches_search))
    side_num_patches_template = int(math.sqrt(model.num_patches_template))
    pe_2D = pe.reshape([b_pe, side_pe, side_pe, c_pe]).permute([0,3,1,2])  #b,c,h,w

    def adjust_pe(pe_2D, side_pe, side_new):
        if pos_type == 'index':
            if side_pe < side_new:
                pe_new_2D = nn.functional.interpolate(pe_2D, [side_new, side_new], align_corners=True, mode='bicubic')
                warnings.warn('The resolution is too large, the POS_TYPE has been modified to \'interpolate\'')
            else:
                pe_new_2D = pe_2D[:,:,0:side_new,0:side_new]
            pe_new = torch.flatten(pe_new_2D.permute([0, 2, 3, 1]), 1, 2)
        elif pos_type == 'interpolate':
            pe_new_2D = nn.functional.interpolate(pe_2D, [side_new, side_new], align_corners=True, mode='bicubic')
            pe_new = torch.flatten(pe_new_2D.permute([0, 2, 3, 1]), 1, 2)#b,l,c
        else:
            raise NotImplementedError('The POS_TYPE should be index or interpolate')
        return pe_new

    if side_pe != side_num_patches_search:
        pe_s = adjust_pe(pe_2D, side_pe, side_num_patches_search)
    else:
        pe_s = pe
    if side_pe != side_num_patches_template:
        pe_t = adjust_pe(pe_2D, side_pe, side_num_patches_template)
    else:
        pe_t = pe
    pe_xz = torch.cat((pe_s, pe_t), dim=1)
    state_dict['pos_embed'] = pe_xz
    auxiliary_keys = ["template_background_token", "template_foreground_token", "search_token"]
    for key in auxiliary_keys:
        if (key in model.state_dict().keys()) and (key not in state_dict.keys()):
            state_dict[key] = model.state_dict()[key]

    ## patch embedding
    patch_embedding_weight = model.state_dict()['patch_embed.proj.weight']
    patch_embedding_weight_pretrained = state_dict['patch_embed.proj.weight']
    
    # 获取模型和预训练权重的输入通道数
    model_in_chans = patch_embedding_weight.shape[1]
    pretrained_in_chans = patch_embedding_weight_pretrained.shape[1]
    
    # 处理不同输入通道数的情况
    if model_in_chans == pretrained_in_chans:
        # 通道数相同，直接赋值
        state_dict['patch_embed.proj.weight'] = patch_embedding_weight_pretrained
    elif model_in_chans > pretrained_in_chans:
        # 模型通道数大于预训练权重通道数（如从3通道扩展到8通道）
        if patchembed_init == "copy":
            # 将预训练权重复制到所有通道
            for i in range(model_in_chans // pretrained_in_chans):
                start_idx = i * pretrained_in_chans
                end_idx = (i + 1) * pretrained_in_chans
                if end_idx <= model_in_chans:
                    patch_embedding_weight[:, start_idx:end_idx, :, :] = patch_embedding_weight_pretrained
            # 处理剩余通道
            remaining = model_in_chans % pretrained_in_chans
            if remaining > 0:
                patch_embedding_weight[:, -remaining:, :, :] = patch_embedding_weight_pretrained[:, :remaining, :, :]
        elif patchembed_init == "halfcopy":
            # 将预训练权重除以2后复制到所有通道
            patch_embedding_weight_pretrained_half = patch_embedding_weight_pretrained / 2
            for i in range(model_in_chans // pretrained_in_chans):
                start_idx = i * pretrained_in_chans
                end_idx = (i + 1) * pretrained_in_chans
                if end_idx <= model_in_chans:
                    patch_embedding_weight[:, start_idx:end_idx, :, :] = patch_embedding_weight_pretrained_half
            # 处理剩余通道
            remaining = model_in_chans % pretrained_in_chans
            if remaining > 0:
                patch_embedding_weight[:, -remaining:, :, :] = patch_embedding_weight_pretrained_half[:, :remaining, :, :]
        elif patchembed_init == "random":
            # 只初始化前几个通道，其余通道保持随机初始化
            patch_embedding_weight[:, :pretrained_in_chans, :, :] = patch_embedding_weight_pretrained
        elif patchembed_init == "insert":
            # 插入预训练权重通道到模型通道中
            # 根据光谱波段映射RGB权重到多光谱通道
            # CIE规定：红(R)700.0nm；绿(G)546.1nm；蓝(B)435.8nm
            R_band = 700.0
            G_band = 546.1
            B_band = 435.8
            # MSITrack数据集的8个波段中心波长
            bands = [422.5, 487.5, 550, 602.5, 660, 725, 785, 887.5]
            
            # 提取RGB三个通道的权重
            R_weight = patch_embedding_weight_pretrained[:, 2, :, :]  # 红色通道
            G_weight = patch_embedding_weight_pretrained[:, 1, :, :]  # 绿色通道
            B_weight = patch_embedding_weight_pretrained[:, 0, :, :]  # 蓝色通道
            
            weight_list = []
            for band in reversed(bands):  # 根据数据读取顺序，可能需要调整reversed
                if band <= 500.0:
                    weight = B_weight  # 蓝色波段
                elif band > 500.0 and band < 620.0:
                    weight = G_weight  # 绿色波段
                else:  # band >= 620.0
                    weight = R_weight  # 红色波段
                
                weight = weight.unsqueeze(1)
                weight_list.append(weight)
            
            # 拼接所有波段的权重
            weight_concat = torch.cat(weight_list, dim=1)
            
            # 如果模型通道数与bands数量不一致，进行调整
            if model_in_chans == len(bands):
                patch_embedding_weight[:, :, :, :] = weight_concat
            else:
                raise ValueError('For "insert" method, model_in_chans must equal to the number of defined bands.')
        
        elif patchembed_init == "insert_halfcopy":
            # 结合insert和halfcopy：根据光谱波段映射RGB权重，并将权重除以2
            # CIE规定：红(R)700.0nm；绿(G)546.1nm；蓝(B)435.8nm
            R_band = 700.0
            G_band = 546.1
            B_band = 435.8
            # MSITrack数据集的8个波段中心波长
            bands = [422.5, 487.5, 550, 602.5, 660, 725, 785, 887.5]
            
            # 提取RGB三个通道的权重并除以2
            R_weight = patch_embedding_weight_pretrained[:, 2, :, :] / 2  # 红色通道
            G_weight = patch_embedding_weight_pretrained[:, 1, :, :] / 2  # 绿色通道
            B_weight = patch_embedding_weight_pretrained[:, 0, :, :] / 2  # 蓝色通道
            
            weight_list = []
            for band in reversed(bands):  # 根据数据读取顺序，可能需要调整reversed
                if band <= 500.0:
                    weight = B_weight  # 蓝色波段
                elif band > 500.0 and band < 620.0:
                    weight = G_weight  # 绿色波段
                else:  # band >= 620.0
                    weight = R_weight  # 红色波段
                
                weight = weight.unsqueeze(1)
                weight_list.append(weight)
            
            # 拼接所有波段的权重
            weight_concat = torch.cat(weight_list, dim=1)
            
            # 如果模型通道数与bands数量不一致，进行
            if model_in_chans == len(bands):
                patch_embedding_weight[:, :, :, :] = weight_concat
            else:
                raise ValueError('For "insert_halfcopy" method, model_in_chans must equal to the number of defined bands.')

        else:
            raise NotImplementedError('cfg.MODEL.ENCODER.PATCHEMBED_INIT must be chosen from copy, halfcopy, or random')
        state_dict['patch_embed.proj.weight'] = patch_embedding_weight
    else:
        # 模型通道数小于预训练权重通道数，这种情况不常见，但我们也可以处理
        patch_embedding_weight[:, :, :, :] = patch_embedding_weight_pretrained[:, :model_in_chans, :, :]
        state_dict['patch_embed.proj.weight'] = patch_embedding_weight
    
    model.load_state_dict(state_dict, strict=False)


@register_model
def fastitpnt(pretrained=False, pos_type="interpolate", pretrain_type="", patchembed_init="copy", **kwargs):
    model = Fast_iTPN(
        patch_size=16, in_chans=16, embed_dim=384, depth_stage1=1, depth_stage2=1, depth=12, num_heads=6, bridge_mlp_ratio=3.,
        mlp_ratio=3., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        convmlp=True,
        naiveswiglu=True,
        subln=True,
        pos_type=pos_type,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(pretrain_type, map_location="cpu")
        load_pretrained(model,checkpoint,pos_type,patchembed_init)
    return model


@register_model
def fastitpns(pretrained=False, pos_type="interpolate", pretrain_type="", patchembed_init="copy", **kwargs):
    model = Fast_iTPN(
        patch_size=16, in_chans=16, embed_dim=384, depth_stage1=2, depth_stage2=2, depth=20, num_heads=6, bridge_mlp_ratio=3.,
        mlp_ratio=3., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        convmlp=True,
        naiveswiglu=True,
        subln=True,
        pos_type=pos_type,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(pretrain_type, map_location="cpu")
        load_pretrained(model,checkpoint,pos_type,patchembed_init)
    return model


@register_model
def fastitpnb(pretrained=False, pos_type="interpolate", pretrain_type="", patchembed_init="copy", **kwargs):
    model = Fast_iTPN(
        patch_size=16, in_chans=16, embed_dim=512, depth_stage1=3, depth_stage2=3, depth=24, num_heads=8, bridge_mlp_ratio=3.,
        mlp_ratio=3., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        convmlp=True,
        naiveswiglu=True,
        subln=True,
        pos_type = pos_type,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(pretrain_type, map_location="cpu")
        load_pretrained(model,checkpoint,pos_type,patchembed_init)
    return model


@register_model
def fastitpnl(pretrained=False, pos_type="interpolate", pretrain_type="", patchembed_init="copy", **kwargs):
    model = Fast_iTPN(
        patch_size=16, in_chans=16, embed_dim=768, depth_stage1=2, depth_stage2=2, depth=40, num_heads=12, bridge_mlp_ratio=3.,
        mlp_ratio=3., qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        convmlp=True,
        naiveswiglu=True,
        subln=True,
        pos_type="interpolate",
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(pretrain_type, map_location="cpu")
        load_pretrained(model,checkpoint,pos_type,patchembed_init)
    return model
