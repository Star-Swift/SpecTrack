"""
SUTrack Model
"""
import torch
import math
from torch import nn
import torch.nn.functional as F
from .encoder import build_encoder
from .decoder import build_decoder
from .task_decoder import build_task_decoder
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed


class SUTRACK(nn.Module):
    """ This is the base class for SUTrack """
    def __init__(self, text_encoder, encoder, decoder, task_decoder,
                 num_frames=1, num_template=1,
                 decoder_type="CENTER", task_feature_type="average"):
        """ Initializes the model.
        """
        super().__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.decoder_type = decoder_type

        self.class_token = False if (encoder.body.cls_token is None) else True
        self.task_feature_type = task_feature_type

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.task_decoder = task_decoder
        self.decoder = decoder

        self.num_frames = num_frames
        self.num_template = num_template


    def forward(self, text_data=None,
                template_list=None, search_list=None, template_anno_list=None,
                text_src=None, task_index=None,
                feature=None, mode="encoder"):
        if mode == "text":
            return self.forward_textencoder(text_data)
        elif mode == "encoder":
            return self.forward_encoder(template_list, search_list, template_anno_list, text_src, task_index)
        elif mode == "decoder":
            return self.forward_decoder(feature), self.forward_task_decoder(feature)
        else:
            raise ValueError

    def forward_textencoder(self, text_data):
        # Forward the encoder
        text_src = self.text_encoder(text_data)
        return text_src

    def forward_encoder(self, template_list, search_list, template_anno_list, text_src, task_index):
        # Forward the encoder
        xz = self.encoder(template_list, search_list, template_anno_list, text_src, task_index)
        return xz

    def forward_decoder(self, feature, gt_score_map=None):

        feature = feature[0]
        if self.class_token:
            feature = feature[:,1:self.num_patch_x * self.num_frames+1]
        else:
            feature = feature[:,0:self.num_patch_x * self.num_frames] # (B, HW, C)

        bs, HW, C = feature.size()
        if self.decoder_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
        if self.decoder_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.decoder(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.decoder_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        elif self.decoder_type == "MLP":
            # run the mlp head
            score_map, bbox, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

    def forward_task_decoder(self, feature):
        feature = feature[0]
        if self.task_feature_type == 'class':
            feature = feature[:, 0:1]
        elif self.task_feature_type == 'text':
            feature = feature[:, -1:]
        elif self.task_feature_type == 'average':
            feature = feature.mean(1).unsqueeze(1)
        else:
            raise NotImplementedError('task_feature_type must be choosen from class, text, and average')
        feature = self.task_decoder(feature)
        return feature

    def get_moce_aux_loss(self):
        """获取编码器中MoCE层的辅助损失"""
        if hasattr(self.encoder.body, 'get_moce_aux_loss'):
            return self.encoder.body.get_moce_aux_loss()
        return 0

    def compute_spectral_contrastive_loss(self, feature, gt_gaussian_map):
        """
        光谱对比损失 (Spectral Contrastive Loss).

        论文图1洞察：目标的光谱特征与模板相似，与背景差异显著。
        当前训练仅有框回归损失和热力图损失，没有损失显式监督编码器利用
        光谱特征区分目标与背景。本方法通过以下方式弥补这一不足：

        - 从编码器输出中分别提取模板原型特征和搜索区域前景/背景特征
        - 使用软 Triplet 损失约束：template_fg 与 search_fg 的余弦相似度
          必须高于与 search_bg 的余弦相似度（margin = 0.1）
        - 显式监督编码器学习光谱判别性特征

        改进论文贡献（2）：使 SHMoE 模块学习真正的光谱语义，而不仅是
        通用的视觉特征。

        参考文献:
        - OSTrack: One-Stream Tracking via Joint Attention (ECCV 2022)
        - SpectralTrack: Spectral-Spatial Visual Tracking (TGRS 2023)
        - SimCLR: A Simple Framework for Contrastive Learning (ICML 2020)

        Args:
            feature: encoder output, list [xz] where xz is [B, N, C]
            gt_gaussian_map: [B, H, W] soft foreground heatmap at patch resolution
        Returns:
            scalar loss
        """
        xz = feature[0]  # [B, N, C]
        B = xz.shape[0]

        # --- Extract search tokens ---
        if self.class_token:
            search_feat = xz[:, 1:self.num_patch_x * self.num_frames + 1, :]
        else:
            search_feat = xz[:, :self.num_patch_x * self.num_frames, :]
        # search_feat: [B, num_patch_x, C]

        # --- Extract template tokens ---
        start = (1 + self.num_patch_x * self.num_frames) if self.class_token else self.num_patch_x * self.num_frames
        end = start + self.num_patch_z * self.num_template
        template_feat = xz[:, start:end, :]  # [B, num_patch_z * num_template, C]

        # Template prototype: mean of all template tokens
        tmpl_proto = F.normalize(template_feat.mean(dim=1), dim=-1)  # [B, C]

        # Foreground weights from GT heatmap
        # gt_gaussian_map: [B, H, W] at patch resolution; H*W must equal num_patch_x
        fg_map_norm = gt_gaussian_map.detach()
        if fg_map_norm.reshape(B, -1).shape[1] != self.num_patch_x:
            # Interpolate heatmap to match patch grid if resolution mismatch
            fg_map_norm = F.interpolate(
                fg_map_norm.unsqueeze(1), size=(self.fx_sz, self.fx_sz),
                mode='bilinear', align_corners=False).squeeze(1)   # [B, fx_sz, fx_sz]
        fg_weights = fg_map_norm.reshape(B, -1)   # [B, num_patch_x]
        fg_weights = fg_weights / (fg_weights.sum(dim=1, keepdim=True) + 1e-8)

        # Background weights: complement of foreground (use same resolution-matched map)
        bg_weights = (1.0 - fg_map_norm).clamp(min=0.0).reshape(B, -1)
        bg_weights = bg_weights / (bg_weights.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted aggregation to get foreground/background prototypes
        search_fg = F.normalize(
            (search_feat * fg_weights.unsqueeze(-1)).sum(dim=1), dim=-1)   # [B, C]
        search_bg = F.normalize(
            (search_feat * bg_weights.unsqueeze(-1)).sum(dim=1), dim=-1)   # [B, C]

        # Soft triplet: template should be spectrally closer to target than to background
        sim_pos = (tmpl_proto * search_fg).sum(dim=-1)   # [B]
        sim_neg = (tmpl_proto * search_bg).sum(dim=-1)   # [B]
        loss = F.relu(sim_neg - sim_pos + 0.1).mean()
        return loss


def build_sutrack(cfg):
    encoder = build_encoder(cfg)
    text_encoder = None
    decoder = build_decoder(cfg, encoder)
    task_decoder = build_task_decoder(cfg, encoder)
    model = SUTRACK(
        text_encoder,
        encoder,
        decoder,
        task_decoder,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
        task_feature_type=cfg.MODEL.TASK_DECODER.FEATURE_TYPE
    )

    return model
