"""
Encoder modules: we use ITPN for the encoder.
"""

import os
from torch import nn
from lib.utils.misc import is_main_process
from lib.models.spectrack import fastitpn as fastitpn_module
from lib.models.spectrack import itpn as oriitpn_module



class EncoderBase(nn.Module):

    def __init__(self, encoder: nn.Module, train_encoder: bool, open_layers: list, num_channels: int):
        super().__init__()
        open_blocks = open_layers[2:]
        open_items = open_layers[0:2]
        for name, parameter in encoder.named_parameters():

            if not train_encoder:
                freeze = True
                for open_block in open_blocks:
                    if open_block in name:
                        freeze = False
                if name in open_items:
                    freeze = False
                if freeze == True:
                    parameter.requires_grad_(False)  # here should allow users to specify which layers to freeze !

        self.body = encoder
        self.num_channels = num_channels

    def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
        xs = self.body(template_list, search_list, template_anno_list, text_src, task_index)
        return xs


class Encoder(EncoderBase):
    """ViT encoder."""
    def __init__(self, name: str,
                 train_encoder: bool,
                 search_size: int,
                 template_size: int,
                 open_layers: list,
                 cfg=None):
        pretrain_type = cfg.MODEL.ENCODER.PRETRAIN_TYPE
        load_pretrained = is_main_process() and bool(pretrain_type) and os.path.exists(pretrain_type)
        if is_main_process() and pretrain_type and not load_pretrained:
            print(f"WARNING: pretrained encoder checkpoint not found: {pretrain_type}. Training from scratch.")

        if "fastitpn" in name.lower():
            encoder = getattr(fastitpn_module, name)(
                pretrained=load_pretrained,
                search_size=search_size,
                template_size=template_size,
                drop_rate=0.0,
                drop_path_rate=0.1,
                attn_drop_rate=0.0,
                init_values=0.1,
                drop_block_rate=None,
                use_mean_pooling=True,
                grad_ckpt=False,
                cls_token=cfg.MODEL.ENCODER.CLASS_TOKEN,
                pos_type=cfg.MODEL.ENCODER.POS_TYPE,
                token_type_indicate=cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE,
                pretrain_type=pretrain_type,
                patchembed_init=cfg.MODEL.ENCODER.PATCHEMBED_INIT,
                # MoCE配置
                use_moce=getattr(cfg.MODEL.ENCODER, 'USE_MOCE', False),
                moce_rank=getattr(cfg.MODEL.ENCODER, 'MOCE_RANK', 64),
                moce_rank_scaling=getattr(cfg.MODEL.ENCODER, 'MOCE_RANK_SCALING', 'constant'),
                moce_num_experts=getattr(cfg.MODEL.ENCODER, 'MOCE_NUM_EXPERTS', 4),
                moce_top_k=getattr(cfg.MODEL.ENCODER, 'MOCE_TOP_K', 2),
                moce_start_layer=getattr(cfg.MODEL.ENCODER, 'MOCE_START_LAYER', 0),
                moce_expert_type=getattr(cfg.MODEL.ENCODER, 'MOCE_EXPERT_TYPE', 'heterogeneous'),
                moce_depth_type=getattr(cfg.MODEL.ENCODER, 'MOCE_DEPTH_TYPE', 'constant'),
                moce_use_freq=getattr(cfg.MODEL.ENCODER, 'MOCE_USE_FREQ', True),
                moce_use_shared=getattr(cfg.MODEL.ENCODER, 'MOCE_USE_SHARED', True),
                moce_freq_mode=getattr(cfg.MODEL.ENCODER, 'MOCE_FREQ_MODE', 'spatial_spectral'),
                moce_shared_type=getattr(cfg.MODEL.ENCODER, 'MOCE_SHARED_TYPE', 'attention'),
                moce_use_complexity_bias=getattr(cfg.MODEL.ENCODER, 'MOCE_USE_COMPLEXITY_BIAS', True),
                moce_complexity_scale=getattr(cfg.MODEL.ENCODER, 'MOCE_COMPLEXITY_SCALE', 'max'),
                moce_spectral_permute_seed=getattr(cfg.MODEL.ENCODER, 'MOCE_SPECTRAL_PERMUTE_SEED', -1),
                moce_spectral_source=getattr(cfg.MODEL.ENCODER, 'MOCE_SPECTRAL_SOURCE', 'latent'),
                moce_routing_noise_eval=getattr(cfg.MODEL.ENCODER, 'MOCE_ROUTING_NOISE_EVAL', True),
            )
            if "itpnb" in name:
                num_channels = 512
            elif "itpnl" in name:
                num_channels = 768
            elif "itpnt" in name:
                num_channels = 384
            elif "itpns" in name:
                num_channels = 384
            else:
                num_channels = 512
        elif "oriitpn" in name.lower():
            encoder = getattr(oriitpn_module, name)(
                pretrained=load_pretrained,
                search_size=search_size,
                template_size=template_size,
                drop_path_rate=0.1,
                init_values=0.1,
                use_mean_pooling=True,
                ape=True,
                rpe=True,
                pos_type=cfg.MODEL.ENCODER.POS_TYPE,
                token_type_indicate=cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE,
                task_num=cfg.MODEL.TASK_NUM,
                pretrain_type=pretrain_type
            )
            if "itpnb" in name:
                num_channels = 512
            else:
                num_channels = 512
        else:
            raise ValueError()
        super().__init__(encoder, train_encoder, open_layers, num_channels)



def build_encoder(cfg):
    train_encoder = (cfg.TRAIN.ENCODER_MULTIPLIER > 0) and (cfg.TRAIN.FREEZE_ENCODER == False)
    encoder = Encoder(cfg.MODEL.ENCODER.TYPE, train_encoder,
                      cfg.DATA.SEARCH.SIZE,
                      cfg.DATA.TEMPLATE.SIZE,
                      cfg.TRAIN.ENCODER_OPEN, cfg)
    return encoder
