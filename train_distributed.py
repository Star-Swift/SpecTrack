#!/usr/bin/env python3
"""
SUTrack 分布式训练脚本
支持4个GPU (0-3)训练
"""

import os
import sys
import subprocess
import argparse
import importlib
import cv2 as cv
import torch
import torch.backends.cudnn
import torch.distributed as dist
import random
import numpy as np

# 设置CUDA可见设备为0-3
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

# 添加项目路径到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
os.environ['PYTHONPATH'] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"

torch.backends.cudnn.benchmark = False

# 修复 PyTorch 2.6 的 weights_only 问题
import torch.serialization
try:
    from lib.train.admin.stats import AverageMeter
    # 添加安全全局变量
    torch.serialization.add_safe_globals([AverageMeter])
    print("✅ Added AverageMeter to safe globals")
except (ImportError, AttributeError) as e:
    print(f"⚠️  Warning: Failed to add safe globals: {e}")

import lib.train.admin.settings as ws_settings
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from lib.train.trainers.ltr_trainer import LTRTrainer
from torch.nn.parallel import DistributedDataParallel as DDP
from lib.train.base_functions import *
from lib.models.sutrack.sutrack import build_sutrack
from lib.train.actors import SUTrack_Actor
from lib.utils.focal_loss import FocalLoss


def init_seeds(seed=42, local_rank=0):
    """初始化随机种子"""
    seed = seed + local_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_distributed():
    """设置分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        print(f"Initializing distributed training: rank={rank}, local_rank={local_rank}, world_size={world_size}")
        
        # 初始化进程组
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # 设置当前GPU
        torch.cuda.set_device(local_rank)
        
        return rank, local_rank, world_size
    else:
        # 单卡训练
        return 0, 0, 1


def check_decoder_type(net, local_rank):
    """检查decoder类型和参数"""
    if local_rank == 0:  # 只在rank 0输出
        print("\n" + "="*50)
        print("DECODER INSPECTION:")
        print("="*50)
        
        decoder = net.module.decoder if hasattr(net, 'module') else net.decoder
        
        if hasattr(net, 'module'):
            decoder = net.module.decoder
        else:
            decoder = net.decoder if hasattr(net, 'decoder') else None
        
        if decoder is not None:
            print(f"Decoder type: {type(decoder).__name__}")
            
            # 检查参数数量
            decoder_params = sum(p.numel() for p in decoder.parameters())
            trainable_decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
            
            print(f"Decoder parameters: {decoder_params:,}")
            print(f"Trainable decoder parameters: {trainable_decoder_params:,}")
            
            # 检查MoE组件
            has_moe = False
            for name, module in decoder.named_modules():
                module_name = type(module).__name__.lower()
                if any(keyword in module_name for keyword in ['moe', 'expert', 'gate', 'router']):
                    has_moe = True
                    print(f"✅ Found MoE component: {name} ({type(module).__name__})")
            
            if not has_moe:
                print("ℹ️  No MoE components detected in decoder")
        else:
            print("❌ No decoder found in the network!")
        
        print("="*50)


def run_training():
    """训练主函数"""
    try:
        # 固定配置参数
        script_name = "sutrack"
        config_name = "sutrack_b224_must"  # 使用你的配置
        save_dir = "./checkpoints"
        use_lmdb = 0
        base_seed = 42
        
        # 设置分布式
        rank, local_rank, world_size = setup_distributed()
        
        if rank == 0:
            print(f"\n🚀 Starting SUTrack training with {world_size} GPUs")
            print(f"Script: {script_name}")
            print(f"Config: {config_name}")
            print(f"Save dir: {save_dir}")
            print(f"Local rank: {local_rank}, Global rank: {rank}, World size: {world_size}")
        
        # 初始化随机种子
        init_seeds(base_seed, local_rank)
        
        # 设置训练环境
        settings = ws_settings.Settings()
        settings.script_name = script_name
        settings.config_name = config_name
        settings.project_path = f'train/{script_name}/{config_name}'
        settings.local_rank = local_rank
        settings.save_dir = os.path.abspath(save_dir)
        settings.use_lmdb = bool(use_lmdb)
        
        # 构建配置文件路径
        prj_dir = project_root
        settings.cfg_file = os.path.join(prj_dir, 'experiments', script_name, f'{config_name}.yaml')
        if not os.path.exists(settings.cfg_file):
            legacy_cfg = os.path.join(os.path.dirname(prj_dir), 'experiments', script_name, f'{config_name}.yaml')
            if os.path.exists(legacy_cfg):
                settings.cfg_file = legacy_cfg
        
        if rank == 0:
            print(f"Config file: {settings.cfg_file}")
            if not os.path.exists(settings.cfg_file):
                raise FileNotFoundError(f"Config file not found: {settings.cfg_file}")
        
        # 更新配置
        config_module = importlib.import_module(f"lib.config.{script_name}.config")
        cfg = config_module.cfg
        config_module.update_config_from_file(settings.cfg_file)
        
        if rank == 0:
            print("\n📋 Configuration:")
            for key in cfg.keys():
                print(f"  {key}: {cfg[key]}")
            print()
        
        # 更新设置
        from lib.train.base_functions import update_settings
        update_settings(settings, cfg)
        
        # 构建数据加载器
        loader_type = getattr(cfg.DATA, "LOADER", "tracking")
        if loader_type == "tracking":
            loader_train = build_dataloaders(cfg, settings)
        else:
            raise ValueError("illegal DATA LOADER")
        
        if rank == 0:
            print(f"✅ Data loader created with {len(loader_train)} batches per epoch")
        
        # 创建网络
        if settings.script_name == "sutrack":
            if rank == 0:
                print("\n🔨 Building SUTrack model...")
            
            # 构建模型
            net = build_sutrack(cfg)
            
            # 检查decoder类型
            if rank == 0:
                check_decoder_type(net, local_rank)
            
            # 尝试加载预训练权重
            checkpoint_path = 'checkpoints/train/sutrack/sutrack_b224/SUTRACK_ep0180.pth.tar'
            if os.path.exists(checkpoint_path) and rank == 0:
                print(f"\n📂 Loading pretrained weights from: {checkpoint_path}")
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    if 'net' in checkpoint:
                        # 加载权重
                        model_dict = net.state_dict()
                        pretrained_dict = {k: v for k, v in checkpoint['net'].items() 
                                         if k in model_dict and model_dict[k].shape == v.shape}
                        
                        model_dict.update(pretrained_dict)
                        net.load_state_dict(model_dict, strict=False)
                        
                        print(f"✅ Loaded {len(pretrained_dict)} parameters from checkpoint")
                    else:
                        print("⚠️  No 'net' key in checkpoint")
                except Exception as e:
                    print(f"⚠️  Failed to load checkpoint: {e}")
            elif rank == 0:
                print("ℹ️  No pretrained checkpoint found, using random initialization")
        
        else:
            raise ValueError(f"Unsupported script: {settings.script_name}")
        
        # 将模型移动到GPU
        net = net.cuda()
        
        # 分布式数据并行
        if world_size > 1:
            if rank == 0:
                print(f"\n🔀 Wrapping model with DistributedDataParallel")
            
            net = DDP(
                net,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,  # 改为False以避免警告
                broadcast_buffers=True
            )
        
        # 计算参数统计
        if rank == 0:
            total_params = sum(p.numel() for p in net.parameters())
            trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            
            print("\n📊 Model Statistics:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Percentage trainable: {trainable_params/total_params*100:.2f}%")
            print()
        
        # 损失函数
        focal_loss = FocalLoss()
        objective = {
            'giou': giou_loss, 
            'l1': l1_loss, 
            'focal': focal_loss, 
            'cls': BCEWithLogitsLoss(),
            'task_cls': CrossEntropyLoss()
        }
        
        loss_weight = {
            'giou': cfg.TRAIN.GIOU_WEIGHT, 
            'l1': cfg.TRAIN.L1_WEIGHT, 
            'focal': 1., 
            'cls': cfg.TRAIN.CE_WEIGHT,
            'task_cls': cfg.TRAIN.TASK_CE_WEIGHT
        }
        
        # 创建actor
        actor = SUTrack_Actor(
            net=net, 
            objective=objective, 
            loss_weight=loss_weight, 
            settings=settings, 
            cfg=cfg
        )
        
        # 优化器和学习率调度器
        optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
        use_amp = getattr(cfg.TRAIN, "AMP", False)
        
        # 创建训练器
        trainer = LTRTrainer(
            actor, 
            [loader_train], 
            optimizer, 
            settings, 
            lr_scheduler, 
            use_amp=use_amp
        )
        
        if rank == 0:
            print("\n🚀 Starting training...")
            print("="*60)
        
        # 开始训练
        trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
        
        if rank == 0:
            print("\n✅ Training completed successfully!")
            print("="*60)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理分布式进程组
        if dist.is_initialized():
            dist.destroy_process_group()
        
        raise e


def main():
    """主函数：启动分布式训练"""
    parser = argparse.ArgumentParser(description='SUTrack Distributed Training')
    parser.add_argument('--script', type=str, default='sutrack', help='Script name')
    parser.add_argument('--config', type=str, default='sutrack_b224_must', help='Config name')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--gpus', type=int, default=4, help='GPUs per node')
    
    args = parser.parse_args()
    
    # 从环境变量获取local_rank
    env_local_rank = os.environ.get('LOCAL_RANK', -1)
    if env_local_rank != -1:
        args.local_rank = int(env_local_rank)
    
    # 检查是否在分布式环境中
    if args.local_rank != -1:
        # 在分布式环境中，直接运行训练
        run_training()
    else:
        # 不在分布式环境中，启动torchrun
        print("🚀 Starting distributed training with torchrun...")
        print(f"  Script: {args.script}")
        print(f"  Config: {args.config}")
        print(f"  Save dir: {args.save_dir}")
        print(f"  GPUs: {args.gpus}")
        print()
        
        cmd = [
            'torchrun',
            '--nproc_per_node', str(args.gpus),
            '--nnodes', '1',
            '--node_rank', '0',
            '--master_addr', '127.0.0.1',
            '--master_port', '29500',
            __file__,  # 运行当前脚本
            '--script', args.script,
            '--config', args.config,
            '--save_dir', args.save_dir
        ]
        
        print(f"Command: {' '.join(cmd)}")
        print("="*60)
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Distributed training failed with exit code: {e.returncode}")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")
            sys.exit(0)


if __name__ == '__main__':
    # 设置分布式训练优化参数
    os.environ['NCCL_DEBUG'] = 'WARN'  # 设置NCCL调试级别
    os.environ['NCCL_IB_DISABLE'] = '1'  # 禁用InfiniBand
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo,eth0'  # 网络接口
    os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'
    os.environ['NCCL_SOCKET_NTHREADS'] = '2'
    os.environ['NCCL_MIN_NCHANNELS'] = '4'
    
    # 设置CUDA相关环境变量
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 调试用，正式训练可设为0
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # 运行主函数
    main()