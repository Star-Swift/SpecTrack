import os
import argparse


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--save_dir', type=str, help='root directory to save checkpoints, logs, and tensorboard')
    parser.add_argument('--mode', type=str, choices=["single", "multiple"], default="multiple",
                        help="train on single gpu or multiple gpus")
    parser.add_argument('--nproc_per_node', type=int, help="number of GPUs per node")  # specify when mode is multiple
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # whether datasets are in lmdb format
    # for knowledge distillation
    parser.add_argument('--distill', type=int, choices=[0, 1], default=0)  # whether to use knowledge distillation
    parser.add_argument('--script_teacher', type=str, help='teacher script name')
    parser.add_argument('--config_teacher', type=str, help='teacher yaml configure file name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if args.mode == "single":
        train_cmd = "python lib/train/run_training.py --script %s --config %s --save_dir %s --use_lmdb %d " \
                    % (args.script, args.config, args.save_dir, args.use_lmdb)
    elif args.mode == "multiple":
        train_cmd = "python -m torch.distributed.run --nproc_per_node %d lib/train/run_training.py " \
                    "--script %s --config %s --save_dir %s --use_lmdb %d " \
                    % (args.nproc_per_node, args.script, args.config, args.save_dir, args.use_lmdb)
    else:
        raise ValueError("mode should be 'single' or 'multiple'.")
    print(train_cmd)
    # os.system(train_cmd)
    if args.mode == "single":
        import sys
        import runpy
        # 1. 将项目根目录加入 sys.path，解决 ModuleNotFoundError: No module named 'lib'
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        # 2. 构造 sys.argv 模拟命令行参数传递给 run_training.py
        sys.argv = ["lib/train/run_training.py", 
                    "--script", args.script, 
                    "--config", args.config, 
                    "--save_dir", str(args.save_dir), 
                    "--use_lmdb", str(args.use_lmdb)]
        
        # 3. 直接在当前进程运行模块
        runpy.run_module("lib.train.run_training", run_name="__main__")
    elif args.mode == "multiple":
        os.system(train_cmd)
    else:
        raise ValueError("mode should be 'single' or 'multiple'.")
    
if __name__ == "__main__":
    main()
