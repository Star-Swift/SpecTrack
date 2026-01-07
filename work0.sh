cd /root/autodl-tmp/SUTrack-moe
conda activate must

# CUDA_VISIBLE_DEVICES=0 python tracking/train.py --script sutrack --config sutrack_t224_must_ihmoe_64_4_6 --save_dir . --mode single
# CUDA_VISIBLE_DEVICES=0 python tracking/train.py --script sutrack --config sutrack_b224_msi_hc --save_dir . --mode single

# CUDA_VISIBLE_DEVICES=0 python tracking/train.py --script sutrack --config sutrack_b224_msi_ihmoe --save_dir . --mode single
# CUDA_VISIBLE_DEVICES=0 python tracking/train.py --script sutrack --config sutrack_b224_must_ihmoe_64_3_2_12 --save_dir . --mode single

# CUDA_VISIBLE_DEVICES=0 python tracking/train.py --script sutrack --config sutrack_b224_must_ihmoe_64_5_2_12 --save_dir . --mode single
# CUDA_VISIBLE_DEVICES=0 python tracking/train.py --script sutrack --config sutrack_b224_msi_ihmoe_32_4_2_12_yfj --save_dir . --mode single


CUDA_VISIBLE_DEVICES=0 python tracking/train.py --script sutrack --config sutrack_b224_must_ihmoe --save_dir . --mode single
# 批量测试 + 自动评估（写入当天日志）
# 用法示例：
#   bash testval.sh --param sutrack_b224_must_ihmoe --dataset MUSTHSI --gpu 0 --threads 20