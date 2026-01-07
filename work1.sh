
cd /root/autodl-tmp/SUTrack-moe
conda activate must

# CUDA_VISIBLE_DEVICES=1 python tracking/train.py --script sutrack --config sutrack_b224_must_ihmoe_32_4_2_12 --save_dir . --mode single

# CUDA_VISIBLE_DEVICES=1 python tracking/train.py --script sutrack --config sutrack_b224_must_ihmoe_32_4_2_12 --save_dir . --mode single
# CUDA_VISIBLE_DEVICES=1 python tracking/train.py --script sutrack --config sutrack_b224_must_ihmoe_16_4_2_12 --save_dir . --mode single
# CUDA_VISIBLE_DEVICES=1 python tracking/train.py --script sutrack --config sutrack_t224_msi_ihmoe_64_4_2_7 --save_dir . --mode single
# CUDA_VISIBLE_DEVICES=1 python tracking/train.py --script sutrack --config sutrack_b224_must_ihmoe_128_4_2_12 --save_dir . --mode single

# CUDA_VISIBLE_DEVICES=1 python tracking/train.py --script sutrack --config sutrack_b224_must_ihmoe_32_4_2_6 --save_dir . --mode single

# CUDA_VISIBLE_DEVICES=1 python tracking/train.py --script sutrack --config sutrack_b224_must_ihmoe_16_4_2_12 --save_dir . --mode single
CUDA_VISIBLE_DEVICES=1 python tracking/train.py --script sutrack --config sutrack_t224_msi_ihmoe_64_4_2_7_yfj --save_dir . --mode single

CUDA_VISIBLE_DEVICES=1 python tracking/train.py --script sutrack --config sutrack_l224_msi_ihmoe_yfj --save_dir . --mode single