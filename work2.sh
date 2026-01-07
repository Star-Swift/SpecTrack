
cd /root/autodl-tmp/SUTrack-moe
conda activate must

# CUDA_VISIBLE_DEVICES=2 python tracking/train.py --script sutrack --config sutrack_l224_msi_ihmoe_32_4_2_20 --save_dir . --mode single


# CUDA_VISIBLE_DEVICES=2 python tracking/train.py --script sutrack --config sutrack_l224_msi_ihmoe --save_dir . --mode single

CUDA_VISIBLE_DEVICES=2 python tracking/train.py --script sutrack --config sutrack_b384_msi_ihmoe --save_dir . --mode single

