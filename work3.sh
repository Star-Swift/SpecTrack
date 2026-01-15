
cd /root/autodl-tmp/SUTrack-moe
conda activate must

python tracking/train.py --script sutrack --config hot2022_t224 --save_dir . --mode single
python tracking/train.py --script sutrack --config hot2022_b224 --save_dir . --mode single
python tracking/train.py --script sutrack --config hot2022_l224 --save_dir . --mode single

cp -r /root/autodl-tmp/SUTrack-moe/checkpoints/train/sutrack/hot2022_t224 /root/autodl-fs/checkpoints/hot2022_t224
cp -r /root/autodl-fs/checkpoints/hot2022_b224 /root/autodl-tmp/SUTrack-moe/checkpoints/train/sutrack/hot2022_b224
sudo shutdown -h now