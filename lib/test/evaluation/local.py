from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/root/autodl-tmp/SUTrack-moe/data/got10k_lmdb'
    settings.got10k_path = '/root/autodl-tmp/SUTrack-moe/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = '/root/autodl-tmp/SUTrack-moe/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/root/autodl-tmp/SUTrack-moe/data/lasot_lmdb'
    settings.lasot_path = '/root/autodl-tmp/SUTrack-moe/data/lasot'
    settings.lasotlang_path = '/root/autodl-tmp/SUTrack-moe/data/lasot'
    settings.msitrack_path = '/root/autodl-tmp/SUTrack-moe/data/MSITrack'
    settings.network_path = '/root/autodl-tmp/SUTrack-moe/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/root/autodl-tmp/SUTrack-moe/data/nfs'
    settings.otb_path = '/root/autodl-tmp/SUTrack-moe/data/OTB2015'
    settings.otblang_path = '/root/autodl-tmp/SUTrack-moe/data/otb_lang'
    settings.prj_dir = '/root/autodl-tmp/SUTrack-moe'
    settings.result_plot_path = '/root/autodl-tmp/SUTrack-moe/test/result_plots'
    settings.results_path = '/root/autodl-tmp/SUTrack-moe/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/root/autodl-tmp/SUTrack-moe'
    settings.segmentation_path = '/root/autodl-tmp/SUTrack-moe/test/segmentation_results'
    settings.tc128_path = '/root/autodl-tmp/SUTrack-moe/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/root/autodl-tmp/SUTrack-moe/data/tnl2k/test'
    settings.tpl_path = ''
    settings.trackingnet_path = '/root/autodl-tmp/SUTrack-moe/data/trackingnet'
    settings.uav_path = '/root/autodl-tmp/SUTrack-moe/data/UAV123'
    settings.vot_path = '/root/autodl-tmp/SUTrack-moe/data/VOT2019'
    settings.youtubevos_dir = ''
    settings.musthsi_path = '/root/autodl-tmp/SUTrack-moe/data/MUSTHSI'
    settings.msitrack_path = '/root/autodl-tmp/SUTrack-moe/data/MSITrack'
    settings.hot2022_path = '/root/autodl-tmp/hot2022'
    return settings

