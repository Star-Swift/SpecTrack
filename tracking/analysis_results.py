import _init_paths
import argparse

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='musthsi', help='e.g. musthsi / msitrack')
    parser.add_argument('--tracker_name', type=str, default='sutrack')
    parser.add_argument('--tracker_param', type=str, default='sutrack_b224_must_ihmoe')
    parser.add_argument('--display_name', type=str, default=None)
    parser.add_argument('--results_dir', type=str, required=True, help='Path to epoch_x result dir')
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--force_evaluation', type=int, default=1)
    args = parser.parse_args()

    dataset_name = (args.dataset_name or '').strip().lower()
    display_name = args.display_name or args.tracker_param

    trackers = []
    trackers.extend(
        trackerlist(
            name=args.tracker_name,
            parameter_name=args.tracker_param,
            dataset_name=dataset_name,
            run_ids=None,
            display_name=display_name
        )
    )

    # 覆盖结果路径到指定 epoch 目录
    trackers[0].results_dir = args.results_dir

    dataset = get_dataset(dataset_name)

    print_results(
        trackers,
        dataset,
        dataset_name,
        merge_results=True,
        plot_types=('success', 'prec', 'norm_prec'),
        force_evaluation=bool(args.force_evaluation)
    )


if __name__ == '__main__':
    main()

