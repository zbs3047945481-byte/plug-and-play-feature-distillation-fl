import argparse
import json
import os

from src.utils.plotting import plotting_available, save_comparison_plots


def parse_args():
    parser = argparse.ArgumentParser(description='Plot comparison charts from multiple experiment metrics.json files.')
    parser.add_argument(
        '--metrics',
        nargs='+',
        required=True,
        help='Paths to metrics.json files.',
    )
    parser.add_argument(
        '--labels',
        nargs='*',
        default=None,
        help='Optional labels for each metrics file.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save generated comparison plots.',
    )
    return parser.parse_args()


def load_experiments(metrics_paths, labels):
    experiments = []
    for index, metrics_path in enumerate(metrics_paths):
        with open(metrics_path, 'r') as infile:
            metrics = json.load(infile)
        label = labels[index] if labels and index < len(labels) else build_default_label(metrics, metrics_path)
        experiments.append({
            'label': label,
            'metrics': metrics,
        })
    return experiments


def build_default_label(metrics, metrics_path):
    plugin_name = metrics.get('plugin_name', 'none')
    model_name = metrics.get('model_name', 'model')
    folder_name = os.path.basename(os.path.dirname(metrics_path))
    return '{}-{}-{}'.format(model_name, plugin_name, folder_name)


def main():
    args = parse_args()
    experiments = load_experiments(args.metrics, args.labels)
    if not plotting_available():
        raise RuntimeError('matplotlib is required to generate comparison plots.')
    os.makedirs(args.output_dir, exist_ok=True)
    save_comparison_plots(experiments, args.output_dir)
    print('Saved comparison plots to {}'.format(args.output_dir))


if __name__ == '__main__':
    main()
