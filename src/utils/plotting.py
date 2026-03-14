import os
import tempfile


try:
    os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'matplotlib-cache'))
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def plotting_available():
    return plt is not None


def save_single_run_plots(metrics, output_dir):
    if not plotting_available():
        return False

    rounds = list(range(len(metrics['acc_on_g_test_data'])))
    acc_values = metrics['acc_on_g_test_data']
    loss_values = metrics['loss_on_g_test_data']

    _save_curve(
        rounds,
        acc_values,
        os.path.join(output_dir, 'test_acc_curve.png'),
        'Test Accuracy',
        'Round',
        'Accuracy',
    )
    _save_curve(
        rounds,
        loss_values,
        os.path.join(output_dir, 'test_loss_curve.png'),
        'Test Loss',
        'Round',
        'Loss',
    )
    return True


def save_comparison_plots(experiments, output_dir):
    if not plotting_available() or not experiments:
        return False

    os.makedirs(output_dir, exist_ok=True)
    _save_multi_curve(
        experiments,
        output_dir,
        metric_key='acc_on_g_test_data',
        filename='compare_test_acc.png',
        title='Test Accuracy Comparison',
        ylabel='Accuracy',
    )
    _save_multi_curve(
        experiments,
        output_dir,
        metric_key='loss_on_g_test_data',
        filename='compare_test_loss.png',
        title='Test Loss Comparison',
        ylabel='Loss',
    )
    _save_summary_bar(
        experiments,
        output_dir,
        metric_getter=lambda exp: exp['metrics'].get('best_test_acc', 0.0),
        filename='compare_best_acc_bar.png',
        title='Best Test Accuracy',
        ylabel='Accuracy',
    )
    _save_summary_bar(
        experiments,
        output_dir,
        metric_getter=lambda exp: exp['metrics'].get('final_test_acc', 0.0),
        filename='compare_final_acc_bar.png',
        title='Final Test Accuracy',
        ylabel='Accuracy',
    )
    return True


def _save_curve(x_values, y_values, output_path, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_values, y_values, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_multi_curve(experiments, output_dir, metric_key, filename, title, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5))
    for experiment in experiments:
        metric_values = experiment['metrics'][metric_key]
        ax.plot(range(len(metric_values)), metric_values, linewidth=2, label=experiment['label'])
    ax.set_title(title)
    ax.set_xlabel('Round')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close(fig)


def _save_summary_bar(experiments, output_dir, metric_getter, filename, title, ylabel):
    labels = [experiment['label'] for experiment in experiments]
    values = [metric_getter(experiment) for experiment in experiments]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close(fig)
