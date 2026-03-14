import json
import os

from tensorboardX import SummaryWriter

from src.plugins import resolve_plugin_name
from src.utils.plotting import save_single_run_plots


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


class Metrics(object):
    def __init__(self, options, clients, name=''):
        self.options = options

        num_rounds = options['round_num'] + 1
        self.bytes_written = {c.id: [0] * num_rounds for c in clients}
        self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}

        self.loss_on_g_test_data = [0] * num_rounds
        self.acc_on_g_test_data = [0] * num_rounds

        self.accumulation_delay = [0] * num_rounds
        self.accumulation_energy = [0] * num_rounds

        self.result_path = mkdir(os.path.join('./result', self.options['dataset_name']))
        suffix = '{}_sd{}_lr{}_ne{}_bs{}'.format(
            name,
            options['seed'],
            options['lr'],
            options['round_num'],
            options['batch_size'],
        )
        self.exp_name = '{}_{}'.format(options['model_name'], suffix)
        self.exp_dir = mkdir(os.path.join(self.result_path, self.exp_name))

        train_event_folder = mkdir(os.path.join(self.exp_dir, 'train.event'))
        test_event_folder = mkdir(os.path.join(self.exp_dir, 'eval.event'))
        self.train_writer = SummaryWriter(train_event_folder)
        self.eval_writer = SummaryWriter(test_event_folder)

    def update_test_stats(self, round_i, eval_stats):
        self.loss_on_g_test_data[round_i] = eval_stats['loss']
        self.acc_on_g_test_data[round_i] = eval_stats['acc']

        self.eval_writer.add_scalar('test_loss', eval_stats['loss'], round_i)
        self.eval_writer.add_scalar('test_acc', eval_stats['acc'], round_i)

    def write(self):
        metrics = {
            'dataset': self.options['dataset_name'],
            'model_name': self.options['model_name'],
            'plugin_name': resolve_plugin_name(self.options) or 'none',
            'loss_on_g_test_data': self.loss_on_g_test_data,
            'acc_on_g_test_data': self.acc_on_g_test_data,
            'best_test_acc': max(self.acc_on_g_test_data),
            'final_test_acc': self.acc_on_g_test_data[-1],
            'best_test_loss': min(self.loss_on_g_test_data),
            'final_test_loss': self.loss_on_g_test_data[-1],
            'options': self.options,
        }
        metrics_dir = os.path.join(self.exp_dir, 'metrics.json')

        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf, indent=2)

        save_single_run_plots(metrics, self.exp_dir)
