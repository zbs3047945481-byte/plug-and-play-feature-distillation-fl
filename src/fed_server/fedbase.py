import numpy as np
import torch
import time
from src.fed_client.client import BaseClient
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import copy
from src.utils.metrics import Metrics
from src.utils.tools import apply_feature_skew, build_client_feature_skews
from src.plugins import build_server_plugin
import torch.nn.functional as F
criterion = F.cross_entropy


class BaseFederated(object):
    def __init__(self, options, dataset, clients_label, model=None, optimizer=None,
                 model_builder=None, optimizer_builder=None, name=''):
        if model is not None and optimizer is not None:
            self.model = model
            self.optimizer = optimizer
        self.options = options
        self.dataset = dataset
        self.clients_label = clients_label
        self.model_builder = model_builder
        self.optimizer_builder = optimizer_builder
        if self.model_builder is None or self.optimizer_builder is None:
            raise ValueError('Both model_builder and optimizer_builder are required.')
        # 如果请求使用 GPU 但 CUDA 不可用，则使用 CPU
        self.gpu = options['gpu'] and torch.cuda.is_available()
        self.batch_size = options['batch_size']
        self.num_round = options['round_num']
        self.per_round_c_fraction = options['c_fraction']
        self.client_feature_skews = build_client_feature_skews(len(self.clients_label), options)
        self.clients = self.setup_clients(self.dataset, self.clients_label)
        self.clients_num = len(self.clients)
        self.name = '_'.join([name, f'wn{int(self.per_round_c_fraction * self.clients_num)}',
                              f'tn{len(self.clients)}'])
        self.metrics = Metrics(options, self.clients, self.name)
        self.latest_global_model = copy.deepcopy(self.get_model_parameters())
        self.server_plugin = build_server_plugin(options, self.gpu)

    @staticmethod
    def move_model_to_gpu(model, options):
        if options['gpu'] is True and torch.cuda.is_available():
            device = 0
            torch.cuda.set_device(device)
            # torch.backends.cudnn.enabled = True
            model.cuda()
            print('>>> Use gpu on device {}'.format(device))
        else:
            if options['gpu'] is True:
                print('>>> GPU requested but CUDA not available, using CPU instead')
            else:
                print('>>> Don not use gpu')

    def get_model_parameters(self):
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_parameters(self, model_parameters_dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_parameters_dict[key]
        self.model.load_state_dict(state_dict)

    def train(self):
        """The whole training procedure

        No returns. All results all be saved.
        """
        raise NotImplementedError

    def setup_clients(self, dataset, clients_label):
        train_data = dataset.train_data
        train_label = dataset.train_label
        all_client = []
        for i in range(len(clients_label)):
            local_indices = self.clients_label[i]
            local_train_data = apply_feature_skew(train_data[local_indices], self.client_feature_skews[i])
            local_train_label = train_label[local_indices]

            local_model = self.model_builder()
            if self.gpu:
                local_model.cuda()
            local_optimizer = self.optimizer_builder(local_model.parameters())
            local_dataset = TensorDataset(
                torch.tensor(local_train_data, dtype=torch.float32),
                torch.tensor(local_train_label, dtype=torch.long),
            )
            local_client = BaseClient(
                self.options,
                i,
                local_dataset,
                local_model,
                local_optimizer,
            )
            all_client.append(local_client)

        return all_client

    def local_train(self, round_i, select_clients, ):
        local_model_paras_set = []
        stats = []
        for i, client in enumerate(select_clients, start=1):
            client.set_model_parameters(self.latest_global_model)
            client.set_learning_rate(self.optimizer.param_groups[0]['lr'])
            if self.server_plugin is not None:
                client.set_plugin_payload(self.server_plugin.build_broadcast_payload())
            update, stat = client.local_train()
            local_model_paras_set.append(update)
            stats.append(stat)
            if True:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s ".format(
                       round_i, client.id, i, int(self.per_round_c_fraction * self.clients_num),
                       stat['loss'], stat['acc'] * 100, stat['time'], ))
        return local_model_paras_set, stats



    def aggregate_parameters(self, local_model_paras_set):
        # Each element is update dict: {"weights", "num_samples", "aux"}
        averaged_paras = copy.deepcopy(self.model.state_dict())
        train_data_num = 0
        for var in averaged_paras:
            averaged_paras[var] = 0
        for update in local_model_paras_set:
            num_sample = update["num_samples"]
            local_model_paras = update["weights"]
            for var in averaged_paras:
                averaged_paras[var] += num_sample * local_model_paras[var]
            train_data_num += num_sample
        for var in averaged_paras:
            averaged_paras[var] /= train_data_num

        # FedFed plugin: aggregate class-wise prototypes.
        if self.server_plugin is not None:
            self.server_plugin.aggregate_client_payloads(local_model_paras_set)
        return averaged_paras



    def test_latest_model_on_testdata(self, round_i):
        # Collect stats from total test data
        begin_time = time.time()
        stats_from_test_data = self.global_test(use_test_data=True)
        end_time = time.time()

        if True:
            print('= Test = round: {} / acc: {:.3%} / '
                  'loss: {:.4f} / Time: {:.2f}s'.format(
                   round_i, stats_from_test_data['acc'],
                   stats_from_test_data['loss'], end_time-begin_time))
            print('=' * 102 + "\n")

        self.metrics.update_test_stats(round_i, stats_from_test_data)

    def global_test(self, use_test_data=True):
        assert self.latest_global_model is not None
        self.set_model_parameters(self.latest_global_model)
        test_data = self.dataset.test_data
        test_label = self.dataset.test_label
        testDataLoader = DataLoader(
            TensorDataset(
                torch.tensor(test_data, dtype=torch.float32),
                torch.tensor(test_label, dtype=torch.long),
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for X, y in testDataLoader:
                if self.gpu:
                    X, y = X.cuda(), y.cuda()
                pred = self.model(X)
                loss = criterion(pred, y)
                _, predicted = torch.max(pred, 1)

                correct = predicted.eq(y).sum()
                test_acc += correct.item()
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        stats = {'acc': test_acc / test_total,
                 'loss': test_loss / test_total,
                 'num_samples': test_total,}
        return stats
