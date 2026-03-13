from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import torch
import copy
from src.plugins import build_client_plugin
criterion = F.cross_entropy


class BaseClient():
    def __init__(self, options, id, local_dataset, model, optimizer, ):
        self.options = options
        self.id = id
        self.local_dataset = local_dataset
        self.model = model
        # 如果请求使用 GPU 但 CUDA 不可用，则使用 CPU
        self.gpu = options['gpu'] and torch.cuda.is_available()
        self.optimizer = optimizer

        self.plugin = build_client_plugin(options, self.model, self.gpu)
        self.plugin_payload = None

    def set_plugin_payload(self, payload):
        self.plugin_payload = payload

    def set_global_sensitive_feature(self, global_sensitive_feature):
        """Backward-compatible alias for legacy server hook names."""
        self.set_plugin_payload(global_sensitive_feature)

    def set_learning_rate(self, learning_rate):
        for group in self.optimizer.param_groups:
            group['lr'] = learning_rate

    def get_model_parameters(self):
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_parameters(self, model_parameters_dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_parameters_dict[key]
        self.model.load_state_dict(state_dict)

    def local_train(self, ):
        begin_time = time.time()
        local_model_paras, return_dict, aux = self.local_update(self.local_dataset, self.options, )
        end_time = time.time()
        stats = {'id': self.id, "time": round(end_time - begin_time, 2)}
        stats.update(return_dict)
        # Update structure: weights (FedAvg) + num_samples + optional aux (FedFed)
        update = {"weights": local_model_paras, "num_samples": len(self.local_dataset), "aux": aux}
        return update, stats

    def local_update(self, local_dataset, options, ):
        use_plugin = self.plugin is not None

        localTrainDataLoader = DataLoader(local_dataset, batch_size=options['batch_size'], shuffle=True)
        self.model.train()
        if use_plugin:
            self.plugin.on_round_start(self.optimizer.param_groups[0]['lr'], self.plugin_payload)

        train_loss = train_acc = train_total = 0

        for epoch in range(options['local_epoch']):
            train_loss = train_acc = train_total = 0
            for X, y in localTrainDataLoader:
                if self.gpu:
                    X, y = X.cuda(), y.cuda()
                if use_plugin:
                    pred, loss = self.plugin.train_batch(X, y)
                else:
                    self.optimizer.zero_grad()
                    pred = self.model(X)
                    loss = criterion(pred, y)
                    loss.backward()
                    self.optimizer.step()

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)
                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size

        local_model_paras = copy.deepcopy(self.get_model_parameters())
        return_dict = {"id": self.id,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}

        aux = self.plugin.build_upload_payload() if use_plugin else None

        return local_model_paras, return_dict, aux
