from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import numpy as np
import torch.nn as nn
import torch
import copy
from src.models.feature_split import FeatureSplitModule
criterion = F.cross_entropy
mse_loss = nn.MSELoss()


class BaseClient():
    def __init__(self, options, id, local_dataset, model, optimizer, ):
        self.options = options
        self.id = id
        self.local_dataset = local_dataset
        self.model = model
        # 如果请求使用 GPU 但 CUDA 不可用，则使用 CPU
        self.gpu = options['gpu'] and torch.cuda.is_available()
        self.optimizer = optimizer

        # FedFed plugin: optional feature split module (local only, not aggregated)
        self.use_fedfed_plugin = options.get('use_fedfed_plugin', False)
        self.feature_split_module = None
        self.local_optimizer = None
        self.global_sensitive_feature = None  # set by server before local_train
        if self.use_fedfed_plugin:
            fd = options.get('fedfed_feature_dim', 512)
            sd = options.get('fedfed_sensitive_dim', 64)
            self.feature_split_module = FeatureSplitModule(fd, sd)
            if self.gpu:
                self.feature_split_module.cuda()
            # Local optimizer includes model + feature_split for plugin training
            self.local_optimizer = torch.optim.Adam(
                list(self.model.parameters()) + list(self.feature_split_module.parameters()),
                lr=options.get('lr', 0.001)
            )

    def set_global_sensitive_feature(self, global_sensitive_feature):
        """Set global sensitive feature from server (for L_distill). None when plugin off or round 0."""
        self.global_sensitive_feature = global_sensitive_feature

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

    def _clip_and_noise_z_s(self, z_s, clip_norm, noise_sigma):
        """L2 clip (scale down if ||z_s|| > clip_norm) and add Gaussian noise (engineering-level privacy)."""
        if clip_norm is not None and clip_norm > 0:
            norm = z_s.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = torch.clamp(clip_norm / norm, max=1.0)  # scale down only when norm > clip_norm
            z_s = z_s * scale
        if noise_sigma is not None and noise_sigma > 0:
            z_s = z_s + noise_sigma * torch.randn_like(z_s, device=z_s.device)
        return z_s

    def local_update(self, local_dataset, options, ):
        use_plugin = self.use_fedfed_plugin
        optimizer = self.local_optimizer if use_plugin else self.optimizer
        if use_plugin and self.local_optimizer is not None:
            # Sync lr from server optimizer
            self.local_optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']

        localTrainDataLoader = DataLoader(local_dataset, batch_size=options['batch_size'], shuffle=True)
        self.model.train()
        if use_plugin and self.feature_split_module is not None:
            self.feature_split_module.train()

        train_loss = train_acc = train_total = 0
        z_s_list = []  # collect z_s for mean (only when plugin on)

        for epoch in range(options['local_epoch']):
            train_loss = train_acc = train_total = 0
            for X, y in localTrainDataLoader:
                if self.gpu:
                    X, y = X.cuda(), y.cuda()
                if use_plugin and self.feature_split_module is not None:
                    pred, h = self.model(X, return_feature=True)
                    z_s, z_r = self.feature_split_module(h)
                    loss_cls = criterion(pred, y)
                    loss = loss_cls
                    if self.global_sensitive_feature is not None:
                        target = self.global_sensitive_feature.unsqueeze(0).expand(z_s.size(0), -1)
                        if self.gpu:
                            target = target.cuda()
                        loss_distill = mse_loss(z_s, target)
                        lambda_d = options.get('fedfed_lambda_distill', 1.0)
                        loss = loss_cls + lambda_d * loss_distill
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    with torch.no_grad():
                        z_s_list.append(z_s.detach())
                else:
                    pred = self.model(X)
                    loss = criterion(pred, y)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

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

        aux = None
        if use_plugin and self.feature_split_module is not None and z_s_list:
            with torch.no_grad():
                z_s_all = torch.cat(z_s_list, dim=0)  # (N, sensitive_dim)
                z_s_mean = z_s_all.mean(dim=0)  # (sensitive_dim,)
                clip_norm = options.get('fedfed_clip_norm', 1.0)
                noise_sigma = options.get('fedfed_noise_sigma', 0.1)
                z_s_mean = self._clip_and_noise_z_s(z_s_mean.unsqueeze(0), clip_norm, noise_sigma).squeeze(0)
                aux = {"sensitive_feature": z_s_mean.cpu()}  # upload mean vector only

        return local_model_paras, return_dict, aux
