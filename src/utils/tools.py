import random

import numpy as np
import torch


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_by_counts(indices, counts):
    assignments = []
    start = 0
    for count in counts:
        end = start + count
        assignments.append(indices[start:end].tolist())
        start = end
    return assignments


def _ensure_min_samples(assignments, min_samples):
    if min_samples <= 0:
        return assignments

    lengths = [len(items) for items in assignments]
    for client_id, current_len in enumerate(lengths):
        while current_len < min_samples:
            donor_id = int(np.argmax(lengths))
            if donor_id == client_id or lengths[donor_id] <= min_samples:
                break
            assignments[client_id].append(assignments[donor_id].pop())
            lengths[client_id] += 1
            lengths[donor_id] -= 1
            current_len += 1
    return assignments


def _sample_client_capacities(client_num, total_num, beta, min_samples):
    if client_num * min_samples > total_num:
        raise ValueError('min_samples_per_client is too large for the current dataset size.')

    base = np.full(client_num, min_samples, dtype=int)
    remaining = total_num - base.sum()
    if remaining <= 0:
        return base

    weights = np.random.dirichlet(np.full(client_num, beta))
    extra = np.random.multinomial(remaining, weights)
    return base + extra


def _build_iid_partition(train_labels, client_num, min_samples, enable_quantity_skew, quantity_skew_beta):
    shuffled_indices = np.random.permutation(len(train_labels))
    if enable_quantity_skew:
        counts = _sample_client_capacities(client_num, len(train_labels), quantity_skew_beta, min_samples)
    else:
        counts = np.full(client_num, len(train_labels) // client_num, dtype=int)
        counts[:len(train_labels) % client_num] += 1
    return _split_by_counts(shuffled_indices, counts)


def _build_dirichlet_partition(train_labels, client_num, alpha, min_samples, enable_quantity_skew, quantity_skew_beta):
    labels = np.asarray(train_labels)
    assignments = [[] for _ in range(client_num)]
    client_activity = np.ones(client_num, dtype=float)
    if enable_quantity_skew:
        client_activity = np.random.dirichlet(np.full(client_num, quantity_skew_beta))

    for cls in np.unique(labels):
        class_indices = np.where(labels == cls)[0]
        np.random.shuffle(class_indices)
        class_weights = np.random.dirichlet(np.full(client_num, alpha))
        class_weights = class_weights * client_activity
        class_weights = class_weights / class_weights.sum()
        split_points = (np.cumsum(class_weights) * len(class_indices)).astype(int)[:-1]
        splits = np.split(class_indices, split_points)
        for client_id, split in enumerate(splits):
            assignments[client_id].extend(split.tolist())

    return _ensure_min_samples(assignments, min_samples)


def get_each_client_data_index(train_labels, client_num, options=None):
    options = options or {}
    strategy = options.get('partition_strategy', 'dirichlet')
    min_samples = options.get('min_samples_per_client', 0)
    enable_quantity_skew = options.get('enable_quantity_skew', False)
    quantity_skew_beta = options.get('quantity_skew_beta', 1.0)

    if strategy == 'iid':
        return _build_iid_partition(
            train_labels,
            client_num,
            min_samples,
            enable_quantity_skew,
            quantity_skew_beta,
        )

    if strategy == 'dirichlet':
        return _build_dirichlet_partition(
            train_labels,
            client_num,
            options.get('dirichlet_alpha', 0.3),
            min_samples,
            enable_quantity_skew,
            quantity_skew_beta,
        )

    raise ValueError('Unsupported partition strategy: {}'.format(strategy))


def build_client_feature_skews(client_num, options):
    if not options.get('enable_feature_skew', False):
        return [None] * client_num

    low = options.get('feature_scale_low', 1.0)
    high = options.get('feature_scale_high', 1.0)
    bias_std = options.get('feature_bias_std', 0.0)
    noise_std = options.get('feature_noise_std', 0.0)

    skews = []
    for _ in range(client_num):
        skews.append({
            'scale': float(np.random.uniform(low, high)),
            'bias': float(np.random.normal(0.0, bias_std)),
            'noise_std': float(max(noise_std, 0.0)),
        })
    return skews


def apply_feature_skew(data, skew):
    if skew is None:
        return data.copy()

    transformed = data.astype(np.float32).copy()
    transformed = transformed * skew['scale'] + skew['bias']
    if skew['noise_std'] > 0:
        transformed = transformed + np.random.normal(
            0.0,
            skew['noise_std'],
            size=transformed.shape,
        ).astype(np.float32)
    return np.clip(transformed, 0.0, 1.0)
