# 第一阶段工程化改动说明

本阶段目标是先把当前项目从“能跑的原型”修到“可信的实验基础设施”。

## 1. 本阶段完成的内容

### 1.1 数据异构建模

- 新增可配置的数据划分策略：
  - `iid`
  - `dirichlet`
- 新增数量异构（quantity skew）支持：
  - 通过 `quantity_skew_beta` 控制不同客户端样本量不均衡程度
- 新增特征异构（feature skew）支持：
  - 每个客户端可注入独立的缩放、偏置和高斯噪声
- 新增最小样本数约束：
  - 避免 Dirichlet 极端情况下个别客户端样本过少

对应文件：
- `src/utils/tools.py`
- `src/options.py`
- `main.py`

### 1.2 客户端隔离修复

修复了原来所有客户端共享同一个模型实例和优化器实例的问题。

现在的行为是：
- 服务器维护自己的全局模型
- 每个客户端在初始化时创建自己的模型副本
- 每个客户端拥有自己的本地优化器
- 每轮训练前由服务器下发最新全局参数到客户端本地模型
- 学习率由服务器同步给各客户端

对应文件：
- `src/fed_server/fedavg.py`
- `src/fed_server/fedbase.py`
- `src/fed_client/client.py`

### 1.3 数据入口与可复现性

- `main.py` 不再硬编码 `MNIST`
- 数据集入口改为读取 `options["dataset_name"]`
- 新增随机种子统一设置
- `getdata.py` 去掉 import 时自动加载数据的副作用
- `getdata.py` 支持 `mnist` 及其别名形式（如 `mnist_dir_0.1`）

对应文件：
- `main.py`
- `getdata.py`
- `src/utils/tools.py`

### 1.4 配置能力增强

新增参数：

- `partition_strategy`
- `dirichlet_alpha`
- `min_samples_per_client`
- `enable_quantity_skew`
- `quantity_skew_beta`
- `enable_feature_skew`
- `feature_noise_std`
- `feature_scale_low`
- `feature_scale_high`
- `feature_bias_std`

同时修复了布尔参数解析，避免 `argparse` 里 `type=bool` 的常见误判。

对应文件：
- `src/options.py`

## 2. 各文件改动摘要

### `main.py`

- 增加随机种子初始化
- 数据集改为由配置驱动
- 客户端数据划分改为传入完整 `options`

### `src/options.py`

- 新增 `str2bool`
- 加入第一阶段异构建模配置
- 让 `is_iid` 与真实划分策略联动

### `src/utils/tools.py`

- 新增 `set_random_seed`
- 重写 `get_each_client_data_index`
- 新增：
  - `build_client_feature_skews`
  - `apply_feature_skew`

### `getdata.py`

- 支持 `mnist` 系列别名
- 删除 import 时自动执行的数据加载代码

### `src/models/models.py`

- 增加未知模型时报错逻辑

### `src/fed_client/client.py`

- 新增 `set_learning_rate`
- 本地训练前显式 `zero_grad`
- 为独立客户端优化器同步学习率

### `src/fed_server/fedbase.py`

- 增加 `model_builder` 和 `optimizer_builder`
- 每个客户端初始化独立模型和独立优化器
- 客户端本地数据可叠加 feature skew
- 测试集评估改为使用配置的 `batch_size`
- 去掉测试阶段无意义的 `testLabel` 输出

### `src/fed_server/fedavg.py`

- 服务器端引入模型/优化器构造器
- 客户端不再复用服务器模型与优化器实例

## 3. 建议的运行方式

### 3.1 轻量 Non-IID 场景

```bash
python main.py \
  --dataset_name mnist \
  --partition_strategy dirichlet \
  --dirichlet_alpha 0.3 \
  --enable_quantity_skew true \
  --quantity_skew_beta 0.5 \
  --enable_feature_skew true \
  --feature_noise_std 0.05 \
  --feature_scale_low 0.85 \
  --feature_scale_high 1.15 \
  --feature_bias_std 0.05
```

### 3.2 插件联动测试

```bash
python main.py \
  --dataset_name mnist \
  --partition_strategy dirichlet \
  --dirichlet_alpha 0.3 \
  --enable_quantity_skew true \
  --enable_feature_skew true \
  --use_fedfed_plugin true
```

## 4. 本阶段验证结果

已完成两类验证：

- 静态验证：
  - `python -m compileall main.py src getdata.py`
- 最小训练验证：
  - 在 `round_num=1`、`num_of_clients=5`、`use_fedfed_plugin=true` 下成功完成训练和测试

## 5. 本阶段尚未解决的问题

这些不属于第一阶段范围，但必须进入下一阶段：

- 当前蒸馏目标仍然是“全局单均值敏感特征”，信息粒度偏粗
- `z_r` 仍未真正进入本地任务路径
- 还没有统一的 baseline runner（如 `FedProx`、`SCAFFOLD`）
- 还没有批量实验脚本、画图脚本和结果汇总脚本
- 还没有 server/client 进程化部署结构

## 6. 下一阶段建议

第二阶段建议优先做两件事：

1. 将“全局单均值敏感特征”升级为“类别原型蒸馏”
2. 将当前单机模拟框架拆出更清晰的算法接口和插件接口
