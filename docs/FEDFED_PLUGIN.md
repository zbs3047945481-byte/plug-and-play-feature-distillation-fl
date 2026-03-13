# FedFed Feature Distillation Plugin

当前版本已经从“单均值敏感特征蒸馏”升级为“类原型蒸馏”。

## 1. 当前插件做了什么

在不改变 FedAvg 主流程的前提下，插件增加了一条低维特征通道：

- 客户端从中间特征 `h` 中提取低维敏感特征 `z_s`
- 按类别统计本地 prototype
- 对 prototype 做裁剪和加噪
- 上传类原型到服务器
- 服务器按类别聚合为全局 prototype bank
- 客户端在下一轮本地训练中，用全局 prototype bank 计算蒸馏损失

插件关闭时，行为与原始 FedAvg 一致。

## 2. 关键参数

- `--use_fedfed_plugin true|false`
- `--plugin_name none|fedfed_prototype`
- `--fedfed_sensitive_dim`
- `--fedfed_feature_dim`
- `--fedfed_clip_norm`
- `--fedfed_noise_sigma`
- `--fedfed_lambda_distill`

## 3. 当前工作流

### 客户端

每个 batch：

1. `pred, h = model(X, return_feature=True)`
2. `z_s, z_r = feature_split_module(h)`
3. 计算分类损失 `L_cls`
4. 如果服务器已经下发全局 prototype bank：
   - 对当前 batch 内出现的类别计算本地 prototype
   - 与对应的全局 prototype 做 MSE
   - 得到蒸馏损失 `L_distill`
5. 总损失：
   - `L = L_cls + lambda * L_distill`

每轮结束后：

- 统计本地每个类别的 prototype
- 对每个 prototype 单独裁剪和加噪
- 上传：

```python
aux = {
    "prototypes": {
        class_id: {
            "prototype": tensor,
            "count": int,
        }
    }
}
```

### 服务器

每轮聚合：

1. 正常做 FedAvg 参数聚合
2. 对客户端上传的 `aux["prototypes"]` 按类别聚合
3. 按 `count` 加权平均，形成：

```python
global_prototypes = {
    class_id: prototype_tensor
}
```

4. 下一轮下发给客户端

## 4. 为什么用类原型而不是单均值

单均值方案的问题：

- 把所有类别压成一个向量，信息过粗
- 对分类任务不够自然
- 与真实接入场景偏差较大

类原型方案的优势：

- 更贴近真实分类任务
- 保留类别判别结构
- 通信量仍然很小
- 更适合作为现有联邦学习项目中的插件模块

## 5. 如何运行

示例：

```bash
python main.py \
  --dataset_name mnist \
  --plugin_name fedfed_prototype \
  --partition_strategy dirichlet \
  --dirichlet_alpha 0.3 \
  --enable_quantity_skew true \
  --enable_feature_skew true \
  --fedfed_sensitive_dim 64 \
  --fedfed_feature_dim 512 \
  --fedfed_clip_norm 1.0 \
  --fedfed_noise_sigma 0.1 \
  --fedfed_lambda_distill 1.0
```

## 6. 当前实现边界

当前实现已经比第一阶段更贴近真实任务，但仍然是模拟验证环境：

- 仍为单机模拟
- 仍未做网络化 server/client 拆分
- 仍未做 secure aggregation
- 当前为“每类一个 prototype”
- 尚未升级到“每类多个簇中心”

## 7. 下一步

如果继续沿“可插拔模块”路线推进，下一步最合适的是：

1. 把 prototype 蒸馏逻辑抽成独立插件接口
2. 视任务需求升级为每类多个簇中心
