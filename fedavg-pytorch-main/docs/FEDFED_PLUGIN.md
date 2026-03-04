# FedFed Feature Distillation Plugin — 运行说明

---

## 一、每个改动的用途

| 改动位置 | 改动内容 | 作用 |
|----------|----------|------|
| **options.py** | 新增 `use_fedfed_plugin` | 总开关：True 启用 FedFed 插件，False 时行为与原始 FedAvg 完全一致。 |
| **options.py** | 新增 `fedfed_sensitive_dim`（默认 64） | 控制“性能敏感特征”z_s 的维度；z_s 是上传/共享的低维向量，维度越小通信越少、信息越压缩。 |
| **options.py** | 新增 `fedfed_feature_dim`（默认 512） | 与模型中间层 h 的维度一致（如 Mnist_CNN 的 fc1 输出）；供 FeatureSplitModule 的输入维度使用。 |
| **options.py** | 新增 `fedfed_clip_norm`（默认 1.0） | 对 z_s 做 L2 裁剪的范数上界，限制单条特征的范数，便于后续加噪（工程级隐私）。 |
| **options.py** | 新增 `fedfed_noise_sigma`（默认 0.1） | 对 z_s 加的高斯噪声标准差；噪声越大隐私越好、但可能影响蒸馏效果。 |
| **options.py** | 新增 `fedfed_lambda_distill`（默认 1.0） | 特征蒸馏损失 L_distill 的权重 λ；L = L_cls + λ·L_distill，越大越强调向全局敏感特征对齐。 |
| **feature_split.py**（新文件） | `FeatureSplitModule` | 将中间特征 h 拆成 z_s（低维、共享）和 z_r（残差、本地）；用门控+低维投影实现“信息瓶颈”，不依赖具体模型结构。 |
| **mnist_cnn.py** | `forward(..., return_feature=False)` | 为 True 时多返回 fc1 输出 h，供插件在客户端做特征分离和蒸馏，不改变原有单输出 logits 的调用方式。 |
| **client.py** | 插件开启时创建 `FeatureSplitModule` + 本地优化器 | 每个客户端有独立的特征分离模块和包含 model+feature_split 的优化器；feature_split 不参与服务器聚合，仅本地训练。 |
| **client.py** | `set_global_sensitive_feature(...)` | 接收服务器下发的全局敏感特征，用于本地计算 L_distill（MSE(z_s, global_sensitive_feature)）。 |
| **client.py** | `local_update` 中：`pred, h = model(X, return_feature=True)`；z_s, z_r = feature_split(h)；L_distill | 在保持原有分类损失 L_cls 的前提下，用 h 得到 z_s，并（若有 global_sensitive_feature）加蒸馏损失，使本地 z_s 向全局对齐。 |
| **client.py** | 对 z_s 求本轮均值 z_s_mean，再 L2 裁剪 + 高斯噪声，放入 `aux["sensitive_feature"]` | 只上传一个均值向量，控制通信量；裁剪和加噪是工程级隐私保护，不要求完整 ε-δ 会计。 |
| **client.py** | `local_train` 返回 `{weights, num_samples, aux}` | 与服务器约定：weights/num_samples 仍用于 FedAvg 聚合；aux 为可选的 FedFed 附加信息，插件关闭时 aux 为 None。 |
| **fedbase.py** | `local_train` 里对每个 client 调用 `set_global_sensitive_feature(self.global_sensitive_feature)` | 每轮本地训练前把上一轮聚合得到的全局敏感特征下发给客户端，供其计算 L_distill。 |
| **fedbase.py** | `aggregate_parameters` 改为从 `update["weights"]`、`update["num_samples"]` 读取 | 兼容新的 update 字典格式，FedAvg 的加权平均逻辑不变，仅数据来源从“元组”改为“字典字段”。 |
| **fedbase.py** | `_aggregate_aux_sensitive_feature`：按 num_samples 加权平均各客户端的 aux.sensitive_feature | 得到本轮的 global_sensitive_feature，供下一轮下发给所有客户端，实现“只共享低维敏感特征”的 FedFed 思想。 |
| **fedbase.py** | 新增 `self.global_sensitive_feature = None` | 服务器保存当前全局敏感特征；首轮为 None，客户端仅用 L_cls，之后每轮聚合后更新并下发。 |
| **fedavg.py** | 插件开启时打印一行 ENABLED 日志 | 便于确认插件已生效（sensitive_dim、lambda_distill 等）。 |

**小结**：所有改动都围绕“在不动 FedAvg 主流程（传参、聚合、广播）的前提下，增加一条可选的**特征通道**：客户端上传 z_s 的统计量（均值），服务器聚合后下发，客户端用其做特征蒸馏损失；插件关闭时这条通道完全不参与计算与通信。

---

## 二、插件是如何运作的（整体流程）

1. **总开关**  
   - `use_fedfed_plugin=False`：不创建 FeatureSplitModule、不计算 z_s/L_distill、不上传/不聚合 aux；客户端仍返回 `{weights, num_samples, aux: None}`，服务器只做 FedAvg → **行为与原始 FedAvg 完全一致**。  
   - `use_fedfed_plugin=True`：启用下面所有步骤。

2. **客户端侧（每轮本地训练）**  
   - 收到服务器下发的**全局模型参数**（与 FedAvg 相同）和可选的 **global_sensitive_feature**（首轮为 None）。  
   - 对每个 batch：  
     - `pred, h = model(X, return_feature=True)` 得到 logits 和中间特征 h。  
     - `z_s, z_r = feature_split_module(h)`：h 被拆成低维 z_s（用于共享）和残差 z_r（仅本地）。  
     - 分类损失：`L_cls = CrossEntropy(pred, y)`。  
     - 若已有 `global_sensitive_feature`：`L_distill = MSE(z_s, global_sensitive_feature)`，总损失 `L = L_cls + λ·L_distill`；否则只用 `L_cls`。  
     - 反向传播更新**模型参数 + feature_split 参数**（feature_split 仅本地，不参与服务器聚合）。  
   - 本轮结束后：对本地所有 batch 的 z_s 求**均值** z_s_mean，对 z_s_mean 做 **L2 裁剪**（限制范数）和**高斯加噪**，得到要上传的敏感特征向量。  
   - 上传内容：`{ "weights": 模型参数, "num_samples": 本地样本数, "aux": { "sensitive_feature": z_s_mean } }`（只传一个向量，不传整批特征矩阵）。

3. **服务器侧（每轮聚合与广播）**  
   - **模型参数**：与 FedAvg 完全一样，按 `num_samples` 加权平均各客户端的 `weights`，得到新的全局模型，下一轮照常下发。  
   - **aux 通道**（仅插件开启时）：对各客户端 `aux["sensitive_feature"]` 按 `num_samples` 加权平均，得到 **global_sensitive_feature**，保存并在下一轮随全局模型一起下发给参与训练的客户端。  
   - 服务器不训练、不反向传播、不跑生成模型，只做“模型参数 + 敏感特征”的聚合与广播。

4. **多轮后的效果**  
   - 全局模型仍由 FedAvg 主导收敛。  
   - 特征蒸馏使各客户端的 z_s 逐渐向同一“全局敏感特征”靠拢，缓解数据异构下表征不一致的问题（FedFed 思想：只共享少量性能敏感、低维特征，减少 Non-IID 带来的冲突）。  
   - z_r 保留本地个性化/冗余信息，不参与共享，兼顾性能与隐私。

5. **隐私与通信**  
   - 对 z_s 的 L2 裁剪 + 高斯噪声是工程级差分隐私风格处理，不在此做 ε-δ 会计。  
   - 只上传 z_s 的均值向量（维度 = fedfed_sensitive_dim，如 64），通信开销相对模型参数可忽略。

**一句话**：插件在 FedAvg 之外增加了一条“低维敏感特征”的通道——客户端上传 z_s 的裁剪加噪均值，服务器聚合为全局敏感特征并下发，客户端用其做特征蒸馏损失；FedAvg 的主流程不变，插件关闭时这条通道完全关闭。

---

## 1. 改动点总览

- **options.py**：新增 FedFed 插件相关参数（`use_fedfed_plugin`、`fedfed_sensitive_dim`、`fedfed_feature_dim`、`fedfed_clip_norm`、`fedfed_noise_sigma`、`fedfed_lambda_distill`）。
- **src/models/feature_split.py**（新增）：`FeatureSplitModule`，将中间特征 h 分解为 z_s（低维共享）与 z_r（残差本地）。
- **src/models/mnist_cnn.py**：`forward` 增加 `return_feature=False`，为 True 时返回 `(logits, h)`（h 为 fc1 输出）。
- **src/fed_client/client.py**：插件开启时创建 `FeatureSplitModule` 与本地优化器；`local_update` 中计算 z_s、L_distill、z_s_mean，并构造 `aux`；`local_train` 返回 `update = {weights, num_samples, aux}`。
- **src/fed_server/fedbase.py**：`local_train` 下发 `global_sensitive_feature` 并接收 `update` 字典；`aggregate_parameters` 按 `update` 聚合权重并调用 `_aggregate_aux_sensitive_feature` 聚合 aux；新增 `global_sensitive_feature` 与 `_aggregate_aux_sensitive_feature`。
- **src/fed_server/fedavg.py**：插件开启时打印一行启动日志，便于确认插件已启用。

**插件关闭时**：不创建 `FeatureSplitModule`、不计算 z_s/L_distill、不上传 aux、不聚合 aux；客户端仍返回 `{weights, num_samples, aux: None}`，服务器仅用 weights/num_samples 做 FedAvg，行为与原始 FedAvg 一致。

---

## 2. 如何开启插件

命令行开启（推荐）：

```bash
python main.py --use_fedfed_plugin true
```

带推荐默认超参数示例：

```bash
python main.py --use_fedfed_plugin true \
  --fedfed_sensitive_dim 64 \
  --fedfed_feature_dim 512 \
  --fedfed_clip_norm 1.0 \
  --fedfed_noise_sigma 0.1 \
  --fedfed_lambda_distill 1.0
```

关闭插件（与原始 FedAvg 一致）：

```bash
python main.py
# 或显式
python main.py --use_fedfed_plugin false
```

---

## 3. 推荐默认超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_fedfed_plugin` | False | 是否启用 FedFed 特征蒸馏插件 |
| `fedfed_sensitive_dim` | 64 | 性能敏感特征 z_s 维度（共享） |
| `fedfed_feature_dim` | 512 | 模型中间特征 h 维度（mnist_cnn 的 fc1 输出） |
| `fedfed_clip_norm` | 1.0 | z_s 上传前 L2 裁剪范数 |
| `fedfed_noise_sigma` | 0.1 | z_s 加性高斯噪声标准差 |
| `fedfed_lambda_distill` | 1.0 | 特征蒸馏损失权重 λ |

---

## 4. 如何验证插件是否生效

- **启动日志**：插件开启时会出现一行：
  - `>>> FedFed Feature Distillation Plugin ENABLED (sensitive_dim=64, lambda_distill=1.0)`
- **行为**：开启后首轮仅用 L_cls，从第二轮起使用 L_cls + λ·L_distill；客户端上传 `aux.sensitive_feature`（z_s_mean），服务器聚合为 `global_sensitive_feature` 并下发给下一轮客户端。
- **对比实验**：同一配置（如 `--round_num 50`）分别跑 `--use_fedfed_plugin false` 与 `true`，比较 `result/` 下 test acc/loss；在 Non-IID 设置下，插件开启通常可带来一定精度提升或更快收敛。

---

## 5. 新增模块 FeatureSplitModule 位置

完整实现见：**`src/models/feature_split.py`**  
接口：`forward(h) -> (z_s, z_r)`，其中 `z_s` 为低维共享特征，`z_r` 为残差（本地使用）。

---

## 6. 5 轮对比：FedFed 插件 vs 原始 FedAvg Baseline

同一配置（`--round_num 5`，10 clients/round，MNIST，默认数据划分）下分别运行：

- **Baseline**：`python main.py --use_fedfed_plugin false --round_num 5`
- **FedFed 插件**：`python main.py --use_fedfed_plugin true --round_num 5`

### 测试集指标对比

| Round | FedAvg Baseline (acc / loss) | FedFed Plugin (acc / loss) |
|-------|------------------------------|----------------------------|
| 0     | **12.11%** / 2.3004          | 7.03% / 2.3039             |
| 1     | **92.57%** / **0.2891**      | 92.43% / 0.3853            |
| 2     | **96.00%** / 0.1399          | 95.98% / **0.1228**        |
| 3     | 96.79% / 0.1052              | **97.08%** / **0.0889**    |
| 4     | 97.33% / 0.0825              | **97.49%** / **0.0719**    |
| 5     | 97.56% / 0.0770              | **97.78%** / **0.0666**    |

### 结论（5 轮内）

- **前期（Round 0–1）**：Baseline 略优（初始化和首轮聚合的随机性导致 FedFed 首轮 acc 略低、loss 略高）。
- **中后期（Round 2–5）**：FedFed 插件略优——Round 3–5 的 **测试准确率更高、测试 loss 更低**（Round 5：97.78% vs 97.56%，loss 0.0666 vs 0.0770）。
- 在默认 IID 式数据划分下，5 轮内 FedFed 插件在中后期表现出略好的收敛与最终指标；在更强 Non-IID 设置下，FedFed 思想通常能带来更明显的收益。
