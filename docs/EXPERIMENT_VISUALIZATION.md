# 实验结果可视化说明

当前项目已经支持两类实验图表生成：

## 1. 单次实验自动出图

每次训练结束后，程序会在对应实验目录下自动生成：

- `metrics.json`
- `test_acc_curve.png`
- `test_loss_curve.png`

这些文件默认位于：

```text
result/<dataset_name>/<exp_name>/
```

其中：

- `metrics.json` 保存完整指标
- `test_acc_curve.png` 保存测试精度曲线
- `test_loss_curve.png` 保存测试损失曲线

## 2. 多实验对比图

项目新增了：

- `plot_experiments.py`

它可以读取多个 `metrics.json`，生成：

- 测试精度对比曲线
- 测试损失对比曲线
- 最佳精度柱状图
- 最终精度柱状图

示例：

```bash
python plot_experiments.py \
  --metrics \
    result/mnist/exp_a/metrics.json \
    result/mnist/exp_b/metrics.json \
  --labels \
    FedAvg \
    FedFed \
  --output_dir result/comparisons
```

## 3. 适合论文的常见对比组合

你后续可以按这些组合直接出图：

- `FedAvg` vs `fedfed_prototype`
- `fedfed_prototype` vs `fedfed_single_file`
- 不同 `dirichlet_alpha`
- 不同 `fedfed_lambda_distill`
- 不同 `fedfed_sensitive_dim`
- 不同 `fedfed_noise_sigma`

## 4. 当前可视化相关代码位置

- 自动单次实验出图：
  - `src/utils/metrics.py`
  - `src/utils/plotting.py`

- 多实验对比图：
  - `plot_experiments.py`

## 5. 依赖

绘图依赖：

- `matplotlib`

如果环境中没有安装，训练仍可进行，但图表不会生成。
