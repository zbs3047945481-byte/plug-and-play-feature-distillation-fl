
## 1 Install Enviornment

1.1 Create and activate conda virtual environment / 创建 和 激活 conda虚拟环境
```
conda create --name (env_name) python=3.10 # example: conda create --name fl python=3.10
conda activate (env_name) # example: conda activate fl 
```

1.2 install pytorch / 安装pytorch
```
pip3 install torch torchvision torchaudio #  It is recommended to search for the code on the official pytorch website. / 建议官方寻找代码
```
optional:
This code requires tensorboardX to be installed in order to run
You can also disable tensorboardX
```
pip install tensorboardX
```
For experiment visualization, install matplotlib:
```
pip install matplotlib
```
Besides,

2025-03-09 The conda environment is exported as environment.yml


## 2 Quick Start

you can enter the code below to run the federated learning demo. 

```
python main.py
```

After training, the project will automatically save:

- `metrics.json`
- `test_acc_curve.png`
- `test_loss_curve.png`

under the corresponding experiment folder in `result/`.

To compare multiple experiments visually:

```bash
python plot_experiments.py \
  --metrics path/to/exp1/metrics.json path/to/exp2/metrics.json \
  --labels FedAvg FedFed \
  --output_dir result/comparisons
```

## Acknowledgement
This repository needs to thank this paper, i.e, ``Communication-Efficient Learning of Deep Networks from Decentralized Data.''.

## Some final words
en: If this repository has been helpful to you, could you please give it a star? It would be a great honour, and I would very much appreciate it! You are welcome to fork this repository, but please indicate the source in code or others.

