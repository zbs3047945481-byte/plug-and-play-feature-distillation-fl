# 第五阶段改动说明：最小接入面与模板化交付

本阶段目标是继续压缩插件接入面，让当前项目更接近“可复制到现有联邦学习项目中的模块文件”。

## 1. 协议收缩结果

### 客户端插件

原来的客户端协议包含较多细粒度 hook，例如：

- 设置学习率
- 设置服务端 payload
- train mode
- reset state
- zero grad
- step
- compute loss

现在收缩为 3 个核心动作：

1. `on_round_start(learning_rate, server_payload)`
2. `train_batch(X, y)`
3. `build_upload_payload()`

这意味着外部联邦学习项目在接入时，只需要关心：

- 每轮开始时通知插件
- 每个 batch 调一次插件训练
- 每轮结束时取出插件上传内容

### 服务端插件

原来的服务端协议也做了收缩，变为 2 个核心动作：

1. `build_broadcast_payload()`
2. `aggregate_client_payloads(local_model_paras_set)`

## 2. 代码层改动

### 协议定义

文件：

- `src/plugins/base.py`

客户端和服务端插件协议都已经收缩成更少的 hook。

### FedFed 具体实现

文件：

- `src/plugins/fedfed_plugin.py`

现在 `FedFedClientPlugin` 会在 `train_batch(...)` 内部完成：

- 清梯度
- 前向
- 蒸馏损失计算
- 反向
- 参数更新
- 原型统计

也就是说，训练骨架不再关心插件内部优化细节。

### 骨架调用层

文件：

- `src/fed_client/client.py`
- `src/fed_server/fedbase.py`

现在骨架只需要调用最小化后的几个 hook。

## 3. 新增最小模板

新增文件：

- `src/plugins/minimal_template.py`

用途：

- 给后续移植到其他 FL 项目时提供最小参考骨架
- 作为单文件交付的压缩方向模板

## 4. 当前的工程意义

到这一阶段，项目已经具备：

- 通用插件协议
- 插件注册与工厂
- 最小化后的接入动作
- 一个可运行的具体插件实现
- 一个最小模板文件

这比第四阶段更接近最终毕设交付物，因为：

- 需要改动现有联邦项目的点更少
- 更容易写接入说明
- 更容易向“单文件可插拔”继续收缩

## 5. 下一步建议

如果继续往最终交付推进，下一步最合适的是：

1. 提供一份真正面向外部项目的接入文档
2. 给出“需要修改的 3 个接入点”清单
3. 把 `fedfed_prototype` 再压缩成更单文件化的交付版本
