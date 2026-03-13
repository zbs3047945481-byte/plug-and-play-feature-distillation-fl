# 第四阶段改动说明：统一插件协议与注册机制

本阶段目标是让训练骨架不再依赖具体插件名字，而是依赖统一的插件协议。

## 1. 新增的抽象层

新增文件：

- `src/plugins/base.py`

定义了两个基础协议：

- `BaseClientPlugin`
- `BaseServerPlugin`

### `BaseClientPlugin` 规定的能力

- 设置学习率
- 接收服务端 payload
- 切换训练模式
- 重置轮次状态
- 清梯度
- 执行一步优化
- 计算带插件逻辑的 loss
- 构造上传 payload

### `BaseServerPlugin` 规定的能力

- 构造下发给客户端的 payload
- 从客户端更新中聚合插件辅助信息

## 2. 新增的注册与工厂层

新增文件：

- `src/plugins/__init__.py`

这里做了三件事：

1. 定义插件注册表 `PLUGIN_REGISTRY`
2. 提供 `resolve_plugin_name(options)`
3. 提供：
   - `build_client_plugin(...)`
   - `build_server_plugin(...)`

当前注册的插件是：

- `fedfed_prototype`

## 3. 现有 FedFed 插件如何接入协议

文件：

- `src/plugins/fedfed_plugin.py`

现在：

- `FedFedClientPlugin` 继承 `BaseClientPlugin`
- `FedFedServerPlugin` 继承 `BaseServerPlugin`

也就是说，FedFed 已经从“直接被训练骨架引用的实现”变成“协议下的一个具体插件实现”。

## 4. 训练骨架如何变化

### 客户端

文件：

- `src/fed_client/client.py`

现在客户端不再直接 import `FedFedClientPlugin`，而是通过：

- `build_client_plugin(options, model, gpu)`

来获得插件实例。

这意味着：

- 训练骨架依赖的是插件协议
- 而不是某个具体插件类名

### 服务端

文件：

- `src/fed_server/fedbase.py`

现在服务端不再直接 import `FedFedServerPlugin`，而是通过：

- `build_server_plugin(options, gpu)`

来获得插件实例。

### 日志与配置入口

文件：

- `src/fed_server/fedavg.py`
- `src/options.py`

新增：

- `--plugin_name`

当前可选：

- `none`
- `fedfed_prototype`

同时保留：

- `--use_fedfed_plugin`

作为向后兼容入口。

## 5. 这一步的意义

到这一阶段，项目已经从：

- “在联邦学习代码中嵌一个算法”

走到了：

- “训练骨架 + 插件协议 + 一个具体插件实现”

这比第三阶段更接近最终毕设目标，因为后续你要做的已经不再是“大改骨架”，而是：

- 压缩协议
- 固化接入点
- 输出更轻的模块文件

## 6. 下一阶段更适合做什么

如果继续朝“最终可复制到现有 FL 项目中的模块文件”推进，下一步建议：

1. 把插件协议再收缩到更少的 hook
2. 提供一份最小接入模板
3. 尝试把 `fedfed_prototype` 收敛成单文件插件版本
