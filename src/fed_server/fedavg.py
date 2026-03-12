#FedAvgTrainer=联邦服务器（Server）的控制器，负责：
#选客户端 → 下发模型 → 收回本地模型 → 聚合 → 测试 → 调学习率
#它不是具体训练模型的地方，而是“指挥谁训练、什么时候聚合”的地方


from src.fed_server.fedbase import BaseFederated
from torch import optim
from src.optimizers.adam import MyAdam
import numpy as np
from src.models.models import choose_model

class FedAvgTrainer(BaseFederated):
    #options：实验配置（超参数）
    #dataset：全局数据对象（train/test）
    #clients_label：每个client的数据索引
    def __init__(self, options, dataset, clients_label):
        #根据 options['model_name'] 选择模型
        model = choose_model(options)
        model_builder = lambda: choose_model(options)
        self.move_model_to_gpu(model, options)

        #使用自定义Adam，本质等价于 torch.optim.Adam，这个优化器会被传给客户端用于本地训练
        optimizer_builder = lambda params: MyAdam(params, lr=options['lr'])
        self.optimizer = optimizer_builder(model.parameters())
        super(FedAvgTrainer, self).__init__(
            options,
            dataset,
            clients_label,
            model,
            self.optimizer,
            model_builder=model_builder,
            optimizer_builder=optimizer_builder,
        )
        if options.get('use_fedfed_plugin', False):
            print('>>> FedFed Feature Distillation Plugin ENABLED (sensitive_dim={}, lambda_distill={})'.format(
                options.get('fedfed_sensitive_dim', 64), options.get('fedfed_lambda_distill', 1.0)))

    def train(self):
        print('>>> Select {} clients per round \n'.format(int(self.per_round_c_fraction * self.clients_num)))

        # self.latest_global_model = self.get_model_parameters()
        #self.num_round=通信轮数
        #每一轮=一次完整的：下发模型、本地训练、聚合
        for round_i in range(self.num_round):
            #用当前全局模型、在测试集上评估、记录accuracy/loss
            self.test_latest_model_on_testdata(round_i)
            #客户端抽样
            selected_clients = self.select_clients()

            #对每个client：1.下发全局模型参数；2.在client本地数据上训练local_epoch;3.收集训练后的模型参数
            #返回所有client的模型参数（通常是state_dict）和本地loss、样本数等统计信息
            local_model_paras_set, stats = self.local_train(round_i, selected_clients)

            #对所有客户端模型参数按样本数加权平均
            #得到新的全局模型参数
            self.latest_global_model = self.aggregate_parameters(local_model_paras_set)

            #根据轮数调整学习率；不是FedAvg必需步骤；但有助于收敛稳定
            self.optimizer.adjust_learning_rate(round_i)

        #最后一轮测试，把所有指标写入文件
        self.test_latest_model_on_testdata(self.num_round)
        self.metrics.write()

    #非常标准的FedAvg实现
    def select_clients(self):
        #计算本轮客户端数量，防止抽样数量大于总客户端数
        num_clients = min(int(self.per_round_c_fraction * self.clients_num), self.clients_num)
        #随机选client索引，replace=False：不重复抽样，每轮client是随机的
        index = np.random.choice(len(self.clients), num_clients, replace=False,)
        #根据索引拿到client对象
        #self.clients 是在 BaseFederated 中初始化的
        #每个元素是一个 client 实例
        select_clients = []
        for i in index:
            select_clients.append(self.clients[i])
        #返回client列表
        return select_clients
