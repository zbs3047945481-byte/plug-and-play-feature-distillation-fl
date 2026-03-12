#input_options()：通常用于读取/生成超参数配置
from src.options import input_options
from src.utils.tools import get_each_client_data_index, set_random_seed
from getdata import GetDataSet
from src.fed_server.fedavg import FedAvgTrainer


def main():
    options = input_options()#解析命令行参数，生成配置字典
    set_random_seed(options["seed"])
    dataset = GetDataSet(options["dataset_name"])#加载数据集
    #将训练数据分配给多个客户端，返回每个客户端的数据索引列表
    each_client_label_index = get_each_client_data_index(
        dataset.train_label,
        options["num_of_clients"],
        options,
    )
    #创建 FedAvg 训练器，初始化全局模型（如 CNN），创建客户端对象，每个客户端持有部分数据
    FedAvg = FedAvgTrainer(options, dataset, each_client_label_index)
    FedAvg.train()

if __name__ == '__main__':
    main()



