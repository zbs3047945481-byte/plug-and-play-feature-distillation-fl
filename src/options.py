#定义并解析联邦学习实验的所有超参数，把它们整理成一个 dict（options），供整个系统使用。
import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


#argparse 是 Python 的命令行参数解析库
#它允许你定义命令行参数，并自动解析这些参数，将它们转换为 Python 字典。
def input_options():
    parser = argparse.ArgumentParser()
    #•	创建一个“参数解析器”
	#•	后面所有 add_argument 都是在往这个解析器里注册参数
    # iid
    parser.add_argument('-is_iid', type=str2bool, default=False, help='data distribution is iid.')
    #是否使用 IID 数据分布
    
    parser.add_argument('--dataset_name', type=str, default='mnist', help='name of dataset.')
    #数据集名称 / 数据划分方式标识
    #mnist_dir_0.1 很可能表示：MNIST、Dirichlet α=0.1（Non-IID 程度）

    parser.add_argument('--model_name', type=str, default='mnist_cnn', help='the model to train')
    #指定用哪个模型
    #if model_name == 'mnist_cnn': model = MnistCNN()

    parser.add_argument('--gpu', type=str2bool, default=True, help='gpu id to use')
    #是否使用 GPU
    
    parser.add_argument('--round_num', type=int, default=301, help='number of round in comm')
    #通信轮数：每一轮 = 一次 FedAvg 聚合
    
    parser.add_argument('--num_of_clients', type=int, default=100, help='numer of the clients')
    #联邦系统中的客户端总数 K
    
    parser.add_argument('--c_fraction', type=float, default=0.1,
                        help='C fraction, 0 means 1 client, 1 means total clients')
    #每一轮参与训练的客户端比例
    
    parser.add_argument('--local_epoch', type=int, default=5, help='local train epoch')
    #每个客户端本地训练多少个 epoch
    #比如有1000条数据，把这1000个数据全部用来算一次梯度并更新参数，这完整一轮叫一个epoch

    parser.add_argument('--batch_size', type=int, default=32, help='local train batch size')
    #本地训练 batch size
    #每次梯度更新用多少条样本
    #比如有1000条数据，每轮训练只取100条数据来算梯度并更新参数，这100条数据叫一个batch

    parser.add_argument("--lr", type=float, default=0.001, help="learning rate, \
                        use value from origin paper as default")
    #本地学习率
    
    parser.add_argument('--gn0', type=int, default=1, help='gno')
    
    parser.add_argument('--seed', help='seed for randomness;', type=int, default=3001)
    #随机种子：控制数据划分、模型初始化，方便复现实验
    
    parser.add_argument('--weight_decay', help='weight_decay;', type=int, default=1)
    #权重衰减：防止过拟合

    # ---------- Heterogeneity modeling ----------
    parser.add_argument('--partition_strategy', type=str, default='dirichlet',
                        choices=['iid', 'dirichlet'],
                        help='Client data partition strategy.')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.3,
                        help='Dirichlet alpha for label skew. Smaller means stronger heterogeneity.')
    parser.add_argument('--min_samples_per_client', type=int, default=32,
                        help='Ensure each client owns at least this many training samples.')
    parser.add_argument('--enable_quantity_skew', type=str2bool, default=True,
                        help='Whether to vary client dataset sizes.')
    parser.add_argument('--quantity_skew_beta', type=float, default=0.5,
                        help='Dirichlet beta for client quantity skew. Smaller means more imbalance.')
    parser.add_argument('--enable_feature_skew', type=str2bool, default=True,
                        help='Whether to apply client-specific feature shift/noise.')
    parser.add_argument('--feature_noise_std', type=float, default=0.05,
                        help='Std of client-specific additive Gaussian noise.')
    parser.add_argument('--feature_scale_low', type=float, default=0.85,
                        help='Lower bound of client-specific multiplicative scale.')
    parser.add_argument('--feature_scale_high', type=float, default=1.15,
                        help='Upper bound of client-specific multiplicative scale.')
    parser.add_argument('--feature_bias_std', type=float, default=0.05,
                        help='Std of client-specific additive bias.')

    # ---------- FedFed Feature Distillation Plugin (optional) ----------
    parser.add_argument('--use_fedfed_plugin', type=str2bool, default=False,
                        help='Enable FedFed-style feature distillation plugin.')
    parser.add_argument('--fedfed_sensitive_dim', type=int, default=64,
                        help='Dimension of performance-sensitive feature z_s (shared).')
    parser.add_argument('--fedfed_feature_dim', type=int, default=512,
                        help='Dimension of model intermediate feature h (e.g. mnist_cnn fc1 output).')
    parser.add_argument('--fedfed_clip_norm', type=float, default=1.0,
                        help='L2 clip norm for z_s before adding noise (privacy).')
    parser.add_argument('--fedfed_noise_sigma', type=float, default=0.1,
                        help='Gaussian noise std for z_s (privacy).')
    parser.add_argument('--fedfed_lambda_distill', type=float, default=1.0,
                        help='Weight of feature distillation loss L_distill.')
    
    args = parser.parse_args()
    #从命令行读取参数
    
    options = args.__dict__
    #把参数对象转成字典
    if options['is_iid']:
        options['partition_strategy'] = 'iid'

    return options

#Batch size 决定“每次怎么学”，
#Epoch 决定“学多久”，
#Seed 决定“能不能复现”。
