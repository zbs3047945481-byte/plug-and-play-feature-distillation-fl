import numpy as np




def get_each_client_data_index(train_labels, client_num):
    #train_labels：训练集的所有标签（长度 = 训练样本数）
    np.random.seed(42)
    train_labels_num = len(train_labels)
    #训练样本总数
    
    each_client_label_num = train_labels_num / client_num
    #每个客户端理论上分到多少样本
    
    each_client_label_index = [[] for _ in range(client_num)]
    #创建一个长度为 client_num 的列表
    #每个元素都是一个空列表，用来存储每个客户端的样本索引
    
    current_index = 0
    #用来记录当前切分到数据集的哪一个位置
    
    for i in range(client_num):
        end_index = int((i + 1) * each_client_label_num)
        #计算当前客户端的结束索引
        
        client_indices = list(range(current_index, end_index))
        #生成当前客户端的样本索引列表
        
        each_client_label_index[i] = client_indices
        current_index = end_index

    return each_client_label_index


