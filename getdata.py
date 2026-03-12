import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np
import os
import gzip


class GetDataSet():
    def __init__(self, dataset_name):#self = 当前这个 GetDataSet 实例。
        self.dataset_name = str(dataset_name).lower()

        self.train_data = None
        self.train_label = None
        self.train_data_size = None

        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        #只支持 MNIST（或字符串 mnist）
        if self.dataset_name.startswith('mnist'):
            self.mnistDataDistribution()
        else:
            raise ValueError('Unsupported dataset: {}'.format(dataset_name))

    def mnistDataDistribution(self, ):

        #说明：这个仓库并没有用 torchvision.datasets.MNIST，而是直接读原始 gz 文件
        #所以你必须确保 ./data/MNIST/raw 下已经有这四个文件（通常是手动下载或用脚本下载）
        data_dir = r'./data/MNIST/raw'
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        #把目录和文件名用当前操作系统的路径分隔符连起来，得到完整路径。
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = self.extract_images(train_images_path)
        #用当前对象自己的方法，根据“训练集图像文件路径”把 .gz 里的图像读出来，得到一张大数组，并赋给 train_images。
        # print(train_images.shape) # 图片的形状 (60000, 28, 28, 1) 60000张 28 * 28 * 1  灰色一个通道
        # print('-' * 22 + "\n")
        train_labels = self.extract_labels(train_labels_path)
        # print("-" * 5 + "train_labels" + "-" * 5)
        # print(train_labels.shape)  # label shape (60000, 10)
        # print('-' * 22 + "\n")
        test_images = self.extract_images(test_images_path)
        test_labels = self.extract_labels(test_labels_path)


        # assert train_images.shape[0] == train_labels.shape[0]
        # assert test_images.shape[0] == test_labels.shape[0]
        #
        #
        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]
        #
        # assert train_images.shape[3] == 1
        # assert test_images.shape[3] == 1
        train_images = train_images.reshape(train_images.shape[0], 1, train_images.shape[1], train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], 1, test_images.shape[1], test_images.shape[2])
#把训练图像从 (60000, 28, 28, 1) 变成 (60000, 1, 28, 28)，即从 NHWC 改成 NCHW，以符合 PyTorch 的约定。
        train_images = train_images.astype(np.float32)
        # 数组对应元素位置相乘
        train_images = np.multiply(train_images, 1.0 / 255.0)
        # print(train_images[0:10,5:10])
        #把像素从 [0,255] → [0,1]
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)
        #因为 one-hot 里只有一个位置是 1。
        self.train_data = train_images
        self.train_label = np.argmax(train_labels == 1, axis = 1)
        self.test_data = test_images
        self.test_label = np.argmax(test_labels == 1, axis = 1)
        print(self.train_data.shape)


    #读 MNIST 官方二进制格式
    def extract_images(self, filename):
        """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2051:
                raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
            num_images = self._read32(bytestream)
            rows = self._read32(bytestream)
            cols = self._read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols, 1)
            return data

    def _read32(self, bytestream):
        dt = np.dtype(np.uint32).newbyteorder('>')

        return np.frombuffer(bytestream.read(4), dtype=dt)[0]

#extract_labels 做的事情：按 MNIST 标签格式解析 .gz 文件 → 得到一维标签数组 → 再转成 one-hot 并返回。
    def extract_labels(self, filename):
        """Extract the labels into a 1D uint8 numpy array [index]."""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            magic = self._read32(bytestream)
            if magic != 2049:
                raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
            num_items = self._read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            return self.dense_to_one_hot(labels)

    def dense_to_one_hot(self, labels_dense, num_classes=10):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
