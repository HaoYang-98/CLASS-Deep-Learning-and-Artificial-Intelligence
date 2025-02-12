import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import ndimage

# 数据存储目录
DATA_DIRECTORY = "./data/MNIST"
VALIDATION_SIZE = 5000  # 验证集大小

def extract_data(filename, num):
    """读取未压缩的 MNIST 图像数据，并转换为 PyTorch Tensor"""
    print(f'Extracting {filename}')
    with open(filename, 'rb') as bytestream:
        bytestream.read(16)  # 跳过文件头
        buf = bytestream.read(28 * 28 * num)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num, 1, 28, 28) / 255.0  # 归一化到 [0,1]
    return torch.tensor(data, dtype=torch.float32)

def extract_labels(filename, num):
    """读取未压缩的 MNIST 标签数据，并转换为 PyTorch Tensor"""
    print(f'Extracting {filename}')
    with open(filename, 'rb') as bytestream:
        bytestream.read(8)  # 跳过文件头
        buf = bytestream.read(num)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return torch.tensor(labels, dtype=torch.long)

def augment_data(images, labels):
    """对训练数据进行旋转和偏移增强"""
    augmented_images = []
    augmented_labels = []
    
    for x, y in zip(images, labels):
        augmented_images.append(x)
        augmented_labels.append(y)
        
        for _ in range(4):
            angle = np.random.randint(-90, 90)
            new_img = ndimage.rotate(x.squeeze().numpy(), angle, reshape=False, cval=0)
            new_img = np.expand_dims(new_img, axis=0)  # 添加通道维度
            augmented_images.append(torch.tensor(new_img, dtype=torch.float32))
            augmented_labels.append(y)
    
    return torch.stack(augmented_images), torch.tensor(augmented_labels)

class MNISTDataset(Dataset):
    """用于 PyTorch 的 MNIST 数据集"""
    def __init__(self, images, labels):
        if images.ndim == 3:  # [N, 28, 28]，需要增加通道维度
            self.images = images.unsqueeze(1)  
        else:  # [N, 1, 28, 28]，直接使用
            self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def prepare_MNIST_data(use_data_augmentation=True, expriment=1):
    """准备 MNIST 数据集，解压并进行数据增强"""
    train_data_filename = os.path.join(DATA_DIRECTORY, 'train-images.idx3-ubyte')
    train_labels_filename = os.path.join(DATA_DIRECTORY, 'train-labels.idx1-ubyte')
    test_data_filename = os.path.join(DATA_DIRECTORY, 't10k-images.idx3-ubyte')
    test_labels_filename = os.path.join(DATA_DIRECTORY, 't10k-labels.idx1-ubyte')

    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    # 生成验证集
    validation_data = train_data[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]
    
    # 数据增强
    if use_data_augmentation:
        if expriment == 0:
            raise ValueError("Invalid expriment type. Please choose a value between 1 and 4.")
        if expriment == 1:
            pass
        elif expriment == 2:
            train_data, train_labels = augment_data(train_data, train_labels)
        elif expriment == 3:
            validation_data, validation_labels = augment_data(validation_data, validation_labels)
            test_data, test_labels = augment_data(test_data, test_labels)
        elif expriment == 4:
            train_data, train_labels = augment_data(train_data, train_labels)
            validation_data, validation_labels = augment_data(validation_data, validation_labels)
            test_data, test_labels = augment_data(test_data, test_labels)

    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels
