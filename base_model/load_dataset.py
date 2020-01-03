import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader

"""加载minist数据集
    input：data_dir 数据集下载存放的路径
            config 是超参数配置信息
    return: train_loader,test_loader, 训练和测试的torch_dataloader
"""
def load_minist(config):
    if not os.path.exists(config.data_dir):
        os.makedirs(config.data_dir)

    train_dataset = torchvision.datasets.MNIST(
        root=config.data_dir,train=True,transform = transforms.ToTensor(),download=True)
    test_dataset = torchvision.datasets.MNIST(
        root=config.data_dir,train=False,transform = transforms.ToTensor())

    train_loader = DataLoader(dataset=train_dataset,batch_size=config.batch_size,shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,batch_size=config.batch_size,shuffle=False)

    print("train_batch numbers:{}\t batch_size:{}\ttest_batch numbers:{}\t batch_size:{}"
          .format(len(train_loader),config.batch_size,len(test_loader),config.batch_size))

    print("load data successfully")
    return train_loader,test_loader