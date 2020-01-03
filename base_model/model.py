import torch
import torch.nn as nn
import torch.nn.functional as F


""" 逻辑回归多分类"""
class LogisticRegressionMulti(nn.Module):
    def __init__(self,config):
        super(LogisticRegressionMulti,self).__init__()
        self.config = config
        self.LR = nn.Linear(config.input_size,config.output_size)

    def forward(self, x):
        #  torch.Size([64, 1, 28, 28]) ,第二个维度是留个类似卷积操作的channels的
        x = x.view(-1,self.config.input_size)
        # print(x.size())
        #  torch.Size([64, 784])
        x = self.LR(x)
        #  torch.Size([64, 1, 10])
        # print(x.size())
        return x