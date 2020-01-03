import argparse

parser =argparse.ArgumentParser(description="Minist 数据集")

parser.add_argument("--model_name",default="LR",type = str,help = "the model name: FNN,CNN,RNN")

"""数据集的信息"""
parser.add_argument("--input_size",default=784,type= int,help = "minist单张图片，28*28 大小，用于linear层计算")
parser.add_argument("--output_size",default=10,type= int,help = "minsit的label一共有10类")
parser.add_argument("--data_dir",default="./data/",type=str,help = "minist下载后的存放路径")
parser.add_argument("--batch_size",default=64,type=int,help="batch_size的大小")

"""模型的信息"""
parser.add_argument("--learning_rate",default=0.001,type=int,help = "优化器的学习率")
parser.add_argument("--epoch_num",default=10,type=int,help="训练整批数据的次数")

"""log信息"""
parser.add_argument("--chkpt_dir",default="./chkpt/model",type=str,help="存放训练模型的路径")

"""训练还是测试"""
parser.add_argument("--train",default=True,type=bool,help="True 表示需要训练，False表示直接加载模型，不需要训练")
