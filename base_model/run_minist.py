import torch
import torch.nn as nn
from base_model.config import parser

import base_model.model as models
import base_model.load_dataset as load_dataset
import os
import time

from tensorboardX import SummaryWriter
"""定义单批次训练函数
input:
    model:使用的模型
    epoch:当前的训练次数
    train_loader:训练集的dataloader
    config:超参数配置信息
    criterion:损失函数
    optimizer:优化器
    device:CPU or GPU
"""
def train(model,epoch,train_loader,config,criterion,optimizer,device):
    model.train()
    images_num = 0
    loss_all = 0
    for idx,data in enumerate(train_loader):
        """梯度置0"""
        optimizer.zero_grad()

        images,labels = data[0].to(device),data[1].to(device)

        """前馈的过程"""
        output = model(images)
        loss = criterion(output,labels)
        """反向传播的过程"""
        loss.backward()
        optimizer.step()

        images_num += len(images)
        loss_all += len(images)*loss.item()
        if (idx+1) % 10 == 0:
            print("epoch: {} / {} \t images: {} / {} \t Loss: {:.5f} / {:.5f}"
                  .format(epoch,config.epoch_num,idx+1,len(train_loader),loss.item(),loss_all / images_num))
    return loss_all / images_num

"""test 函数
参数基本同上，不介绍了
"""
def test(model,epoch,test_loader,config,criterion,device):
    model.eval()
    test_loss = 0
    correct = 0
    num = 0
    with torch.no_grad():
        for idx,data in enumerate(test_loader):
            images,labels = data[0].to(device),data[1].to(device)
            out = model(images)

            test_loss += criterion(out,labels).item()
            num += labels.size(0)
            predict = torch.max(out,1)[1].view(-1,labels.size(0)).long()
            correct += labels.eq(predict.data).cpu().sum().item()

        print("-" * 20)
        print("Epoch:{} \t Test Average loss : {:.5f} \t Accuracy {:.5f} \t".format(epoch, test_loss / len(test_loader),correct / num))
        print("-"*20)
    return correct / num


def main(config):
    """main函数"""
    print("model name is {}".format(config.model_name))

    """加载device"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if torch.cuda.is_available():
        print("use GPU: %s"%n_gpu)
    else:
        print("use CPU")

    """load model"""
    if config.model_name =="LR":
        model = models.LogisticRegressionMulti(config).to(device)
        path = os.path.join(config.chkpt_dir,'LR')
        save_path = os.path.join(path,'model_epochs{}.ckpt'.format(config.epoch_num))
        writer = SummaryWriter(log_dir=path + '/' + time.strftime('%H:%M:%S', time.gmtime()))

    if not os.path.exists(path):
        os.makedirs(path)

    """load dataset"""
    train_loader,test_loader = load_dataset.load_minist(config)

    """ loss function and optimizer """

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = model.parameters(),lr = config.learning_rate)

    """训练测试"""
    for epoch in range(1,config.epoch_num + 1):
        if config.train:
            loss = train(model,epoch,train_loader,config,criterion,optimizer,device)
            writer.add_scalar("Train Loss", loss, epoch)
        else:  # 直接加载模型，做测试
            model.load_state_dict(torch.load(save_path))

        acc = test(model,epoch,test_loader,config,criterion,device)
        writer.add_scalar("Test accuracy", acc, epoch)

    writer.close()

    # train时，才需要保存model
    if config.train:
        torch.save(model.state_dict(), save_path)
if __name__ =="__main__":
    config = parser.parse_args()
    main(config)