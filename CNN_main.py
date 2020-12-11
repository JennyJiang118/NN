import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
from tensorboardX import SummaryWriter


BATCH_SIZE = 64
LR = 0.005
EPOCH = 2
#每个batch_size的shape：[64,1,28,28]


def get_data():
    train_data = torchvision.datasets.MNIST(
        root="./mnist",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    test_data = torchvision.datasets.MNIST(
        root="./mnist",
        transform=torchvision.transforms.ToTensor(),
        train=False
    )

    # 划分训练集&验证集
    train_data, val_data = Data.random_split(train_data, [50000, 10000])

    train_loader = Data.DataLoader(train_data, BATCH_SIZE, shuffle=True)
    val_loader = Data.DataLoader(val_data, BATCH_SIZE, shuffle=True)
    test_loader = Data.DataLoader(test_data, BATCH_SIZE, shuffle=True)

    return train_loader, val_loader, test_loader



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__() #前面都是规定结构

        #第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,#灰度图，channel=1
                out_channels=16,#自己设定
                kernel_size=3,#卷积核大小
                stride=1,#步长
                padding=1
            ),
            nn.ReLU(),#激活函数
            nn.MaxPool2d(kernel_size=2)#池化降维，取2*2窗口最大值， 宽、高减半，channel不变
        )#shape:[16,14,14]



        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )#shape:[32,7,7]


        #定义全连接层
        self.prediction = nn.Linear(32*7*7, 10)

    #向前传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.prediction(x)
        return output



def train(train_loader, val_loader, test_loader):
    # 创建网络
    cnn = CNN()

    # 大数据常用Adam优化器，参数需要model的参数，以及lr
    optimizer = torch.optim.Adam(cnn.parameters(), LR)
    # 定义损失函数，交叉熵
    loss_func = nn.CrossEntropyLoss()

    logger = SummaryWriter(log_dir="data/choose")

    # 训练阶段
    for epoch in range(EPOCH):
        print("epoch:", epoch)
        # step: 在第几个BATCH_SIZE
        # batch_x: 训练集的图像
        # batch_y: 训练集的标签
        for step, (batch_x, batch_y) in enumerate(train_loader):
            global_iter_num = epoch * len(train_loader)+step+1
            # model只接受Variable的数据，需要先转化
            b_x = Variable(batch_x)
            b_y = Variable(batch_y)
            output = cnn(b_x)
            # 计算误差
            loss = loss_func(output, b_y)
            #将梯度变为零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 优化参数
            optimizer.step()
            if global_iter_num % 50 == 0:
                #test
                acc_sum = []
                acc_val_sum = []
                for i, (test_x, test_y) in enumerate(test_loader):
                    test_x = Variable(test_x)
                    test_y = Variable(test_y)
                    test_output = cnn(test_x)
                    pre_y = torch.max(test_output,1)[1].data.squeeze()
                    acc = float((pre_y == test_y).sum())/float(test_y.size(0))
                    acc_sum.append(acc)
                for i, (val_x, val_y) in enumerate(val_loader):
                    val_x = Variable(val_x)
                    val_y = Variable(val_y)
                    val_output = cnn(val_x)
                    pre_y = torch.max(val_output, 1)[1].data.squeeze()
                    val_acc = float((pre_y == val_y).sum())/float(val_y.size(0))
                    acc_val_sum.append(val_acc)
                print("global_step:", global_iter_num, "| train loss:%.4f" % loss.data, "|validation accuracy:%.4f"%np.mean(acc_val_sum), "|test accuracy：%.4f" % np.mean(acc_sum))
                logger.add_scalar("train loss", loss.data, global_step=global_iter_num)
                logger.add_scalar("validation accuracy", np.mean(acc_val_sum), global_step=global_iter_num)
                logger.add_scalar("test accuracy", np.mean(acc_sum), global_step=global_iter_num)

                for name, param in cnn.named_parameters():
                    logger.add_histogram(name, param.data.numpy(), global_step=global_iter_num)






def main():
    train_loader, val_loader, test_loader = get_data()
    train(train_loader, val_loader, test_loader)


if __name__ == '__main__':
    main()







