import torch
import torch.nn as nn
import torch.nn.functional as F


class FashionMNISTCNN(nn.Module):

    def __init__(self):
        super(FashionMNISTCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            #   卷积核设置
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            #   归一化
            nn.BatchNorm2d(32),
            #   激活
            nn.ReLU(),
            #   池化
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)
        #   全连接层
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        #   卷两层，全连接
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    #   默认模型，之后的参数都在这之上修订
