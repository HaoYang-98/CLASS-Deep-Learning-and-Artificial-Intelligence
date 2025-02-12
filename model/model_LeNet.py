import torch.nn as nn
import torch.nn.functional as F
import F_conv as fn

# 将卷积改成等变卷积的LeNet模型
class F_Conv_LeNet(nn.Module):
    def __init__(self):
        super(F_Conv_LeNet, self).__init__()
        tranNum = 8  # 2*pi/tranNum degree rotation equviariant
        kernel_size = 5
        c_in = 1
        c_out = 16
        self.conv1 = fn.Fconv_PCA(kernel_size, c_in, c_out // tranNum, tranNum, ifIni=1)  # 修改输入通道数为1
        self.pool1 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*12*12, 120)  # 修改全连接层输入尺寸
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))    # input(1, 28, 28) output(16, 24, 24)
        x = self.pool1(x)            # output(16, 12, 12)
        x = x.view(-1, 16*12*12)     # output(16*12*12)
        x = F.relu(self.fc1(x))      # output(120)
        x = self.fc2(x)              # output(10)
        return x

# 减少网络深度的LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)  # 修改输入通道数为1
        self.pool1 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*12*12, 120)  # 修改全连接层输入尺寸
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))    # input(1, 28, 28) output(16, 24, 24)
        x = self.pool1(x)            # output(16, 12, 12)
        x = x.view(-1, 16*12*12)     # output(16*12*12)
        x = F.relu(self.fc1(x))      # output(120)
        x = self.fc2(x)              # output(10)
        return x

# 最初的LeNet模型
# import torch.nn as nn
# import torch.nn.functional as F
# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, 5)  # 修改输入通道数为1
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(16, 32, 5)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(32*4*4, 120)  # 修改全连接层输入尺寸
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))    # input(1, 28, 28) output(16, 24, 24)
#         x = self.pool1(x)            # output(16, 12, 12)
#         x = F.relu(self.conv2(x))    # output(32, 8, 8)
#         x = self.pool2(x)            # output(32, 4, 4)
#         x = x.view(-1, 32*4*4)       # output(32*4*4)
#         x = F.relu(self.fc1(x))      # output(120)
#         x = F.relu(self.fc2(x))      # output(84)
#         x = self.fc3(x)              # output(10)
#         return x
