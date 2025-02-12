import torch.nn as nn
import torch
import F_conv as fn
import math

class F_Conv_AlexNet(nn.Module):
    def __init__(self, num_classes=10, init_weights=False):
        super(F_Conv_AlexNet, self).__init__()
        self.features = nn.Sequential(
            fn.Fconv_PCA(5, 1, 64 // 8, 8, ifIni=1, padding=2),  # 5x5，保持尺寸
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (28,28) → (14,14)

            fn.Fconv_PCA(3, 64 // 8, 192 // 8, 8, ifIni=0, padding=1),  # (14,14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (14,14) → (7,7)

            fn.Fconv_PCA(3, 192 // 8, 384 // 8, 8, ifIni=0, padding=1),  # (7,7)
            nn.ReLU(inplace=True),

            fn.Fconv_PCA(3, 384 // 8, 256 // 8, 8, ifIni=0, padding=1),  # (7,7)
            nn.ReLU(inplace=True),

            fn.Fconv_PCA_out(3, 256 // 8, 256, 8, padding=1),  # (7,7)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (7,7) → (3,3)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 3 * 3, 4096),  # 由于输入变小，FC 需要改
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        pass


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),  # 改为 5x5，保持尺寸
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (28,28) → (14,14)

            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),  # (14,14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (14,14) → (7,7)

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),  # (7,7)
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),  # (7,7)
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # (7,7)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (7,7) → (3,3)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 3 * 3, 4096),  # 由于输入变小，FC 需要改
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)  # 通过卷积层
        x = torch.flatten(x, start_dim=1)  # 展平成 (batch_size, 256*3*3)
        x = self.classifier(x)  # 通过全连接层
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
