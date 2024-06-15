import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicResBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = nn.Sequential(
                    nn.Conv3d(
                        inplanes,
                        planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes))

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride != 1 or self.inplanes != self.planes:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DABlock(nn.Module):
    def __init__(self, in_channels, k = 3, s = 1, p = 1, gamma = 2):
        super(DABlock, self).__init__()
        
        s_channels = in_channels // gamma

        if s_channels < 1:
            s_channels = 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, s_channels, kernel_size=k, stride=s, padding=p, bias=False),
            nn.ReLU(),  
            nn.Conv2d(s_channels, in_channels, kernel_size=k, stride=s, padding=p, bias=False),
            nn.Tanh()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, s_channels, kernel_size=k, stride=s, padding=p, bias=False),
            nn.ReLU(),
            nn.Conv2d(s_channels, in_channels, kernel_size=k, stride=s, padding=p, bias=False),
            nn.Tanh()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, s_channels, kernel_size=k, stride=s, padding=p, bias=False),
            nn.ReLU(),
            nn.Conv2d(s_channels, in_channels, kernel_size=k, stride=s, padding=p, bias=False),
            nn.Tanh()
        )
        

    def forward(self, x):
        # x.shape: [batch_size, C, H, W, D]
        # print(x.shape)

        # alpha1 = x.mean(2)
        alpha1, _ = torch.max(x, dim=2)
        alpha1 = self.conv1(alpha1).unsqueeze(2)

        alpha2, _ = torch.max(x, dim=3)
        alpha2 = self.conv2(alpha2).unsqueeze(3)

        alpha3, _ = torch.max(x, dim=4)
        alpha3 = self.conv3(alpha3).unsqueeze(4)

        alpha = (alpha1.expand_as(x) + alpha2.expand_as(x) + alpha3.expand_as(x)) / 3
        x = alpha * x + x

        return x

class DACNN(nn.Module):
    def __init__(self, dilations=[1,2,3,4], fc_num=2, dropout_rate=0, gamma=2, DA=True):
        super(DACNN, self).__init__()

        self.DA = DA

        self.DA_block_0 = DABlock(1, gamma=gamma)
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                1,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        )

        if DA:
            self.layer1 = nn.Sequential(
                DABlock(64, gamma=gamma),
                BasicResBlock(64, 64, 1, dilations[0])
            )

            self.layer2 = nn.Sequential(
                DABlock(64, gamma=gamma),
                BasicResBlock(64, 128, 1, dilations[1])
            )

            self.layer3 = nn.Sequential(
                DABlock(128, gamma=gamma),
                BasicResBlock(128, 256, 1, dilations[2])
            )

            self.layer4 = nn.Sequential(
                DABlock(256, gamma=gamma),
                BasicResBlock(256, 512, 1, dilations[3])
            )
        else:
            self.layer1 = BasicResBlock(64, 64, 1, dilations[0])
            self.layer2 = BasicResBlock(64, 128, 1, dilations[1])
            self.layer3 = BasicResBlock(128, 256, 1, dilations[2])
            self.layer4 = BasicResBlock(256, 512, 1, dilations[3])

        self.pool = nn.MaxPool3d(kernel_size=2)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.fc_num = fc_num
        if self.fc_num == 1:
            self.fc = nn.Linear(512, 1)
        elif self.fc_num == 2:
            self.fc1 = nn.Linear(512, 64)
            self.fc2 = nn.Linear(64, 1)
        else:
            self.fc3 = nn.Linear(512, 64)
            self.fc4 = nn.Linear(64, 2)

        self.dropout_rate = dropout_rate

    def forward(self, x):
        
        if self.DA:
            x = self.DA_block_0(x)

        x = self.conv1(x)

        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = self.layer4(x)
        x1 = self.pool(x)

        x = self.global_pool(x1)
        x = x.view(x.shape[0], 512)

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        if self.fc_num == 1:
            x = self.fc(x)
        elif self.fc_num == 2:
            x = self.fc2(F.relu(self.fc1(x)))
        else:
            x = self.fc4(F.relu(self.fc3(x)))
        

        return x

if __name__ == '__main__':
    model = DACNN()
    img = torch.randn(2, 1, 128, 128, 128)