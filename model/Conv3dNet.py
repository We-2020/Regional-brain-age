import torch.nn as nn
import torch

class Conv3dModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.Conv3d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.Conv3d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        self.bn = nn.BatchNorm3d(96)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 = nn.Linear(96, 32)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.bn(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)

        x = self.fc2(self.relu(self.fc1(x)))

        return x

if __name__ == '__main__':
    model = Conv3dModule()
    img = torch.randn(1, 1, 128, 128, 128)

    pred = model(img)
    print(pred)