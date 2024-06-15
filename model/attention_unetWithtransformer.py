import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch,size):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(size=size,mode='trilinear', align_corners=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, img_ch=1, output_ch=1):
        super(AttU_Net, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3],size=(11,11,13))
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2],size=(22,22,27))
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1],size=(45,45,54))
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0],size=(91, 91, 109))
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])



#         self.fc2 = nn.Linear(13996800, 1)
#         self.fc1 = nn.Linear(57768256, 1)
        #self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        #print(x5.shape)
        d5 = self.Up5(e5)
        # print(d5.shape)
        # print(e4.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        # out = self.Conv(d2)

      #  out = self.active(out)

        return e5, d5, d4, d3, d2
    
class FeatureAdjuster(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FeatureAdjuster, self).__init__()
        # 使用1x1x1卷积来改变特征图的通道数
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class UNetWithTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(UNetWithTransformerEncoder, self).__init__()
        self.AttU_Net = AttU_Net()
        self.channel = 16
        self.adjust_d2 = nn.Conv3d(self.channel, d_model,kernel_size=1)
        self.adjust_d3 = nn.Conv3d(self.channel*2, d_model,kernel_size=1)
        self.adjust_d4 = nn.Conv3d(self.channel*4, d_model,kernel_size=1)
        self.adjust_d5 = nn.Conv3d(self.channel*8, d_model,kernel_size=1)
        self.adjust_e5 = nn.Conv3d(self.channel*16, d_model,kernel_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        self.regressor = nn.Linear(d_model*5*5*5*6, 1)
        # Transformer编码器层的设置
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                    dim_feedforward=dim_feedforward, 
                                                    dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, x):
        # 从U-Net模型中获取特征
        e5, d5, d4, d3, d2 = self.AttU_Net(x)

        d2_resized = self.adjust_d2(F.interpolate(d2, size=e5.shape[2:], mode='trilinear', align_corners=True))
        d3_resized = self.adjust_d3(F.interpolate(d3, size=e5.shape[2:], mode='trilinear', align_corners=True))
        d4_resized = self.adjust_d4(F.interpolate(d4, size=e5.shape[2:], mode='trilinear', align_corners=True))
        d5_resized = self.adjust_d5(F.interpolate(d5, size=e5.shape[2:], mode='trilinear', align_corners=True))
        e5_resized = self.adjust_e5(e5)

#         d2 = self.adjust_d2(F.interpolate(d2, size=d4.shape[2:], mode='trilinear', align_corners=True))
#         d3 = self.adjust_d3(F.interpolate(d3, size=d4.shape[2:], mode='trilinear', align_corners=True))
#         d4 = self.adjust_d4(d4)
#         d5 = self.adjust_d5(F.interpolate(d5, size=d4.shape[2:], mode='trilinear', align_corners=True))
#         e5 = self.adjust_e5(F.interpolate(e5, size=d4.shape[2:], mode='trilinear', align_corners=True))

        # 展平特征以适应TransformerEncoder
        d2_flat = d2_resized.view(d2_resized.size(0), d2_resized.size(1), -1).permute(2, 0, 1)
        d3_flat = d3_resized.view(d3_resized.size(0), d3_resized.size(1), -1).permute(2, 0, 1)
        d4_flat = d4_resized.view(d4_resized.size(0), d3_resized.size(1), -1).permute(2, 0, 1)
        d5_flat = d5_resized.view(d5_resized.size(0), d5_resized.size(1), -1).permute(2, 0, 1)
        e5_flat = e5_resized.view(e5_resized.size(0), e5_resized.size(1), -1).permute(2, 0, 1)

        # 合并特征
        combined_features = torch.cat([d2_flat, d3_flat, d4_flat, d5_flat, e5_flat], dim=0)
        # Transformer融合
#         x = torch.flatten(combined_features.permute(1, 2, 0),1)
        x = self.transformer_encoder(combined_features)
#         x = x  # 重排为 [batch, features, seq_len]，即 [4, 512, 750]
        x = x.permute(1, 2, 0)  # 重排为 [batch, features, seq_len]，即 [4, 512, 750]

#         # 全局平均池化，压缩序列维度
#         x = self.global_avg_pool(x)  # 形状变为 [batch, features, 1]

#         # 展平特征，准备用于线性层
        x = torch.flatten(x, 1)  # 形状变为 [batch, features]

#         # 通过线性层进行回归预测
        x = self.regressor(x)  # 形状变为 [batch, output_features]

        # 后续处理...

        return x


if __name__ == '__main__':
    model = AttU_Net()
    a = torch.randn((4, 1, 91, 91, 109))
    print(model(a).shape)
    total = sum([param.nelement() for param in model.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))