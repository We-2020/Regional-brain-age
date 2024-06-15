from functools import partial
from itertools import repeat


import logging
import os
from collections import OrderedDict

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.utils.data
from einops.layers.torch import Rearrange

from timm.models.layers import DropPath, trunc_normal_






class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='trilinear', align_corners=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
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
            nn.Conv3d(F_l, F_int, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=3, stride=1, padding=1, bias=False),
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

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=filters[0])
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

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=True,
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv3d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', nn.BatchNorm3d(dim_in)),
                ('rearrage', Rearrange('b c d h w -> b (d h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool3d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c d h w -> b (d h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, d, h, w):
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, d*h*w], 1)

        x = rearrange(x, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c d h w -> b (d h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c d h w -> b (d h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c d h w -> b (d h w) c')

        if self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)

        return q, k, v

    def forward(self, x, d, h, w):
        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, d, h, w)

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        flops = 0

        _, T, C = input.shape
        H = W = int(np.sqrt(T-1)) if module.with_cls_token else int(np.sqrt(T))

        H_Q = H / module.stride_q
        W_Q = H / module.stride_q
        T_Q = H_Q * W_Q + 1 if module.with_cls_token else H_Q * W_Q

        H_KV = H / module.stride_kv
        W_KV = W / module.stride_kv
        T_KV = H_KV * W_KV + 1 if module.with_cls_token else H_KV * W_KV

        # C = module.dim
        # S = T
        # Scaled-dot-product macs
        # [B x T x C] x [B x C x T] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        flops += T_Q * T_KV * module.dim
        # [B x T x S] x [B x S x C] --> [B x T x C]
        flops += T_Q * module.dim * T_KV

        if (
            hasattr(module, 'conv_proj_q')
            and hasattr(module.conv_proj_q, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_q.conv.parameters()
                ]
            )
            flops += params * H_Q * W_Q

        if (
            hasattr(module, 'conv_proj_k')
            and hasattr(module.conv_proj_k, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_k.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        if (
            hasattr(module, 'conv_proj_v')
            and hasattr(module.conv_proj_v, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_v.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        params = sum([p.numel() for p in module.proj_q.parameters()])
        flops += params * T_Q
        params = sum([p.numel() for p in module.proj_k.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj_v.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj.parameters()])
        flops += params * T

        module.__flops__ += flops


class Block(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        self.with_cls_token = kwargs['with_cls_token']

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(
            dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop,
            **kwargs
        )

        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=dim_mlp_hidden,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x, d, h, w):
        res = x

        x = self.norm1(x)
        attn = self.attn(x, d, h, w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class UNetWithTransformerEncoder(nn.Module):
    def __init__(self):
        super(UNetWithTransformerEncoder, self).__init__()
        self.AttU_Net = AttU_Net()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.block5 = self.CreatBlock(depth = 3, dim_in=filters[4],dim_out=filters[4],num_heads=n1,mlp_ratio=4.,qkv_bias=False)
        self.block4 = self.CreatBlock(depth = 3, dim_in=filters[3],dim_out=filters[3],num_heads=n1,mlp_ratio=4.,qkv_bias=False)
        self.block3 = self.CreatBlock(depth = 3, dim_in=filters[2],dim_out=filters[2],num_heads=n1,mlp_ratio=4.,qkv_bias=False)
        self.block2 = self.CreatBlock(depth = 3, dim_in=filters[1],dim_out=filters[1],num_heads=n1,mlp_ratio=4.,qkv_bias=False)
        self.block1 = self.CreatBlock(depth = 3, dim_in=filters[0],dim_out=filters[0],num_heads=n1,mlp_ratio=4.,qkv_bias=False)
        self.pool = nn.AdaptiveMaxPool3d((16,16,16))
        self.merge = nn.Conv3d(in_channels=sum(filters),out_channels=n1,kernel_size=3,stride=1,padding=1,bias=True)
        self.linear = nn.Linear(n1*16*16*16,1)
    def CreatBlock(self, depth,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio,
                qkv_bias=False,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                act_layer=nn.GELU
                ):
        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=dim_in,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=nn.LayerNorm,
                    with_cls_token=False
                )
            )
        self.blocks = nn.ModuleList(blocks)
        return self.blocks
    def ProcessBlock(self,decoder,block, D, H, W):
        for i, blk in enumerate(block):
            decoder = blk(decoder, D, H, W)
        return decoder
    def forward(self, x):
        # 从U-Net模型中获取特征
        e5, d5, d4, d3, d2 = self.AttU_Net(x)
        e5 = self.pool(e5)
        d5 = self.pool(d5)
        d4 = self.pool(d4)
        d3 = self.pool(d3)
        d2 = self.pool(d2)
        _,_,D5, H5, W5 = e5.size()
        _,_,D4, H4, W4 = d5.size()
        _,_,D3, H3, W3 = d4.size()
        _,_,D2, H2, W2 = d3.size()
        _,_,D1, H1, W1 = d2.size()

        e5 = rearrange(e5, 'b c d h w -> b (d h w) c')
        d5 = rearrange(d5, 'b c d h w -> b (d h w) c')
        d4 = rearrange(d4, 'b c d h w -> b (d h w) c')
        d3 = rearrange(d3, 'b c d h w -> b (d h w) c')
        d2 = rearrange(d2, 'b c d h w -> b (d h w) c')

        e5 = self.ProcessBlock(e5, self.block5,D5, H5, W5)
        d5 = self.ProcessBlock(d5, self.block4,D4, H4, W4)
        d4 = self.ProcessBlock(d4, self.block3,D3, H3, W3)
        d3 = self.ProcessBlock(d3, self.block2,D2, H2, W2)
        d2 = self.ProcessBlock(d2, self.block1,D1, H1, W1)

        e5 = rearrange(e5, 'b (d h w) c -> b c d h w',d=D5,h=H5,w=W5)
        d5 = rearrange(d5, 'b (d h w) c -> b c d h w',d=D4,h=H4,w=W4)
        d4 = rearrange(d4, 'b (d h w) c -> b c d h w',d=D3,h=H3,w=W3)
        d3 = rearrange(d3, 'b (d h w) c -> b c d h w',d=D2,h=H2,w=W2)
        d2 = rearrange(d2, 'b (d h w) c -> b c d h w',d=D1,h=H1,w=W1)

        feature = torch.cat([e5, d5, d4, d3, d2],dim=1)
        output = self.merge(feature)
        output = torch.flatten(output, 1)
        output = self.linear(output)

        return output




if __name__ == '__main__':
    model = UNetWithTransformerEncoder()
    a = torch.randn((4, 1, 91, 91, 109))
    print(model(a).shape)
    total = sum([param.nelement() for param in model.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))
