'''
    SQET + AUX_LOSS
'''

from torch import nn, einsum
import torch
import torch.nn.functional as F
from einops import rearrange

from model.se_attentions import SEAttention


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class InceptionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_se):
        super(InceptionLayer, self).__init__()
        # branch1: maxpool3*3(stride2 valid)
        self.b1 = nn.MaxPool3d(kernel_size=3, stride=stride, padding=1)

        # branch2: conv3*3(n stride2 valid)
        self.b2 = BasicConv3d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1)

        # branch3: conv1*1(k) --> conv3*3(l) --> conv3*3(m stride2 valid)
        self.b3_1 = BasicConv3d(in_channels, in_channels // 2, kernel_size=1)
        self.b3_2 = BasicConv3d(in_channels // 2, int(in_channels / 1.7), kernel_size=3, padding=1)
        self.b3_3 = BasicConv3d(int(in_channels / 1.7), in_channels, kernel_size=3, stride=stride, padding=1)

        self.bottle_neck = BasicConv3d(3 * in_channels, out_channels, kernel_size=1)
        self.use_se = use_se

        if use_se:
            self.side_path = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(out_channels, out_channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels // 4, out_channels),
                nn.Sigmoid(),
            )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3_3(self.b3_2(self.b3_1(x)))

        outputsRedA = [y1, y2, y3]
        o = torch.cat(outputsRedA, 1)
        o = self.bottle_neck(o)
        if self.use_se:
            se_res = self.side_path(o)
            se_res = se_res.view(se_res.shape[0], se_res.shape[1], 1, 1, 1)
            o = o * se_res
        return o

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


def FeedForward(dim, mult=4, dropout=0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv3d(dim, dim * mult, 1),
        nn.GELU(),
        nn.Dropout(dropout, inplace=True),
        nn.Conv3d(dim * mult, dim, 1)
    )



'''
    dim:                 每一个向量长度
    window_size:         patch
    depth：              一共多少个transformer模块
    dim_head：           多头注意力，每个头的长度
    attn_dropout：
    ff_dropout：
    use_pse:              position embedding
    activation：          激活函数，choices: sigmoid, softmax
    blk_shape:
'''
class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            window_size,
            depth=4,
            dim_head=32,
            attn_dropout=0.,
            ff_dropout=0.,
            use_pse=False,
            activation='sigmoid',  # choices: sigmoid, softmax
            blk_shape=None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                SEAttention(dim, window_size=window_size,
                            dim_head=dim_head, dropout=attn_dropout, use_pse=use_pse, activation=activation, blk_shape=blk_shape),
                FeedForward(dim, dropout=ff_dropout)
            ]))

    def forward(self, x):
        for long_attn, long_ff in self.layers:
            x = long_attn(x) + x
            x = long_ff(x) + x

        return x


class SQET(nn.Module):
    def __init__(
            self,
            *,
            dim=(128, 256, 512),
            depth=(2, 8, 2),
            global_window_size=(8, 4, 2, 1),
            cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4)),
            cross_embed_strides=(2, 2, 2),
            num_classes=1,
            attn_dropout=0.1,
            ff_dropout=0.1,
            channels=1,
            use_sequence_pooling=True,  # use sequence pooling instead of pooling
            use_se=True,                # squeeze and excitation in inception
            use_pse=True,               # position embedding in transformer
            input_shape=[128, 128, 128]
    ):
        super().__init__()

        self.use_sequence_pooling = use_sequence_pooling

        dim = cast_tuple(dim, 3)
        depth = cast_tuple(depth, 3)
        global_window_size = cast_tuple(global_window_size, 3)
        cross_embed_kernel_sizes = cast_tuple(cross_embed_kernel_sizes, 3)
        cross_embed_strides = cast_tuple(cross_embed_strides, 3)

        assert len(dim) == 3
        assert len(depth) == 3
        assert len(global_window_size) == 3
        assert len(cross_embed_kernel_sizes) == 3
        assert len(cross_embed_strides) == 3

        # dimensions

        last_dim = dim[-1]
        dims = [32, *dim]
        dim_in_and_out = tuple(zip(dims[:-1], dims[1:]))

        self.stem = nn.Sequential(
            BasicConv3d(channels, 32, kernel_size=3, stride=2, padding=1),
        )

        # layers
        for i in range(len(input_shape)):
            input_shape[i] //= 2

        self.layers = nn.ModuleList([])
        self.aux_layers = nn.ModuleList([])

        for (dim_in, dim_out), layers, global_wsz, cel_kernel_sizes, cel_stride in \
                zip(dim_in_and_out, depth, global_window_size, cross_embed_kernel_sizes,
                    cross_embed_strides):

            for i in range(len(input_shape)):
                input_shape[i] //= 2

            self.layers.append(nn.ModuleList([
                InceptionLayer(dim_in, dim_out, stride=cel_stride, use_se=use_se),
                # BasicConv3d(dim_in, dim_out, kernel_size=3, stride=cel_stride, padding=1),
                Transformer(dim_out, window_size=global_wsz, depth=layers,
                            attn_dropout=attn_dropout, ff_dropout=ff_dropout, use_pse=use_pse, blk_shape=input_shape)
            ]))


            self.aux_layers.append(nn.Sequential(
                    nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
                    nn.Flatten(),
                    nn.ReLU(),
                    nn.Linear(dim_out, 1)
                ))

        # final logits
        if use_sequence_pooling:
            self.attention_pool = nn.Linear(last_dim, 1)
        else:
            self.global_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.to_logits = nn.Sequential(
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        aux_out = {}
        ii = 0
        for cel, transformer in self.layers:
            x = cel(x)
            x = transformer(x)
            aux_out['aux{}'.format(ii)] = self.aux_layers[ii](x)
            ii += 1
        if self.use_sequence_pooling:
            x = rearrange(x, 'b dim d h w -> b (d h w) dim')
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2).contiguous(), x).squeeze(-2)
        else:
            x = self.global_pool(x).squeeze(-1).squeeze(-1).squeeze(-1)

        aux_out['final'] = self.to_logits(x)
        return aux_out


if __name__ == '__main__':
    model = SQET(
        num_classes=1,  # number of output classes
        dim=(64, 128, 256),  # dimension at each stage
        depth=(4, 8, 5),  # depth of transformer at each stage
        global_window_size=(8, 4, 2),  # global window sizes at each stage
        use_se=True,
        use_pse=True,
        use_sequence_pooling=True,
        input_shape=[128, 128, 128]
    )

    img = torch.randn(2, 1, 128, 128, 128)

    pred = model(img)
    print(pred)
    print(pred['final'].shape)
