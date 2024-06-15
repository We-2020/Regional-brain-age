'''
    crossformer 3d without short attention
'''
import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F


# helpers
# 将单个元素转化成元组
def cast_tuple(val, length = 1):
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


# cross embed layer
# 多个卷积层卷起来
class CrossEmbedLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_sizes,
        use_se,
        stride=2,
    ):
        super().__init__()
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv3d(dim_in, dim_scale, kernel, stride=stride, padding=(kernel - stride) // 2))

        self.use_se = use_se

        if use_se:
            self.side_path = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(dim_out, dim_out // 4),
                nn.ReLU(inplace=True),
                nn.Linear(dim_out // 4, dim_out),
                nn.Sigmoid(),
            )

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        o = torch.cat(fmaps, dim=1)
        if self.use_se:
            se_res = self.side_path(o)
            se_res = se_res.view(se_res.shape[0], se_res.shape[1], 1, 1, 1)
            o = o * se_res
        return o


# transformer classes
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


class Attention(nn.Module):
    def __init__(
        self,
        dim,                # 总的维度
        attn_type,
        window_size,
        dim_head=32,        # default 多头，每个头多少维度
        dropout=0.,
        use_pse=False,
    ):
        super().__init__()
        assert attn_type in {'short', 'long'}, 'attention type must be one of local or distant'
        heads = dim // dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads    # 这里inner_dim == dim

        self.attn_type = attn_type
        self.window_size = window_size

        self.norm = LayerNorm(dim)

        self.dropout = nn.Dropout(dropout, inplace=True)

        self.to_qkv = nn.Conv3d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv3d(inner_dim, dim, 1)

        self.use_pse = use_pse
        if use_pse:
            self.pse = nn.Parameter(torch.randn((self.window_size, self.window_size, self.window_size)))

    def forward(self, x):
        *_, depth, height, width, heads, wsz, device = *x.shape, self.heads, self.window_size, x.device

        # prenorm
        x = self.norm(x)
        # print(x.shape)

        # rearrange for short or long distance attention

        if self.attn_type == 'short':
            x = rearrange(x, 'b c (d s0) (h s1) (w s2) -> (b d h w) c s0 s1 s2', s0=wsz, s1=wsz, s2=wsz)
        elif self.attn_type == 'long':
            x = rearrange(x, 'b c (l0 d) (l1 h) (l2 w) -> (b d h w) c l0 l1 l2', l0=wsz, l1=wsz, l2=wsz)
        # queries / keys / values
        if self.use_pse:
            x = x + self.pse    # what ?

        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        # print(q.shape)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b (h c) d x y -> b h (d x y) c', h=heads), (q, k, v))
        # print(q.shape)
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        # merge heads

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (d x y) c -> b (h c) d x y', d=wsz, x=wsz, y=wsz)
        out = self.to_out(out)

        # rearrange back for long or short distance attention

        if self.attn_type == 'short':
            out = rearrange(out, '(b d h w) c s0 s1 s2 -> b c (d s0) (h s1) (w s2)',
                            d=depth//wsz, h=height // wsz, w=width // wsz)
        elif self.attn_type == 'long':
            out = rearrange(out, '(b d h w) c l0 l1 l2 -> b c (l0 d) (l1 h) (l2 w)',
                            d=depth//wsz, h=height // wsz, w=width // wsz)

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        dim,                    # 每个维度
        *,
        local_window_size,
        global_window_size,
        depth=4,
        dim_head=32,            # 我们一直用默认的
        attn_dropout=0.1,
        ff_dropout=0.1,
        use_pse=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, attn_type='long', window_size=global_window_size,
                          dim_head=dim_head, dropout=attn_dropout, use_pse=use_pse),
                FeedForward(dim, dropout=ff_dropout)
            ]))

    def forward(self, x):
        for long_attn, long_ff in self.layers:
            x = long_attn(x) + x
            x = long_ff(x) + x
        return x


# classes
class CrossFormer(nn.Module):
    def __init__(
        self,
        *,
        dim=(128, 256, 512),
        depth=(2, 8, 2),                # （2,4,3)?
        global_window_size=(8, 4, 1),   # global_window_size
        local_window_size=7,            # local_window_size
        cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4)),      # ((4, 8), (2, 4), (2, 4)) ？
        cross_embed_strides=(2, 2, 2),
        num_classes=1,
        attn_dropout=0.,
        ff_dropout=0.,
        channels=1,
        use_inception=False,         # use inception block instead of origin conv. The origin form is equivalent to False
        use_sequence_pooling=True,   # use sequence pooling instead of pooling. The origin form is equivalent to True.
        use_se=False,                # squeeze and excitation. The origin form is equivalent to False.
        use_pse=False,               # position embedding. The origin form is equivalent to False.
    ):
        super().__init__()
        self.use_sequence_pooling = use_sequence_pooling

        dim = cast_tuple(dim, 3)
        depth = cast_tuple(depth, 3)
        global_window_size = cast_tuple(global_window_size, 3)
        local_window_size = cast_tuple(local_window_size, 3)
        cross_embed_kernel_sizes = cast_tuple(cross_embed_kernel_sizes, 3)
        cross_embed_strides = cast_tuple(cross_embed_strides, 3)

        assert len(dim) == 3
        assert len(depth) == 3
        assert len(global_window_size) == 3
        assert len(local_window_size) == 3
        assert len(cross_embed_kernel_sizes) == 3
        assert len(cross_embed_strides) == 3

        # dimensions

        # self.stem = nn.Sequential(
        #     BasicConv3d(channels, 32, kernel_size=3, stride=2, padding=1),
        #     BasicConv3d(32, 64, kernel_size=3, stride=1, padding=1),
        #     BasicConv3d(64, 64, kernel_size=3, stride=1, padding=1),
        # )
        # dims = [64, *dim]

        self.stem = nn.Sequential(
            BasicConv3d(channels, 32, kernel_size=3, stride=2, padding=1),
        )

        last_dim = dim[-1]
        dims = [32, *dim]   # 增加一个维度32
        dim_in_and_out = tuple(zip(dims[:-1], dims[1:]))    # ((32, 64), (64, 128), (128, 256))

        # layers

        self.layers = nn.ModuleList([])

        # 三个stage
        for (dim_in, dim_out), layers, global_wsz, local_wsz, cel_kernel_sizes, cel_stride in\
                zip(dim_in_and_out, depth, global_window_size, local_window_size, cross_embed_kernel_sizes, cross_embed_strides):   # 对应位置的元素重新组合，生成一个个新的元组
            if use_inception:
                self.layers.append(nn.ModuleList([
                    InceptionLayer(dim_in, dim_out, stride=cel_stride, use_se=use_se),
                    Transformer(dim_out, local_window_size=local_wsz, global_window_size=global_wsz, depth=layers,
                                attn_dropout=attn_dropout, ff_dropout=ff_dropout, use_pse=use_pse)
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    CrossEmbedLayer(dim_in, dim_out, cel_kernel_sizes, stride=cel_stride, use_se=use_se),
                    Transformer(dim_out, local_window_size=local_wsz, global_window_size=global_wsz, depth=layers,
                                attn_dropout=attn_dropout, ff_dropout=ff_dropout, use_pse=use_pse)
                ]))

        # final logits
        if use_sequence_pooling:
            self.attention_pool = nn.Linear(last_dim, 1)    # 改变最后一维
        else:
            self.global_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.to_logits = nn.Sequential(
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        for cel, transformer in self.layers:
            x = cel(x)
            x = transformer(x)
#             print('-----')
#             exit(0)
#             print('*******')
        if self.use_sequence_pooling:
            x = rearrange(x, 'b dim d h w -> b (d h w) dim')
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = self.global_pool(x).squeeze(-1).squeeze(-1).squeeze(-1)

        return self.to_logits(x)


if __name__ == '__main__':
    model = CrossFormer(
        num_classes=1,  # number of output classes
        dim=(64, 128, 256),  # dimension at each stage
        depth=(2, 8, 2),  # depth of transformer at each stage
        global_window_size=(8, 4, 1),  # global window sizes at each stage
        local_window_size=4,  # local window size (can be customized for each stage)
        use_se=True,
        use_pse=True,
        use_inception=True,
        use_sequence_pooling=True,
    )

    img = torch.randn(1, 1, 128, 128, 128)

    pred = model(img)
    print(pred)
