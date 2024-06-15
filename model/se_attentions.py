import torch
from torch import nn, einsum
from einops import rearrange
from copy import deepcopy


class SEAttention(nn.Module):
    def __init__(
            self,
            dim,
            window_size,
            dim_head=32,
            dropout=0.,
            use_pse=False,
            activation='sigmoid',
            blk_shape=None,
    ):
        super().__init__()
        heads = dim // dim_head
        self.window_size = window_size
        self.pre_pooling = nn.AvgPool3d(kernel_size=window_size, stride=window_size)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout, inplace=False)

        if activation == 'sigmoid':
            activation_func = nn.Sigmoid()
        elif activation == 'softmax':
            activation_func = nn.Softmax(dim=-1)
        else:
            assert False
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            activation_func,
        )

        self.use_pse = use_pse

        if use_pse:
            assert blk_shape is not None
            self.blk_shape = deepcopy(blk_shape)
            for i in range(len(self.blk_shape)):
                self.blk_shape[i] //= self.window_size
            self.pse = nn.Parameter(torch.randn(self.blk_shape))

    def forward(self, x):                   # (1,64,32,32,32)
        wsz = self.window_size              # 8
        num_patch = x.shape[-1] // wsz      # 32 // 8 = 4
        x1 = self.pre_pooling(x)            # (b, c, d / wsz, h / wsz, w / wsz)   (1,64,4,4,4)
        if self.use_pse:
            x1 = x1 + self.pse
        x1 = rearrange(x1, 'b c d h w -> b 1 (d h w) c')    # (1,1,64,64)
        x1 = self.norm(x1)
        qkv = self.to_qkv(x1).chunk(3, dim=-1)  # (1,1,64,64*3)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h c) -> b p h n c', h=self.heads), qkv)  # (1,1,64,64)->(1,1,2,64,32)

        dots = torch.matmul(q, k.transpose(-1, -2).contiguous()) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)     # (1,1,2,64,32)

        out = rearrange(out, 'b p h n d -> b p n (h d)')  # (b, 1 , d * h * w, c)   (1,1,64,64)
        out = self.to_out(out)  # (b, 1, d / wsz * h / wsz * w / wsz, c)

        out = rearrange(out, 'b 1 (d h w) c -> b c d h w', d=num_patch, h=num_patch, w=num_patch)   # (1,64,4,4,4)

        # Squeeze and Excitation
        # (1,64,32,32,32) -> (512 1 64 4 4 4)
        x = rearrange(x, 'b c (d pd) (h ph) (w pw) -> (pd ph pw) b c d h w', pd=wsz, ph=wsz, pw=wsz)
        out = out * x
        return rearrange(out, '(pd ph pw) b c d h w -> b c (d pd) (h ph) (w pw)',
                         pd=wsz, ph=wsz, pw=wsz)    # (1,64,32,32,32)


if __name__ == '__main__':
    attention = SEAttention(dim=64,  window_size=8,
                            dim_head=32, dropout=0.1, use_pse=False, activation='sigmoid')

    img = torch.randn(1, 64, 32, 32, 32)

    pred = attention(img)
    print(pred.shape)