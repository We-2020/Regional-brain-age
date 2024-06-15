"""
    3D MAE-input
"""
import torch
from torch import nn

from vit_pytorch.vit import Transformer
from einops.layers.torch import Rearrange

import random

# 3D-vit
class LocalLocalTransformer3d(nn.Module):
    def __init__(
            self,
            masking_ratio=0,  # masking ratio
            channel=1,
            image_size=(128, 128, 128),  # input image size (channel, height, width)
            patch_size=(16, 16, 16),     # patch size
            embed_dim=1024,             # vector dimension after a patch is embedded
            depth=6,                    # hyper-parameters for transformers
            heads=8,                    # hyper-parameters for transformers
            mlp_dim=1024,               # hyper-parameters for transformers
            # The backbone for embedding an input patch with shape (batch, seq_len, p0, p1, p2),
            # and outputs embedded vectors with shape (batch, seq_len, embed_dim).
            # If set to None, a linear layer is then applied by default.
            backbone=None,
            embed_dropout=0.,           # drop out rate for embedding vectors
            trans_dropout=0.,           # hyper-parameters for transformers
            dim_head=64,                # TODO 每一个head qkv的dim
    ):
        super().__init__()

        self.masking_ratio = masking_ratio

        # define function to split the whole image
        p0, p1, p2 = patch_size
        h, w, d = image_size[0] // p0, image_size[1] // p1, image_size[2] // p2
        self.to_patch = Rearrange('b c (h p0) (w p1) (d p2) -> (b) (h w d) p0 p1 p2 c', p0=p0, p1=p1, p2=p2)

        # define embedding function that change patches into embedded vectors

        self.patch_to_emb = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Linear(p0 * p1 * p2 * channel, embed_dim),
            nn.ReLU(),
        )

        # define transformers
        self.pos_embedding = nn.Parameter(torch.randn(1, h*w*d, embed_dim))
        self.dropout = nn.Dropout(embed_dropout)

        self.transformer = Transformer(embed_dim, depth, heads, dim_head, mlp_dim, trans_dropout)
        self.to_latent = nn.Identity()

        # define regression head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, img, train=True):
        # self.masking_ratio = random.uniform(0, 0.75)
        device = img.device

        # split patches
        patches = self.to_patch(img)            # [1, 343, 16, 16, 16, 5]
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)     # [1, 343, 1024]
        tokens = tokens + self.pos_embedding

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        if train:
            masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        else:
            masked_indices = []
            unmasked_indices = torch.arange(num_patches)

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # attend with vision transformer
        tokens = self.dropout(tokens)
        encoded_tokens = self.transformer(tokens)   # [1, 343, 1024]

        # 另一种做法是不加cls字符，对所有的tokens的输出做一个平均
        encoded_tokens = encoded_tokens.mean(dim=1)  # [1, 1024]

        return self.mlp_head(encoded_tokens)  # [1,1024]



if __name__ == '__main__':
    ll = LocalLocalTransformer3d()
    inp = torch.randn((1, 1, 128, 128, 128))
    print(inp.shape)
    o = ll(inp)
    print(o.shape)
