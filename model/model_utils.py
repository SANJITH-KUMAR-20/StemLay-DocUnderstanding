import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.num_patches_w = self.patch_shape[0]
        self.num_patches_h = self.patch_shape[1]

    def forward(self, x, position_embedding=None):
        x = self.proj(x)

        if position_embedding is not None:
            position_embedding = position_embedding.view(1, self.patch_shape[0], self.patch_shape[1], -1).permute(0, 3, 1, 2)
            Hp, Wp = x.shape[2], x.shape[3]
            position_embedding = F.interpolate(position_embedding, size=(Hp, Wp), mode='bicubic')
            x = x + position_embedding

        x = x.flatten(2).transpose(1, 2)
        return x