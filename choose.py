import torch
import torch.nn as nn
from helper import ResidualBlock, NonLocalBlock, DownSampleBlock, UpSampleBlock, GroupNorm, Swish
#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

class ChooseNet(nn.Module):
    def __init__(self, args):
        super(ChooseNet, self).__init__()
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        channels = [128, 128, 128, 256, 256, 512]
        attn_resolutions = [16]
        num_res_blocks = 2
        resolution = 256
        layers = [nn.Conv2d(self.latent_dim, channels[0], 3, 1, 1)]
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            # if i != len(channels) - 2:
            #     layers.append(DownSampleBlock(channels[i + 1]))
            #     resolution //= 2
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], self.latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        #print(self.model)
        #print(z.shape)
        d=self.model(z)
        #print(d.shape)
        d = d.permute(0, 2, 3, 1).contiguous()
        d = d.view(-1, self.latent_dim)
        min_encoding_indices = torch.argmax(d, dim=1)
        return min_encoding_indices
