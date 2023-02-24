import torch
import torch.nn as nn
import torch.nn.functional as F
from vqgan import VQGAN
from choose import ChooseNet
from encoder import Encoder
from decoder import Decoder
from codebook import Codebook

class Choose(nn.Module):
    def __init__(self, args):
        super(Choose, self).__init__()
        self.encoder = Encoder(args).to(device=args.device)
        self.vqgan = self.load_vqgan(args)
        self.decoder = self.vqgan.decoder
        self.codebook = self.vqgan.codebook
        self.beta = args.beta
        self.choose = ChooseNet(args)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)
        self.post_quant_conv = self.vqgan.post_quant_conv
    @staticmethod
    def load_vqgan(args):
        model = VQGAN(args)
        model.load_checkpoint(args.checkpoint_path)
        model = model.eval()
        return model

    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        z=quant_conv_encoded_images
        min_encoding_indices = self.choose(quant_conv_encoded_images)
        #print(z.shape)
        #print(self.codebook.embedding(min_encoding_indices).shape)
        z_q = self.codebook.embedding(min_encoding_indices).view(z.shape)
        #print(z_q.shape)
        q_loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()
        #z_q = z_q.permute(0, 3, 1, 2)
        codebook_mapping=z_q
        codebook_indices=min_encoding_indices
        #print(codebook_mapping.shape)
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images, codebook_indices, q_loss

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
    @torch.no_grad()
    def log_images(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        min_encoding_indices = self.choose(quant_conv_encoded_images)
        print(min_encoding_indices)
        codebook_mapping = self.codebook.embedding(min_encoding_indices).view(quant_conv_encoded_images.shape)
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images