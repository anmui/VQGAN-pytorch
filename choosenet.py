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
        codebook_mapping, codebook_indices, q_loss = self.choose(quant_conv_encoded_images)
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images, codebook_indices, q_loss

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
    @torch.no_grad()
    def log_images(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.choose(quant_conv_encoded_images)
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images