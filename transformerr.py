import math
import torch
from torch import nn

from transformer_nonorm_flx import Transformer
from util.misc import nested_tensor_from_tensor_list, NestedTensor
from vqgan import VQGAN
from decoder import Decoder
from torchvision.models._utils import IntermediateLayerGetter
from position_encoding import build_position_encoding_ours
class Transformerr(nn.Module):
    def __init__(self,args):
        super().__init__()
        #self.vqgan_t,self.vqgan_s = self.load_vqgan(args)
        self.vqgan_t=VQGAN(args).to(device=args.device)
        self.vqgan_s=VQGAN(args).to(device=args.device)
        self.transformer=Transformer(
            d_model=args.latent_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
            enorm=args.enorm,
            dnorm=args.dnorm,
        )
        hidden_dim = self.transformer.d_model
        self.output_proj = nn.Conv2d(hidden_dim, args.latent_dim, kernel_size=1)
        self.position_embedding = build_position_encoding_ours(args)
        # tail_layers = []
        # res_block = ResBlock
        # for ri in range(self.backbone_content.reduce_times):
        #     times = 2 ** ri
        #     content_c = self.backbone_content.num_channels
        #     out_c = 3 if ri == self.backbone_content.reduce_times - 1 else int(content_c / (times * 2))
        #     tail_layers.extend([
        #         res_block(int(content_c / times), int(content_c / (times * 2))),
        #         nn.Upsample(scale_factor=2, mode='bilinear'),
        #         nn.ReflectionPad2d(1),
        #         nn.Conv2d(int(content_c / times), out_c,
        #                   kernel_size=3, stride=1, padding=0),
        #     ])
        self.tail = Decoder(args).to(device=args.device)
        self.input_proj_c = nn.Conv2d(args.latent_dim, hidden_dim, kernel_size=1)
        self.input_proj_s = nn.Conv2d(args.latent_dim, hidden_dim, kernel_size=1)
        self.device=args.device

    @staticmethod
    def load_vqgan(args):
        model_true = VQGAN(args)
        model_true.load_checkpoint(args.checkpoint_path_ture)
        model_true = model_true.eval()
        model_style = VQGAN(args)
        model_style.load_checkpoint(args.checkpoint_path_style)
        model_style = model_style.eval()
        return model_true,model_style
    def forward(self,simg,timg):

        encoded_image_s = self.vqgan_s.encoder(simg)
        encoded_image_t=self.vqgan_t.encoder(timg)
        quant_conv_encoded_images_s = self.vqgan_s.quant_conv(encoded_image_s)
        quant_conv_encoded_images_t = self.vqgan_t.quant_conv(encoded_image_t)
        codebook_mapping_s, codebook_indices_s, q_loss_s = self.vqgan_s.codebook(quant_conv_encoded_images_s)
        #codebook_mapping_t, codebook_indices_t, q_loss_t = self.vqgan_t.codebook(quant_conv_encoded_images_t)
        codebook_mapping_s=self.vqgan_s.post_quant_conv(codebook_mapping_s)
        #codebook_mapping_t = self.vqgan_t.post_quant_conv(codebook_mapping_t)
        codebook_mapping_t=quant_conv_encoded_images_t
        #print(codebook_mapping_s.shape)
        b, c, h, w = codebook_mapping_s.shape
        #print(encoded_image_s.shape)
        dtype = codebook_mapping_s.dtype
        device = codebook_mapping_s.device

        mask = torch.zeros((b, h, w), dtype=torch.bool, device=device)
        src_features=codebook_mapping_t
        style_features = codebook_mapping_s
        style_mask= torch.zeros((b, h, w), dtype=torch.bool, device=device)
        #print(style_mask.shape)
        #print(style_mask)
        B, C, f_h, f_w = src_features.shape

        pos = self.position_embedding(NestedTensor(src_features, mask)).to(src_features.dtype)
        style_pos = self.position_embedding(NestedTensor(style_features, style_mask)).to(style_features.dtype)

        assert mask is not None

        hs, mem = self.transformer(self.input_proj_s(style_features), style_mask, self.input_proj_c(src_features), pos,
                                   style_pos)  # hs: [6, 2, 100,

        B, h_w, C = hs[-1].shape  # [B, h*w=L, C]
        hs = hs[-1].permute(0, 2, 1).reshape(B, C, f_h, f_w)  # [B,C,h,w]

        res = self.output_proj(hs)  # [B,256*k*k,h*w=L]   L=[(H − k + 2P )/S+1] * [(W − k + 2P )/S+1]  k=16,P=2,S=32

        res = self.tail(res)  # [B,3,H,W]
        #print(res.shape)

        return res

class ResBlock(nn.Module):
    """
    Usual full pre-activation ResNet bottleneck block.
    For more information see
    He, K., Zhang, X., Ren, S., & Sun, J. (2016, October).
    Identity mappings in deep residual networks.
    European Conference on Computer Vision (pp. 630-645).
    Springer, Cham.
    ArXiv link: https://arxiv.org/abs/1603.05027
    """
    def __init__(self, outer_dim, inner_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(outer_dim),
            nn.LeakyReLU(),
            nn.Conv2d(outer_dim, inner_dim, 1),
            nn.BatchNorm2d(inner_dim),
            nn.LeakyReLU(),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.BatchNorm2d(inner_dim),
            nn.LeakyReLU(),
            nn.Conv2d(inner_dim, outer_dim, 1),
        )

    def forward(self, input):
        return input + self.net(input)



