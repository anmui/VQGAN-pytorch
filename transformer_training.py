import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
import util
import utils
from Criterion import SetCriterion
from discriminator import Discriminator
from lpips import LPIPS
from transformerr import Transformerr
from vqgan import VQGAN
from utils import load_data, weights_init
from fcn import FCN



class TrainT:
    def __init__(self, args):
        weight_dict = {'loss_content': args.content_loss_coef, 'loss_style': args.style_loss_coef,
                       'loss_tv': args.tv_loss_coef}
        losses = ['content', 'style', 'tv']
        self.transformer = Transformerr(args).to(device=args.device)
        self.criterion = SetCriterion(1, weight_dict=weight_dict,
                                 eos_coef=args.eos_coef, losses=losses)
        self.criterion.to(device=args.device)
        self.opt_t = self.configure_optimizers(args)
        self.prepare_training()

        self.train(args)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_t = torch.optim.Adam(
            list(self.transformer.transformer.parameters()) +
            list(self.transformer.tail.parameters()) +
            list(self.transformer.output_proj.parameters()) ,
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )


        return opt_t

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def train(self, args):
        train_loader_s,train_loader_t = utils.load_data_2(args)
        for epoch in range(args.epochs):
            with tqdm(range(len(train_loader_t))) as pbar:
                for i, samples,style_images in zip(pbar, train_loader_t,train_loader_s):
                    samples = samples.to(device=args.device)
                    style_images = style_images.to(device=args.device)
                    outputs = self.transformer(samples, style_images)

                    loss_dict = self.criterion(outputs, samples, style_images)
                    weight_dict = self.criterion.weight_dict
                    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                    # reduce losses over all GPUs for logging purposes
                    loss_dict_reduced = util.misc.reduce_dict(loss_dict)
                    loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                                  for k, v in loss_dict_reduced.items()}
                    loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                                for k, v in loss_dict_reduced.items() if k in weight_dict}
                    losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

                    loss_value = losses_reduced_scaled.item()
                    self.opt_t.zero_grad()
                    losses.backward()
                    self.opt_t.step()

                    if i % 10 == 0:
                        with torch.no_grad():
                            real_fake_images = torch.cat((samples[:4].add(1).mul(0.5)[:4], outputs.add(1).mul(0.5)[:4],style_images.add(1).mul(0.5)[:4]))
                            vutils.save_image(real_fake_images, os.path.join("results", f"1_{epoch}_{i}_fix.jpg"), nrow=4)

                    pbar.set_postfix(
                        t_Loss=np.round(loss_value, 5)
                    )
                    pbar.update(0)
                torch.save(self.transformer.state_dict(), os.path.join("checkpoints", f"transformer_epoch_{epoch}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=1, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    #################### jianbo
    parser.add_argument('--model_type', default='nofold', type=str,
                        help="type of model")
    parser.add_argument('--fold_k', default=8, type=int,
                        help="Size of fold kernels")
    parser.add_argument('--fold_stride', default=6, type=int,
                        help="Size of fold kernels")
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--enorm', action='store_true')
    parser.add_argument('--dnorm', action='store_true')
    parser.add_argument('--tnorm', action='store_true')
    parser.add_argument('--model_pre', action='store_true')
    parser.add_argument('--cbackbone_layer', type=int, default=4,
                        help="")
    parser.add_argument('--sbackbone_layer', type=int, default=4,
                        help="")
    # * Loss coefficients

    parser.add_argument('--content_loss_coef', default=1.0, type=float)
    parser.add_argument('--style_loss_coef', default=1e4, type=float)
    parser.add_argument('--tv_loss_coef', default=0, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    args = parser.parse_args()
    args.dataset_path_s = [r"/media/lab/sdb/zzc/zhangdaqian",r"/media/lab/sdb/zzc/A"]
    args.dataset_path_t = [r"/media/lab/sdb/zzc/B"]
    args.checkpoint_path_style=r"/media/lab/sdb/zzc/myVQGAN/checkpoints/vqgan_epoch_99.pt"
    args.checkpoint_path_ture = r"/media/lab/sdb/zzc/myVQGAN/checkpoints/vqganB_epoch_99.pt"
    train_t = TrainT(args)



