# MIT License
# 
# Copyright (c) 2023 Rafael Redondo, Eurecat.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch
import torch.nn.functional as F
from torch.autograd import grad
from utils import range01
from collections import namedtuple
from torchvision import models
from math import exp

"""
This file contains various loss functions, namely:

Classes:
    - Losses: A class that encapsulates various loss functions including L1, L2, BCE, perceptual loss, SSIM, and more.

Methods:
    - MaskedL1: Computes the L1 loss with a mask.
    - KLDiv: Computes the Kullback-Leibler divergence.
    - TV: Computes the Total Variation.
    - MSE_PSNR: Computes the Mean Squared Error and Peak Signal-to-Noise Ratio.
    - SSIM: Computes the Structural Similarity Index.
"""


class Loss():
    def __init__(self,
                 type       = 'wgan-gp',    # loss type in 'wgan-gp', 'wgan', 'hinge', 'ns', 'ls', or 'bce'
                 lambda_gp  = 10.0,         # gradient penalty weight
                 load_pl    = False,
                 device     = 'cpu'
                 ) -> None:

        self.type = type.lower()
        if not any(self.type == s for s in ['wgan-gp', 'wgan', 'hinge', 'ns', 'ls', 'bce']):
            raise ValueError("Unknown loss type")

        print(f'Loss type: {self.type}')
        self.eps = 1e-8

        # Gradient penalty
        self.gp_iters = 0
        self.gp_skips = 1   # 1 no-skips
        self.lambda_gp = lambda_gp
        self.gp = torch.zeros(1)

        # Auxiliary losses
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()
        if self.type == 'bce':
            self.criterion_BCE = torch.nn.BCEWithLogitsLoss()

        self.device = device
        # Perceptual losses
        if load_pl:
            self.vgg = Vgg16(requires_grad=False).to(device).eval()

        # SSIM window
        sigma=1.5
        self.channels = 3
        win_size = 11
        self.pad = win_size//2
        gauss = torch.Tensor([exp(-(x - self.pad)**2/float(2*sigma**2)) for x in range(win_size)])
        gauss = gauss/gauss.sum()
        win_1D = gauss.unsqueeze(1)
        win_2D = win_1D.mm(win_1D.t()).float().unsqueeze(0).unsqueeze(0)
        self.ssim_win = win_2D.expand(self.channels, 1, win_size, win_size).contiguous()

    def MaskedL1(self, seq1, seq2, mask):
        mask = mask.expand(-1, seq1.size(1), -1, -1)
        return self.criterionL1(seq1 * mask, seq2 * mask)

    @staticmethod
    def KLDiv(mu, log_var):
        # Auto-Encoding Variational Bayes https://arxiv.org/pdf/1312.6114.pdf
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

    @staticmethod
    @torch.no_grad()
    def MSE_PSNR(seq1, seq2, max_square=1.0):
        mse = torch.mean((range01(seq1) - range01(seq2)) ** 2)
        return mse, 10 * torch.log10(max_square / mse)
    
    @torch.no_grad()
    def SSIM(self, seq1, seq2):
        # Adapted from https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
        size1 = len(list(seq1.shape))
        assert size1 >= 4, f'SSIM, unsupported input size {size1}D'
        assert seq1.shape[-3] == self.channels, f'SSIM, input channels differ {seq1.shape[-3]} from {self.channels}'

        def _ssim(img1, img2, window):
            mu1 = F.conv2d(img1, window, padding=self.pad, groups=self.channels)
            mu2 = F.conv2d(img2, window, padding=self.pad, groups=self.channels)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.conv2d(img1*img1, window, padding=self.pad, groups=self.channels) - mu1_sq
            sigma2_sq = F.conv2d(img2*img2, window, padding=self.pad, groups=self.channels) - mu2_sq
            sigma12   = F.conv2d(img1*img2, window, padding=self.pad, groups=self.channels) - mu1_mu2

            C1 = 0.01**2
            C2 = 0.03**2

            ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()
            
        window = self.ssim_win.type_as(seq1).to(seq1.device)
        seq1 = range01(seq1)
        seq2 = range01(seq2)
        if size1 == 5:
            ssim = 0
            for l in range(seq1.shape[1]):
                img1 = torch.select(seq1, 1, l)
                img2 = torch.select(seq2, 1, l)
                ssim += _ssim(img1, img2, window)
            return ssim / float(seq1.shape[1])
        else:
            return _ssim(seq1, seq2, window)

    def criterionGAN_G(self, fake_preds):
    
        if 'wgan' in self.type:
            return -torch.mean(fake_preds)
        elif 'hinge' == self.type:
            return -torch.mean(fake_preds)
        elif 'ns' == self.type:
            return -torch.mean(torch.log(fake_preds + self.eps))
        elif 'ls' == self.type:
            return 0.50 * torch.mean((fake_preds - 1.)**2)
        elif 'bce' == self.type:
            ones = torch.ones(fake_preds.size(), dtype=torch.float).cuda()
            return self.criterion_BCE(fake_preds, ones)

    def criterionGAN_D(self, fake_preds, real_preds, disc=None, samples=None):
    
        if 'wgan' in self.type:
            
            # Wasserstein loss
            loss = torch.mean(fake_preds) - torch.mean(real_preds)

            # ## Epsilon penalty: prevents distancing from 0 (proposed in the original PGAN)
            # epsilon_penalty = 0.001 * (real_preds ** 2)
            # self.ep = epsilon_penalty.mean()
            # loss += self.ep

            # Gradient penalty
            # Skip some gradient penalty iterations to save time without impairing performance too much
            # Karras, Tero, et al. "Analyzing and improving the image quality of stylegan." Procc. of the IEEE/CVF Conf. on CVPR. 2020.
            self.gp_iters += 1
            if 'gp' in self.type and (self.gp_iters % self.gp_skips == 0):

                fake_samples, real_samples = samples
                bsize = real_samples.size(0)
                size = [bsize] + [1] * (len(real_samples.shape)-1) # [batch size,1,1,..,1]
                device = real_samples.device

                # fake_samples must be already detached
                alpha = torch.rand(size, requires_grad=True).to(device)
                x_hat = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
                x_hat_preds = disc(x_hat)

                gradients = grad(outputs=x_hat_preds,
                                inputs=x_hat,
                                grad_outputs=torch.ones_like(x_hat_preds),
                                create_graph=True,
                                retain_graph=True)[0]
                gradients = gradients.contiguous().view(bsize, -1)
                # Derivatives of the gradient close to 0 can cause problems because of
                # the square root, so manually calculate norm and add epsilon
                # https://github.com/EmilienDupont/wgan-gp/blob/ef82364f2a2ec452a52fbf4a739f95039ae76fe3/training.py
                # gradient_penalty = 10 * ((gradients.norm(2, dim=1) - 1) ** 2)     
                gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
                gp = self.gp_skips * self.lambda_gp * ((gradients_norm - 1) ** 2).mean()
                self.gp = gp.detach().clone()  # detach+copy to free the retained graph after this scope
                loss += gp

            return loss

        elif 'hinge' == self.type:
            return torch.mean(F.relu(1. - real_preds)) + torch.mean(F.relu(1. + fake_preds))

        elif 'ns' == self.type:
            return -torch.mean(torch.log(real_preds + self.eps) + torch.log(1 - fake_preds + self.eps))
    
        elif 'ls' == self.type:
            return (0.50 * torch.mean((real_preds - 1.)**2)) + (0.50 * torch.mean((fake_preds - 0.)**2))

        elif 'bce' == self.type:
            zeros = torch.zeros(fake_preds.size(), dtype=torch.float).cuda()
            ones = torch.ones(real_preds.size(), dtype=torch.float).cuda()
            return self.criterion_BCE(real_preds, ones) + self.criterion(fake_preds, zeros)

    @staticmethod
    def TV(img):
        w_var = torch.sum(torch.pow(img[...,:,:-1] - img[...,:,1:], 2))
        h_var = torch.sum(torch.pow(img[...,:-1,:] - img[...,1:,:], 2))
        return h_var + w_var

    def perceptual(self, reals, fakes):
        if len(reals.size()) == 5:
            C, H, W = reals.size()[2:]
            reals = reals.contiguous().view(-1, C, H, W)
            fakes = fakes.contiguous().view(-1, C, H, W)
        reals = reals.to(self.device)
        fakes = fakes.to(self.device)
        # Adapted from the official pytorch example neural_style.py
        # normalize using imagenet mean and std, make sure they are RGB mode
        mean = reals.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = reals.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        reals = (range01(reals) - mean) / std
        fakes = (range01(fakes) - mean) / std
        with torch.no_grad():
            feat_reals = self.vgg(reals)
        feat_fakes = self.vgg(fakes)
        return self.criterionL2(feat_reals, feat_fakes)
        
class Vgg16(torch.nn.Module):

    def __init__(self,
                requires_grad=False,
                n_layers=2              # layer= 1-'relu1_2', 2-'relu2_2', 3-'relu3_3', 4-'relu4_3'
                ):
        super(Vgg16, self).__init__()
        # ranges = {'relu1_2':[0,4], 'relu2_2':[4,9], 'relu3_3':[9,16], 'relu4_3':[16,23]}
        ranges = [0,4,9,16,23]
        vgg_features = models.vgg16(pretrained=True).features
        self.module = torch.nn.ModuleList([vgg_features[x] for x in range(0, ranges[n_layers])])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        for layer in self.module:
            x = layer(x)
        return x