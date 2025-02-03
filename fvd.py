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

from typing import Tuple
import scipy
import numpy as np
import torch
import torch.nn.functional as F
from dnnlib.util import open_url

class FVD():
    '''This class implements the Frechet Video Distance borrowing code from
     https://github.com/cvpr2022-stylegan-v/fvd-comparison
    '''
    def __init__(self,
                source_samples: torch.Tensor,           # batch of videos [B,T,3,H,W]
                max_fakes: int                  = 16,   # max fake videos to calculate metric
                device: str                     ='cpu'  # input sa
        ) -> None:

        self.fakes = []
        self.max_fakes = max_fakes
        self.batch_size = 16        # batch size for processing purposes
        self.device = device
        detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
        self.detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.

        with open_url(detector_url, verbose=False) as f:
            self.detector = torch.jit.load(f).eval().to(device)

        feats_real = self.get_feats(source_samples.to(device))
        self.mu_real, self.sigma_real = self.compute_stats(feats_real)

    @torch.no_grad()
    def get_feats(self, samples: torch.Tensor) -> torch.Tensor:
        feats = []
        for i in range(0, samples.size(0), self.batch_size):
            input = torch.permute(samples[i:i+self.batch_size], (0,2,1,3,4)) # channel fist, video length 
            T, H, W = input.shape[-3:]
            if H != 224 or W != 224:
                input = F.interpolate(input, size=(T,224,224))
            feats_batch = self.detector(input, **self.detector_kwargs)
            feats.append(feats_batch.cpu().numpy())
        return np.concatenate(feats, axis=0)

    def add_fake(self, fake: torch.Tensor):
        self.fakes.append(fake.to(self.device))
        if len(self.fakes) > self.max_fakes:
            self.fakes.pop(0)

    def compute_fvd(self) -> float:

        if not len(self.fakes):
            RuntimeWarning('Insufficient samples to calculate FVD.')
            return 0

        fakes = torch.cat(self.fakes, dim=0)
        feats_fake = self.get_feats(fakes)
        mu_gen, sigma_gen = self.compute_stats(feats_fake)

        m = np.square(mu_gen - self.mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, self.sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + self.sigma_real - s * 2))

        return float(fid)

    def compute_stats(self, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = feats.mean(axis=0)             # [d]
        sigma = np.cov(feats, rowvar=False) # [d, d]
        return mu, sigma
