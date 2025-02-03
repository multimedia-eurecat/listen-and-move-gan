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
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

"""
This file implements a ConvGRU Cell with bidirectional kernels.

Usage:
    - DirConv2d can be used as a drop-in replacement for nn.Conv2d in PyTorch.
    - It supports three types of kernels: vertical, horizontal, and regular (no direction).
    - Example:
        conv = DirConv2d(in_channels=3, out_channels=64, kernel_size=3, orientation='h')
        output = conv(input_tensor, direction='forward')

For more details, see https://arxiv.org/pdf/2406.16155
"""


class DirConv2d(nn.Module):
    def __init__(self,
                in_channels,        # input channels
                out_channels,       # output channels
                kernel_size,        # kernel size (single value)
                orientation = None  # kernel orientation: 'h' height/vertical, 'w' width/horizontal, or None for no direction (square shape).
                ) -> None:
        super().__init__()
        assert kernel_size % 2 != 0, 'Kernel size must be odd.'
        assert orientation in ['h', 'w', None], 'invalid kernel orientation.'
        self.orientation = orientation

        # No orientation, regular conv
        if self.orientation is None:
            pad = kernel_size // 2
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad)
            nn.init.orthogonal_(self.conv.weight)
            nn.init.constant_(self.conv.bias, 0.)
            return

        # Directional convolution
        self.convs = nn.ModuleList()
        for s in range(0, kernel_size, 2):
            ks = s+1
            if orientation == 'h':      # Along height/vertical orientation
                ksize = [1, ks]
                pad = [0, ks//2]
            elif orientation == 'w':    # Along width/horizontal orientation
                ksize = [ks, 1]
                pad = [ks//2, 0]

            conv = nn.Conv2d(in_channels, out_channels, ksize, padding=pad, bias=False)
            nn.init.orthogonal_(conv.weight)    # init weights, no bias            
            self.convs.append(conv)
        
        self.bias = nn.Parameter(torch.zeros(1,out_channels,1,1), requires_grad=True)
    
    def forward(self, input, direction):

        # No orientation
        if self.orientation is None:
            return self.conv(input)

        # Directional kernels
        for stride, conv in enumerate(self.convs):
            
            x = conv(input)
            
            if stride == 0:
                    output = x
            else:
                if self.orientation == 'h':
                    if direction == 1:
                        output = output + F.pad(x[...,:-stride,:],(0,0,stride,0),'replicate')
                    elif direction == -1:
                        output = output + F.pad(x[...,stride:,:],(0,0,0,stride),'replicate')
                elif self.orientation == 'w':
                    if direction == 1:
                        output = output + F.pad(x[...,:,:-stride],(stride,0,0,0),'replicate')
                    elif direction == -1:
                        output = output + F.pad(x[...,:,stride:],(0,stride,0,0),'replicate')

        return output + self.bias

class DirConvGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, orientation):
        super().__init__()

        self.hidden_size = hidden_channels
        self.reset  = DirConv2d(in_channels + hidden_channels, hidden_channels, kernel_size, orientation)
        self.update = DirConv2d(in_channels + hidden_channels, hidden_channels, kernel_size, orientation)
        self.output = DirConv2d(in_channels + hidden_channels, hidden_channels, kernel_size, orientation)
    
    def forward(self, inputs, hidden=None, direction=None):

        # if provided no hidden, generate empty prev_state [B C H W]
        if hidden is None:
            spatial_size = inputs.data.size()[2:]
            state_size = [1, self.hidden_size] + list(spatial_size)
            hidden = torch.zeros(state_size).to(inputs.device)

        # batch forward pass
        out_states = []
        for t in range(inputs.size(0)):
            input = inputs[t:t+1,...]
            stacked_inputs = torch.cat([input, hidden], dim=1)
            z = torch.sigmoid(self.update(stacked_inputs, direction))
            r = torch.sigmoid(self.reset(stacked_inputs, direction))
            h = torch.tanh(self.output(torch.cat([input, hidden * r], dim=1), direction))
            hidden = hidden * (1 - z) + h * z
            out_states.append(hidden)

        return torch.cat(out_states, dim=0)
