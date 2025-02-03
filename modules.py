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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import flip as tflip
from convgru import DirConvGRUCell
arrow="\u2192"


"""
This file contains various neural network modules, namely:

Classes:
    - GenNet: The generator network for the GAN.
    - Discriminator: Image and Video discriminator network for the GAN.
    - DirConvGRUCell: A GRU cell with directional convolutions.
"""

def check_param(param, values, message):
    if not any(param == v for v in values):
        raise ValueError(message)

def calculate_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

def init_weights(m, init_type='normal', init_gain=0.02):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        if init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0, init_gain)  # Opt. CAR-GAN 0, 0.2
        elif init_type == 'xavier':
            nn.init.xavier_normal_(m.weight.data, gain=init_gain)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight.data, gain=init_gain)
        else:
            raise NotImplementedError(f'Initialization method "{init_type}" not implemented')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight.data, 1.0, init_gain)
        nn.init.constant_(m.bias.data, 0.0)

class Conv(nn.Module):
    def __init__(self, type, in_channels, out_channels, kernel_size=3, stride=1, padding=1, equalized=False, **kwargs) -> None:
        super().__init__()
        if '2d' in type.lower():
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, **kwargs)
        elif '3d' in type.lower():
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, **kwargs)

        # self.conv = nn.utils.spectral_norm(self.conv)

        self.equalized = equalized
        if equalized:
            if '2d' in type.lower():
                self.F_conv = F.conv2d
            elif '3d' in type.lower():
                self.F_conv = F.conv3d
            fan_in = math.prod(self.conv.weight.data.size()[1:])
            self.scale = math.sqrt(2) / math.sqrt(fan_in)

    def forward(self, input):
        if self.equalized:
            return self.F_conv(
                input=input,
                weight=self.conv.weight * self.scale,
                bias=self.conv.bias,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=self.conv.groups
            )
        else:
            return self.conv(input)

class BatchNorm(nn.Module):
    def __init__(self, type, n_channels, **kwargs) -> None:
        super().__init__()
        if '2d' in type.lower():
            self.module = nn.BatchNorm2d(n_channels, **kwargs)
        elif '3d' in type.lower():
            self.module = nn.BatchNorm3d(n_channels, **kwargs)

    def forward(self, input):
        return self.module(input)

class InstanceNorm(nn.Module):
    def __init__(self, type, n_channels, **kwargs) -> None:
        super().__init__()
        if '2d' in type.lower():
            self.module = nn.InstanceNorm2d(n_channels, **kwargs)
        elif '3d' in type.lower():
            self.module = nn.InstanceNorm3d(n_channels, **kwargs)

    def forward(self, input):
        return self.module(input)

class ConditionalInstanceNorm(nn.Module):
    def __init__(self, type, n_channels) -> None:
        super().__init__()
        if type == '3d':
            NotImplementedError('Currently unsupported 5D input vectors.')

    def forward(self, input, mu_std, eps=1e-05):
        mean = torch.mean(input, dim=(-2,-1), keepdim=True)
        var = torch.var(input, dim=(-2,-1), unbiased=False, keepdim=True)
        input_norm = (input - mean) / torch.sqrt(var + eps)
        B, C = input.shape[:2]
        size = mu_std.shape[1] // 2
        mean_cond = mu_std[:,:size].view(B,C,1,1)
        std_mean = mu_std[:,size:].view(B,C,1,1)
        return input_norm * std_mean + mean_cond

class PixelNorm(nn.Module):
    def __init__(self, type=None, n_channels=None):
        super().__init__()

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class ResConvBlock(nn.Module):
    def __init__(self, type, in_channels, out_channels, noise=False, norm='batch') -> None:
        super().__init__()

        if norm == 'batch':
            NormLayer = BatchNorm
        elif norm == 'instance':
            NormLayer = InstanceNorm
        elif norm == 'pixel':
            NormLayer = PixelNorm
        elif norm == 'condIN':
            NormLayer = ConditionalInstanceNorm

        if noise:
            modules = nn.ModuleList([
                Conv(type, in_channels, out_channels),
                NoiseInjection(),
                NormLayer(type, out_channels),
                nn.LeakyReLU(0.2),
                Conv(type, out_channels, out_channels),
                NoiseInjection(),
                NormLayer(type, out_channels)
            ])
        else:
            modules = nn.ModuleList([
                Conv(type, in_channels, out_channels),
                NormLayer(type, out_channels),
                nn.LeakyReLU(0.2),
                Conv(type, out_channels, out_channels),
                NormLayer(type, out_channels)
            ])

        self.block = modules
        # self.block = nn.Sequential(*modules)
        self.residual = Conv(type, in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.activation = nn.LeakyReLU(0.2)
        self.res_norm = 1 / math.sqrt(2)

    def forward(self, input, cond_norm=None):
        x = input
        for module in self.block:
            if isinstance(module, ConditionalInstanceNorm):
                x = module(x, cond_norm)
            else:
                x = module(x)

        return self.activation((x + self.residual(input)) * self.res_norm)

class ConvBlock(nn.Module):
    def __init__(self, type, in_channels, out_channels, noise=False, norm='batch') -> None:
        super().__init__()

        if norm == 'batch':
            NormLayer = BatchNorm
        elif norm == 'instance':
            NormLayer = InstanceNorm
        elif norm == 'pixel':
            NormLayer = PixelNorm
        else:
            NotImplementedError(f'Unsupported {norm} type for regular conv blocks.')

        if noise:
            self.module = nn.Sequential(
                Conv(type, in_channels, out_channels),
                NoiseInjection(),
                NormLayer(type, out_channels),
                nn.LeakyReLU(0.2),
            )
        else:
            self.module = nn.Sequential(
                Conv(type, in_channels, out_channels),
                NormLayer(type, out_channels),
                nn.LeakyReLU(0.2),
            )

    def forward(self, input):
        return self.module(input)

class Upsample(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.type = type.lower()

    def forward(self, input):
        # upsample only the last two spatial dimensions
        shape = list(input.shape)
        if '2d' in self.type:
            size = [shape[-2] * 2, shape[-1] * 2]
        elif '3d' in self.type:
            size = [shape[-3], shape[-2] * 2, shape[-1] * 2]
        return F.interpolate(input, size=size, mode='nearest')

class AvgPool(nn.Module):
    def __init__(self, type, keep_depth=False) -> None:
        super().__init__()
        if '2d' in type.lower():
            self.module = nn.AvgPool2d((2,2))
        elif '3d' in type.lower():
            kernel_size = (1, 2, 2) if keep_depth else (2, 2, 2)
            self.module = nn.AvgPool3d(kernel_size)

    def forward(self, input):
        return self.module(input)

class Replicate(nn.Module):
    def __init__(self, type, pad) -> None:
        super().__init__()
        if '2d' in type.lower():
            self.replicate = nn.ReplicationPad2d((pad, pad-1, pad, pad-1))
        elif '3d' in type.lower():
            self.replicate = nn.ReplicationPad3d((pad, pad-1, pad, pad-1, 0, 0))

    def forward(self, input):
        return self.replicate(input)

class NoiseInjection(nn.Module):
    def __init__(self, size=1):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size))

    def forward(self, input):
        if input.dim() == 3:
            B, L, C = input.shape
            noise = input.new_empty(B, L, C).normal_()
        elif input.dim() == 4:
            B, _, H, W = input.shape
            noise = input.new_empty(B, 1, H, W).normal_()
        elif input.dim() == 5:
            B, _, L, H, W = input.shape
            noise = input.new_empty(B, 1, L, H, W).normal_()
        else:
            raise Exception('Noise Injection currently supports only 4D-5D tensors.')
        return input + self.weight * noise

# ----------------------------------------------------------------------------------------------

class MotionEncoder(nn.Module):
    def __init__(self,
                 input_size,                # size of input tensors
                 type           = 'basic',  # basic or feedback (Ref:Sound-guided Semantic Video Generation).
                 layers         = 3,        # number of layers
                 dilation       = 2         # dilation factor (dilation)**layer
                 ) -> None:
        super().__init__()
        print(f'[Motion Encoder]')

        self.type = type.lower()
        self.dilation = dilation

        # Residual connection
        self.activation = nn.LeakyReLU(0.2)
        self.res_norm = 1 / math.sqrt(2)

        # Basic sound feature linear encoder
        if 'back' in self.type:
            self.sound_encoder = nn.Sequential(
                    nn.Linear(input_size, input_size),
                    nn.LeakyReLU(0.2))
            self.apply(init_weights)                # init before RNN layers

        # Default GRU initialization https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
        # hidden state h_0 and cell state c_0 init as zeros by default 
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.GRUCell(input_size=input_size, hidden_size=input_size))

        self.reset_train()
        self.reset_eval()

        print(f'    input size......... {input_size}')
        print(f'    output size........ {input_size}')
        print(f'    type............... {type}')
        print(f'    layers............. {layers}')
        print(f'    dilation........... {dilation}')
        calculate_model_size(self)
            
    def reset_train(self):
        self.hidden_train = []
        for l in range(len(self.layers)):
            self.hidden_train.append([None for _ in range(self.dilation**l)])

    def reset_eval(self):
        self.hidden_eval = []
        for l in range(len(self.layers)):
            self.hidden_eval.append([None for _ in range(self.dilation**l)])

    def forward(self, input):
        hidden = self.hidden_train if self.training else self.hidden_eval
        input = input.squeeze(0)    # assuming batch size 1
        outputs = []

        for t in range(input.shape[0]):
            
            if 'back' in self.type:
                h_in = self.sound_encoder(input[t])
                if len(outputs):
                    h_in += outputs[-1]
                elif hidden[-1][-1] is not None:
                    h_in += hidden[-1][0]
            else:
                h_in = input[t]

            for GRUCell, h in zip(self.layers, hidden):
                h_out = GRUCell(h_in, h[0])
                h.append(h_out)
                h.pop(0)
                if len(self.layers) > 1:    # residual connection
                    h_in = self.activation((h_out + h_in) * self.res_norm)
                else:
                    h_in = h_out
            
            outputs.append(h_in)

            if self.training:
                self.hidden_train = [[h.detach() if torch.is_tensor(h) else h for h in hl] for hl in hidden]
            else:
                self.hidden_eval = [[h.detach() if torch.is_tensor(h) else h for h in hl] for hl in hidden]

        return torch.stack(outputs, dim=0).unsqueeze(0)

# ----------------------------------------------------------------------------------------------

class Generator(nn.Module):
    def __init__(self,
                 input_depth,                   # Depth of forward tensor as [B, depth, 1, 1]
                 output_size,                   # Size of the generated image (assumed square)
                 cond_size          = 0,        # Size of conditional feature vectors
                 gen_type           = '2d',     # Generator type: '2d' or '3d' (incompatible with PixelShuffle)
                 max_channels       = 512,      # Maximum number of feature channels per layer
                 channels_base      = 1024,     # From which to calculate a decreasing number of channels per layer
                 vid_pred           = '',       # Video prediction: None (empty), basic, or dir (directional).
                 double_finest      = False,    # If True, capacity of finest (outer) layers is doubled
                 architecture       = 'skip'    # 'basic', 'skip', or 'residual'
                ) -> None:
        super().__init__()
        print(f'[Generator]')
        check_param(gen_type, ['2d','3d'], 'Invalid generator type')
        check_param(architecture, ['basic', 'skip', 'residual'], 'Invalid generator architecture.')
        if cond_size > 0 and architecture != 'residual':
            RuntimeWarning('Unsupported normalization type for current architecture, switching to batch norm.')
        if vid_pred and architecture == 'skip':
            raise NotImplementedError(f'Currently unsupported video prediction for skip connections.')
        if vid_pred and gen_type == '3d':
            raise NotImplementedError(f'Currently unsupported video prediction for 3D generator.')
        
        self.output_size = output_size
        input_size = 2**2
        n_layers = int(math.log2(output_size) - math.log2(input_size))

        def nc(stage):
            m = 2.0 if (n_layers - stage) < 2 and double_finest else 1.0
            return min(int(m * channels_base // 2.0**stage), max_channels)

        # Format layer
        pad = input_size // 2
        self.format_layer = nn.Sequential(
            Replicate(gen_type, pad),
            ConvBlock(gen_type, input_depth, nc(0), noise=True))

        # Upsamplings
        self.upsample = Upsample(gen_type)

        # Generative layers
        self.architecture = architecture
        self.layers = nn.ModuleList()
        if self.architecture == 'residual':
            norm_type = 'condIN' if cond_size > 0 else 'batch'
            for l in range(n_layers):
                self.layers.append(ResConvBlock(gen_type, nc(l), nc(l+1), noise=True, norm=norm_type))
        else:
            for l in range(n_layers):
                self.layers.append(nn.Sequential(
                    ConvBlock(gen_type, nc(l),   nc(l+1), noise=True),
                    ConvBlock(gen_type, nc(l+1), nc(l+1), noise=True)))

        # Conditions (modulation) by using layer normalization
        self.conditions = nn.ModuleList()
        if cond_size > 0:
            for l in range(n_layers):
                self.conditions.append(nn.Sequential(
                    nn.Linear(cond_size, nc(l+1) * 2),
                    nn.LeakyReLU(0.2)))

        # To RGB Convert layers
        if self.architecture == 'skip':
            self.toRGB = nn.ModuleList()
            for l in range(n_layers):
                self.toRGB.append(nn.Sequential(
                    Conv(gen_type, nc(l+1), 3, kernel_size=1, padding=0),
                    nn.Tanh()))
        else:
            self.toRGB = nn.Sequential(
                Conv(gen_type, nc(n_layers), 3, kernel_size=1, padding=0),
                nn.Tanh())

        self.apply(init_weights)

        # Video prediction
        self.vp_layers = nn.ModuleDict()
        if vid_pred:
            kernel_size = 3 if output_size < 512 else 5
            # Add video prediction layers by inserting indexes
            vp_indexes = [n_layers-1]
            for l in vp_indexes:
                self.vp_layers[str(l)] = VideoPredictor(nc(l+1), nc(l+1), nc(l+1), kernel_size, None, 'dir' in vid_pred)

            # this configuration can substitute the last VP layer in a more amenable way
            # however it is less robust against unseen audio input for conditional normalization 
            # self.vp_layers['rgb'] = VideoPredictor(nc(n_layers), nc(n_layers+1), 3, 5, 'tanh')

        print(f'    type............... {gen_type}')
        print(f'    architecture....... {architecture}')
        print(f'    format layer....... {input_depth}{"xL" if gen_type=="3d" else ""}x{input_size}x{input_size}')
        print(f'    generative layers.. {n_layers}')
        print(f'    layer channels..... {" | ".join([f"{nc(l)}{arrow}{nc(l+1)}" for l in range(n_layers)])}')
        print(f'    conditioned........ {len(self.conditions)}')
        print(f'    video prediction... {vid_pred if vid_pred else "False"}')
        print(f'    double finest...... {double_finest}')
        print(f'    output............. {3}x{output_size}x{output_size}')
        calculate_model_size(self)
        # print(self)

    def reset_train(self):
        for lname in self.vp_layers:
            self.vp_layers[lname].reset_train()

    def reset_eval(self):
        for lname in self.vp_layers:
            self.vp_layers[lname].reset_eval()

    def forward(self, input, feats):

        x = self.format_layer(input)

        if self.architecture == 'skip':

            for l, (layer, toRGB) in enumerate(zip(self.layers, self.toRGB)):

                x = self.upsample(x)
                x = layer(x)

                if l == 0:
                    rgb = toRGB(x)
                else:
                    shape = list(x.shape)
                    size = [shape[-2] * 2, shape[-1] * 2]
                    if len(shape) == 5: size.insert(0, shape[-3])  # only the last two spatial dims
                    rgb = toRGB(x) + F.interpolate(rgb, size=size, mode='nearest')
        else:

            for l, layer in enumerate(self.layers):
                
                x = self.upsample(x)

                if self.conditions:
                    cond_norm = self.conditions[l](feats)
                else: cond_norm = None
                
                x = layer(x, cond_norm)
                
                if str(l) in self.vp_layers:
                    x = self.vp_layers[str(l)](x, x)
            
            rgb = self.toRGB(x)
            
            if 'rgb' in self.vp_layers:
                rgb = self.vp_layers['rgb'](x, rgb)

            return rgb

# ----------------------------------------------------------------------------------------------

class VideoPredictor(nn.Module):
    '''Inspired from ContextVP PyraMiD-LSTM https://arxiv.org/pdf/1710.08518.pdf
        Efficient implementation of Weighted Blending and Directional Weight-Sharing.
    '''
    def __init__(self,
                 in_channels,               # number of input channels
                 hidden_channels,           # number of hidden channels (ideally hidden_channels == in_channels)
                 output_channels,           # number of output channels
                 kernel_size,               # size of convolutional kernel
                 activation = None,         # activation function of the last layer: tanh, relu, or None
                 directional = True,        # Uses directional kernels, otherwise a regular ConvGRU layer
                ) -> None:
        super().__init__()
        print(f'[VideoPrediction]')

        # Flow prediction layers
        if directional:
            self.flows = nn.ModuleList([
                DirConvGRUCell(in_channels, hidden_channels, kernel_size, None),
                DirConvGRUCell(in_channels, hidden_channels, kernel_size, 'h'),
                DirConvGRUCell(in_channels, hidden_channels, kernel_size, 'w')
            ])

            # Directional weighting: a mask is regressed in the last channel
            self.pmd_weighting = Conv('2d', 5 * hidden_channels, output_channels + 1, kernel_size=1, padding=0)

        else:
            self.flows = self.flows = nn.ModuleList([
                DirConvGRUCell(in_channels, output_channels + 1, kernel_size, None)
            ])

        # Activation: Tanh for RGB output (3 channels), otherwise ReLU or Identity (None)
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = None

        self.apply(init_weights)
        self.reset_train()
        self.reset_eval()

        print(f'    input channels..... {in_channels}')
        print(f'    hidden channels.... {hidden_channels}')
        print(f'    output channels.... {output_channels}')
        print(f'    activation......... {activation}')
        print(f'    directional........ {directional}')
        calculate_model_size(self)
        # print(self)

    def reset_train(self):
        self.hidden_train = [None, None, None, None, None] if len(self.flows) > 1 else None

    def reset_eval(self):
        self.hidden_eval = [None, None, None, None, None] if len(self.flows) > 1 else None

    def forward(self, input, output):

        hidden = self.hidden_train if self.training else self.hidden_eval

        if len(self.flows) > 1:

            # Calculate 5 flow equivalent directions [B x C_hidden x H x W]
            flow0 = self.flows[0](input, hidden[0], 0)
            flow1 = self.flows[1](input, hidden[1], 1)
            flow2 = self.flows[1](input, hidden[2],-1)
            flow3 = self.flows[2](input, hidden[3], 1)
            flow4 = self.flows[2](input, hidden[4],-1)

            # Concatenate flow predictions [B x 5 * C_hidden x H x W]
            flow_pmd = torch.cat([flow0, flow1, flow2, flow3, flow4], dim=1)

            # Aggregate weighted PyraMiD flows  [B x C_out x H x W]
            agg_pred = self.pmd_weighting(flow_pmd)

        else:
            agg_pred = self.flows[0](input, hidden, 0)
    
        # Channel prediction [B x C_out x H x W]
        pred = agg_pred[:,:-1,...]
        if self.activation:
            pred = self.activation(pred)

        # Mask regression [B x 1 x H x W]
        mask = torch.sigmoid(agg_pred[:,-1:,...])

        # Pixels-wise masking prediction + hallucinated
        output = output * mask + pred * (1.0 - mask)
        
        # Update hidden from the last output
        if len(self.flows) > 1:
            hidden = torch.split(flow_pmd[-1:].detach(), flow_pmd.size(1)//5, dim=1)
        else:
            hidden = agg_pred[-1:].detach()

        if self.training:
            self.hidden_train = hidden
        else:
            self.hidden_eval = hidden

        return output

# ----------------------------------------------------------------------------------------------

class GenNet(nn.Module):
    def __init__(self,
                 output_size,               # Size of the generated image (assumed square)
                 dim_audio_feat,            # Size of input vector (number of sound features)
                 dim_e_motion    = 0,       # Size of motion random vector (will feed the rnn)
                 dim_z_content   = 0,       # Size of content random vector to directly feed G
                 sound_route     = 'gen',   # Sound features feed 'rnn', 'gen' (generator) or 'rnngen' for both
                 motion_layers   = 1,       # Number of motion encoder layers.
                 motion_type     = 'basic', # basic or feedback (Ref:Sound-guided Semantic Video Generation).
                 gen_type        = '2d',    # Generator type: '2d' or '3d' (motion arguments ignored)
                 cond_gen        = False,   # If true, activates generator's conditional instance normalization.
                 **kwargs
                 ) -> None:
        super().__init__()
        check_param(sound_route, ['rnn', 'gen', 'rnngen'], 'Invalid sound route.')
        print(f'[Generative Network]')
        print(f'    sound route........ {sound_route}')
        print(f'    e motion noise..... {dim_e_motion}')
        print(f'    z content noise.... {dim_z_content}')

        self.gen_type      = gen_type
        self.sound_route   = sound_route
        self.output_size   = output_size
        self.dim_e_motion  = dim_e_motion
        self.dim_z_content = dim_z_content
        self.dim_rnn_in    = dim_audio_feat + dim_e_motion if 'rnn' in sound_route else self.dim_e_motion
        self.dim_rnn_out   = self.dim_rnn_in # * (motion_layers+1)
        self.dim_z         = self.dim_rnn_out + dim_audio_feat + dim_z_content if 'gen' in sound_route else self.dim_rnn_out + dim_z_content
        self.cond_size     = dim_audio_feat if cond_gen else 0

        self.motion        = MotionEncoder(self.dim_rnn_in, layers=motion_layers, type=motion_type)
        self.generator     = Generator(gen_type=self.gen_type, input_depth=self.dim_z, cond_size=self.cond_size, output_size=output_size, **kwargs)
    
    def reset_train(self):
        self.motion.reset_train()
        self.generator.reset_train()

    def reset_eval(self):
        self.motion.reset_eval()
        self.generator.reset_eval()
    
    def sample_z_content(self, batch_size, seq_len, device):
        return torch.rand((batch_size, seq_len, self.dim_z_content), device=device, requires_grad=True)

    def sample_e_motion(self, batch_size, seq_len, device):
        return torch.rand((batch_size, seq_len, self.dim_e_motion), device=device, requires_grad=True)

    def forward(self, audio_features):
        B, T, F = audio_features.size()

        # Content Noise and Motion recurrency
        z_content = self.sample_z_content(B, T, audio_features.device)
        e_motion = self.sample_e_motion(B, T, audio_features.device)

        if 'rnn' == self.sound_route:
            e_feat_motion = torch.cat((e_motion, audio_features), dim=2)
            z_feat_motion = self.motion(e_feat_motion)
            z = torch.cat((z_feat_motion, z_content), dim=2)
        elif 'gen' == self.sound_route:
            e_motion = self.motion(e_motion)
            z = torch.cat((e_motion, audio_features, z_content), dim=2)
        else: # 'rnngen'
            e_feat_motion = torch.cat((e_motion, audio_features), dim=2)
            z_feat_motion = self.motion(e_feat_motion)
            z = torch.cat((z_feat_motion, audio_features, z_content), dim=2)

        # Generator forward
        if self.gen_type == '2d':
            rgb = self.generator(z.contiguous().view(B*T, -1, 1, 1), audio_features.contiguous().view(B*T, -1))
            rgb = rgb.view(B, T, 3, self.output_size, self.output_size)
        else:
            rgb = self.generator(torch.permute(z,(0,2,1)).view(B, -1, T, 1, 1))
            rgb = torch.permute(rgb,(0,2,1,3,4))

        return rgb

# ----------------------------------------------------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self,
                 seq_len,                   # Video sequence length (1 default for images)
                 input_size,                # Size of input image (assumed square)
                 max_channels   = 512,      # Maximum number of feature channels per layer
                 channels_base  = 1024,     # From which to calculate a decreasing number of channels per layer (StyleGAN2 16384)
                 double_finest  = False,    # If True, capacity of finest (outer) layers is doubled
                 architecture   = 'skip',   # 'basic', 'skip' or 'residual'
                 store_latents  = False,    # If True, all intermediate activations are stored
                 ) -> None:
        super().__init__()
        print(f'[Discriminator]')

        assert seq_len > 0, 'Invalid sequence length'
        self.disc_type = '2d' if seq_len == 1 else '3d'
        self.architecture = architecture

        patch_size = 2**2
        n_layers = int(math.log2(input_size) - math.log2(patch_size))

        def nc(stage):
            m = 2.0 if (n_layers - stage) < 2 and double_finest else 1.0
            return min(int(m * channels_base//2.0**stage), max_channels)

        # Number of layers with depth > 1 (video sequence dim > 1)
        self.seq_layers = 0
        if self.disc_type == '3d':
            assert seq_len % 2 == 0, 'Only even length sequences supported.'
            self.seq_layers = min(self.__div2_times__(seq_len), n_layers)
        
        # From RGB Convert layers
        if self.architecture == 'skip':
            self.fromRGB = nn.ModuleList()
            for l in reversed(range(n_layers)):
                self.fromRGB.append(Conv(self.disc_type, 3, nc(l+1), kernel_size=1, padding=0))
        else:
            self.fromRGB = Conv(self.disc_type, 3, nc(n_layers), kernel_size=1, padding=0)

        # Discriminative layers
        self.layers = nn.ModuleList()
        if self.architecture == 'residual':
            for l in reversed(range(n_layers)):
                keep_depth = l < n_layers - self.seq_layers
                self.layers.append(nn.Sequential(
                    AvgPool(self.disc_type, keep_depth=keep_depth),
                    ResConvBlock(self.disc_type, nc(l+1), nc(l))
                ))
        else:
            for l in reversed(range(n_layers)):
                keep_depth = l < n_layers - self.seq_layers
                self.layers.append(nn.Sequential(
                    AvgPool(self.disc_type, keep_depth=keep_depth),
                    ConvBlock(self.disc_type, nc(l+1), nc(l)),
                    ConvBlock(self.disc_type, nc(l), nc(l))
                ))

        # Patch GAN: may need an activation function for other than Wasserstein loss
        self.decision_layer = Conv(self.disc_type, nc(0), 1, kernel_size=1, padding=0)

        self.apply(init_weights)

        self.latents = {}
        if store_latents:
            def get_activation(name):
                def hook(model, input, output):
                    self.latents[name] = output.detach()
                return hook
            for l, layer in enumerate(self.layers):
                layer.register_forward_hook(get_activation(f'layer{l}'))

        out_width = input_size//2**n_layers
        out_depth = seq_len//2**self.seq_layers
        print(f'    type............... {self.disc_type}')
        print(f'    sequence len....... {seq_len}')
        print(f'    input size......... {3}x{input_size}x{input_size}')
        print(f'    architecture....... {architecture}')
        print(f'    format layer....... {nc(n_layers)}x{input_size}x{input_size}')
        print(f'    num layers......... {n_layers}')
        print(f'    layer channels..... {" | ".join([f"{nc(l+1)}{arrow}{nc(l)}" for l in reversed(range(n_layers))])}')
        print(f'    output size........ {nc(0)}x{out_depth}x{out_width}x{out_width}')
        if self.disc_type=='3d':
            print(f'    scale depth steps.. {" | ".join([f"{seq_len//2**s}{arrow}{seq_len//2**(s+1)}" for s in range(self.seq_layers)])}')
        calculate_model_size(self)
        # print(self)

    def __div2_times__(self, k):
        '''Calculates how many times k can be divided by 2'''
        times = 0
        while (k % 2) == 0:
            times += 1 
            k = k / 2
        return times

    def forward(self, input):
        if self.disc_type == '3d':
            input = torch.permute(input, (0,2,1,3,4))   # C first, then time dimension

        if self.architecture == 'skip':
            for l, (layer, fromRGB) in enumerate(zip(self.layers, self.fromRGB)):
                # Resample skip
                if l > 0:
                    if self.disc_type == '2d':
                        scale_factor = [0.5, 0.5]
                    elif self.disc_type == '3d':
                        scale_factor = [0.5, 0.5, 0.5] if l < self.seq_layers else [1.0, 0.5, 0.5]
                    input = F.interpolate(input, scale_factor=scale_factor, mode='nearest')

                y = fromRGB(input)
                if l == 0:
                    x = layer(y)
                else:
                    x = layer(x + y)
        else:
            x = self.fromRGB(input)
            for layer in self.layers:
                x = layer(x)

        return self.decision_layer(x)
if __name__ == '__main__':

    # disc_2 = Discriminator('3d', I, l, C, L).cuda()

    gen = Generator(64, 1024)
    disc1 = Discriminator(1, 1024)
    disc2 = Discriminator(16, 1024)
