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

from datetime import datetime
import atexit
from torch.utils.tensorboard import SummaryWriter
import os


"""
This file contains logging utilities, including configuration settings, training losses, and other relevant metrics.
It also provide functionality for writing logs to a file and TensorBoard.

Classes:
    - Logger: A class that handles logging of training progress and metrics.

Methods:
    - log_config: Logs the configuration settings.
    - log_loss: Logs the training losses.
    - log_metrics: Logs other relevant metrics.
    - write_to_file: Writes logs to a file.
    - write_to_tensorboard: Writes logs to TensorBoard.
"""

class Logger():
    def __init__(self,
                log_path,       # output path
                config_summary, # summary of training configuration (string)
                ) -> None:

        # Counters
        self.reset_counters()
        self.start_time = datetime.now()

        # Folder
        os.makedirs(log_path, exist_ok=True)
        folder_name = os.path.split(log_path)[-1]

        # Log file
        self.log_filename = os.path.join(log_path, 'log_train-' + folder_name + '.txt')
        self.__write_file__(config_summary + '\n\n', 'w')

        # Tensor board
        self.WRITER = SummaryWriter(os.path.join('runs', folder_name))
        self.WRITER.add_text(f'{log_path}-CONFIG', config_summary)

        atexit.register(self.__close__)
        print(f'Log folder: {log_path}')

    def __write_file__(self, msg, mode='a+'):
        file = open(self.log_filename, mode)
        file.write(msg)
        file.close()

    def get_counters(self):
        return self.losses

    def reset_counters(self):
        self.log_period = 0
        self.losses = {
            'Gi'    :   0,
            'Gv'    :   0,
            'Gi_L1' :   0,
            'Gv_L1' :   0,
            'Gi_pl' :   0,
            'Gv_pl' :   0,
            'Gi_FM' :   0,
            'Gv_FM' :   0,
            'Gv_TV' :   0,
            'Di' :      0,
            'Dv' :      0,
            'Dp' :      0,
            'Di_gp' :   0,
            'Dv_gp' :   0,
            'Dp_gp' :   0
            }
    
    def sum_counters(self, counters):
        self.log_period += 1
        for key in counters:
            self.losses[key] += counters[key]
        # adds permutation loss to the main video loss (dirty)
        self.losses['Dv'] += counters['Dp']
        self.losses['Dv_gp'] += counters['Dp_gp']

    def add_entry(self, iter, mse, psnr, ssim, fvd):
        
        # Normalize all losses across log period
        for key in self.losses:
            self.losses[key] /= self.log_period

        def format(key, value=None):
            return f' {key}:{self.losses[key]:.6f} ' if value is None else f' {key}:{value:.6f} '
        
        def add_losses(label, keys):
            self.WRITER.add_scalars('Loss/' + label, {k:self.losses[k] for k in keys}, iter)
        
        def add_metric(label, value):
            self.WRITER.add_scalars('Metrics/' + label, {label:value}, iter)

        def loss_entry(key, label):
            if self.losses[key] != 0:
                add_losses(label, [key])
                return format(key)
            else: return ''

        losses_labels = {
            'Gi'    : 'Total',
            'Gv'    : 'Total',
            'Di'    : 'Total',
            'Dv'    : 'Total',
            'Di_gp' : 'Penalty',
            'Dv_gp' : 'Penalty',
            'Gi_L1' : 'L1',
            'Gv_L1' : 'L1',
            'Gi_pl' : 'Perceptual',
            'Gv_pl' : 'Perceptual',
            'Gi_FM' : 'Perceptual',
            'Gv_FM' : 'Perceptual',
            'Gv_TV' : 'TV',
        }

        state_msg = f'Iter:{iter} '
        for key, label in losses_labels.items():
            state_msg += loss_entry(key, label)

        state_msg += format('MSE', mse)
        add_metric('MSE', mse)

        state_msg += format('PSNR', psnr)
        add_metric('PSNR', psnr)

        state_msg += format('SSIM', ssim)
        add_metric('SSIM', ssim)

        state_msg += format('FVD', fvd)
        add_metric('FVD', fvd)

        self.__write_file__(state_msg + '\n')  
        self.reset_counters()      

    def get_elapsed_time(self):
        return datetime.now() - self.start_time

    def finish(self):
        ets = self.get_elapsed_time()
        msg = f'Elapsed time: {ets}'
        self.__write_file__('\n\n' + msg)
        print(msg)
        self.__close__()

    def __close__(self):
        self.WRITER.close()

    