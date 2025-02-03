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

import sys, os
sys.path.append(os.path.join(os.getcwd(), '..'))
import numpy as np
import torch
from modules import GenNet
from converter import AudioConverter
import torch.utils.data.dataloader as data
from datasets import VideoLoader
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from datetime import datetime

"""
This file performs the assessment of video quality over time to evaluate the quality of generated video sequences using various metrics: FVD, FID, MSE, PSNR, SSIM, and LPIPS.
"""

colors = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:olive',
]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Performs assessment of video quality over time.')
    parser.add_argument('--train_source',   type=str,                               help='Path to the training source: video or URMP+split+instrument.')
    parser.add_argument('--checkpoints',    type=str,   default=None,               help='List of paths to checkpoints.')
    parser.add_argument('--labels',         type=str,   default=None,               help='List of checkpoint names.')
    parser.add_argument('--image_size',     type=int,   default=256,                help='Video frame size.')
    parser.add_argument('--crop',           type=str,   default=None,               help='Video frame cropping coordinates at loading.')
    parser.add_argument('--max_seq_len',    type=int,   default=1024,               help='Maximum sequence length to evaluate.')
    parser.add_argument('--fps',            type=int,   default=20,                 help='Video frame rate.')
    parser.add_argument('--chunk_len',      type=float, default=0.085,              help='Duration of audio chunks in seconds (>=0.16).')
    parser.add_argument('--feat_type',      type=str,   default='lms',              help='Sound feature descriptors (mel spectrogram): mel, lms, or mfcc.')
    parser.add_argument('--mel_bands',      type=int,   default=64,                 help='Number of Mel bands, typically 64 or 128.')
    parser.add_argument('--e_motion',       type=int,   default=2,                  help='Size of motion random vector (feeds the rnn).')
    parser.add_argument('--z_content',      type=int,   default=0,                  help='Size of noise content (feeds the generator).')
    parser.add_argument('--sound_route',    type=str,   default='gen',              help='Sound features will be routed to "rnn", "gen" (generator) or "rnngen" for both.')    
    parser.add_argument('--motion_layers',  type=int,   default=1,                  help='Number of motion encoder layers.')    
    parser.add_argument('--motion_type',    type=str,   default='basic',            help='Recurrent motion encoding type: basic or accumulative.')
    parser.add_argument('--g_type',         type=str,   default='2d',               help='Generator type: 2d (residual) or 3d.')
    parser.add_argument('--g_arch',         type=str,   default='residual',         help='Generator architecture: basic, skip, or residual.')
    parser.add_argument('--vid_pred',       type=str, default='',nargs='?',const='',help='Video prediction: None (empty), basic, or dir (directional).')
    parser.add_argument('--double_finest',  action=argparse.BooleanOptionalAction,  help='If given, the finest (outer) G and D layers double its channel capacity.')
    parser.add_argument('--cond_gen',       action=argparse.BooleanOptionalAction,  help='If given, activates generator\'s conditional instance normalization.')
    parser.add_argument('--save_plot',      type=str,   default='',                 help='Metric to be plotted: none, mse, psnr, ssim.')
    args = parser.parse_args()

    device = 'cuda'
    T = args.max_seq_len
    aconverter = AudioConverter(chunk_len_sec=args.chunk_len, feat_type=args.feat_type, mel_bands=args.mel_bands)

    if args.crop is not None: args.crop = [int(c) for c in args.crop.split(',')]
    ds = VideoLoader(args.train_source,
                     audio_converter=aconverter,
                     split='validation',
                     seq_len=T,
                     target_fps=args.fps,
                     image_size=args.image_size,
                     crop=args.crop)
    dataloader = data.DataLoader(dataset=ds,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=False)


    checkpoints = args.checkpoints.split(',') if ',' in args.labels else [args.checkpoints]
    labels = args.labels.split(',') if ',' in args.labels else [args.labels]
    lc = len(checkpoints)

    from losses import Loss
    loss = Loss()
    from fid import FID
    fid = FID(ds.getRandomFrames(500), device)
    from fvd import FVD
    fvd_freq = 4
    fvd = FVD(ds.getRandomSeqs(20, T), fvd_freq)
    import lpips as LPIPS
    loss_lpips = LPIPS.LPIPS(net='alex').to(device) # best forward scores


    for checkpoint, label, color in zip(checkpoints, labels, colors[:lc]):
        print(f'Processing {label}: {checkpoint}')
        Gn = GenNet(output_size=args.image_size,
            dim_audio_feat=aconverter.get_num_features(),
            dim_e_motion=args.e_motion,
            dim_z_content=args.z_content,
            sound_route=args.sound_route,
            motion_layers=args.motion_layers,
            motion_type=args.motion_type,
            gen_type=args.g_type,
            cond_gen=args.cond_gen,
            vid_pred=args.vid_pred,
            architecture=args.g_arch,
            double_finest=args.double_finest)
        Gn.load_state_dict(torch.load(checkpoint)) 
        Gn = Gn.to(device)
        Gn.eval()

        plot_metric = torch.zeros((len(ds), T))
        mses, psnr, ssim, fids, lpips, fvds = [], [], [], [], [], []
        elapsed_time = []

        for it, (real_feats, real_videos) in enumerate(dataloader):
            with torch.no_grad():
                start_time = datetime.now()
                fake_videos = Gn(real_feats.to(device))
                elapsed_time += [(datetime.now() - start_time).total_seconds()]

            real_videos = real_videos.to(device)
            fake_frames = []
            for l in range(T):
                real_frame = torch.select(real_videos, 1, l)
                fake_frame = torch.select(fake_videos, 1, l)
                
                fake_frames.append(fake_frame.squeeze(0))

                err_mse, err_psnr = loss.MSE_PSNR(real_frame, fake_frame)
                err_ssim = loss.SSIM(real_frame, fake_frame)
                with torch.no_grad():
                    err_lpips = loss_lpips(real_frame, fake_frame).detach()

                mses.append(err_mse.cpu().numpy())
                psnr.append(err_psnr.cpu().numpy())
                ssim.append(err_ssim.cpu().numpy())
                lpips.append(err_lpips.cpu().numpy())

                if args.save_plot == 'mse':
                    plot_metric[it, l] = err_mse
                if args.save_plot == 'psnr':
                    plot_metric[it, l] = err_psnr
                if args.save_plot == 'ssim':
                    plot_metric[it, l] = err_ssim
                if args.save_plot == 'lpips':
                    plot_metric[it, l] = err_lpips
            
            fids.append(fid.calculate_FID(fake_frames))
            fvd.add_fake(fake_videos)
            if it % fvd_freq == 0 or (it+1) == len(dataloader):
                fvds.append(fvd.compute_fvd())

            if it * T > 1000:
                break

        print(f'Quality metrics for checkpoint: {checkpoint}')
        print(f'FVD:  {np.mean(fvds):.01f} \u00B1 {np.std(fvds):.01f}')
        print(f'FID:  {np.mean(fids):.01f} \u00B1 {np.std(fids):.01f}')
        print(f'MSE:  {np.mean(mses):.05f} \u00B1 {np.std(mses):.05f}')
        print(f'PSNR: {np.mean(psnr):.01f} \u00B1 {np.std(psnr):.01f}')
        print(f'SSIM: {np.mean(ssim):.02f} \u00B1 {np.std(ssim):.02f}')
        print(f'LPIPS:{np.mean(lpips):.02f} \u00B1 {np.std(lpips):.02f}')

        if args.save_plot:
            plot_metric = plot_metric.numpy()
            mean = np.mean(plot_metric, axis=0)
            std = np.std(plot_metric, axis=0)
            l = np.arange(1, T+1)
            plt.plot(l, mean, color=color, label=label)
            plt.fill_between(l, mean - std, mean + std, color=color, alpha=0.2)

        print(f'From {it} sequences of {T} frames')
        print(f'Average fps: {(len(elapsed_time)*T)/sum(elapsed_time):.2f}')

    if args.save_plot:
        plt.rcParams.update({'font.size': 14})
        plt.ylabel(str(args.save_plot).upper())
        if args.save_plot == 'mse':
            plt.ylim(bottom=0)
        plt.xlabel('Video frame')
        plt.xlim([1,T])
        plt.legend(loc="upper right")

        video_name = os.path.splitext(os.path.basename(args.train_video))[0]
        check_name = args.sound_route
        # check_name = '-'.join(labels)
        # check_name = os.path.splitext(os.path.basename(checkpoint))[0]
        # path = os.path.split(os.path.split(checkpoint)[0])[0]
        path = './'
        fig_name = os.path.join(path, f'eval-{video_name}-{check_name}-{args.save_plot}.png')
        plt.savefig(fig_name)
