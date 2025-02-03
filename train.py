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

import numpy as np
import random
import torch
from utils import finiteCheck
# -------------------------------------------------------------------------------------

def trainer():

    dataset = iter(dataloader)
    torch.autograd.set_detect_anomaly(True)

    from tqdm import tqdm
    for i in tqdm(range(start_iter, 1+args.total_iter), desc='Training'):
        
        Gn.train(); Di.train(); Dv.train()

        # ----------------------------------------------------
        # Get a real batch

        try:
            real_feats, real_videos = next(dataset)

        except (OSError, StopIteration):
            dataloader.dataset.offset = random.randint(0,args.seq_len-1)
            dataset = iter(dataloader)
            real_feats, real_videos = next(dataset)
            Gn.reset_train()

        real_feats, real_videos = real_feats.to(device0), real_videos.to(device1)
        pickup = np.random.randint(0, T)
        real_images = torch.select(real_videos, 1, pickup)

        # ----------------------------------------------------
        # Get a fake batch, feed G with input spectrograms

        fake_videos = Gn(real_feats)
        fake_videos = fake_videos.to(device1)
        fake_images = torch.select(fake_videos, 1, pickup)

        # ----------------------------------------------------
        # Another fake batch through frame permutation

        perm_videos = real_videos[:,np.random.permutation(T),...]

        # ----------------------------------------------------
        # Discriminators passes
        def accumulate(loss_name, error):
            counts[loss_name] = error.item()
            return error

        def pass_disc(disc, reals, fakes, stage, retain=False):
            fake_preds = disc(fakes)

            # on generator training
            if stage in ['Gi','Gv']:
                error = loss.criterionGAN_G(fake_preds)
                if stage == 'Gv':
                    if args.lambda_L1 > 0:
                        error_L1 = args.lambda_L1 * loss.criterionL1(reals, fakes)
                        error += accumulate(stage + '_L1', error_L1)
                    if args.lambda_pl > 0:
                        error_PL = args.lambda_pl * loss.perceptual(reals, fakes)
                        error += accumulate(stage + '_pl', error_PL)
                    
            # on discriminator training
            else:
                real_preds = disc(reals)
                error = loss.criterionGAN_D(fake_preds,
                                            real_preds,
                                            disc=disc,
                                            samples=[fakes, reals])
                counts[stage + '_gp'] = loss.gp.item()

            error.backward(retain_graph=retain)
            finiteCheck(disc.parameters())
            counts[stage] = torch.sum(error).item()

        # ----------------------------------------------------
        # 1. Discriminators training

        Dv_optim.zero_grad()
        pass_disc(Dv, real_videos, fake_videos.detach(), stage='Dv', retain=True)
        pass_disc(Dv, real_videos, perm_videos, stage='Dp')
        Dv_optim.step()

        Di_optim.zero_grad()
        pass_disc(Di, real_images.to(device0), fake_images.detach(), stage='Di') 
        Di_optim.step()

        # ----------------------------------------------------
        # 2. Generator training
        
        Gn_optim.zero_grad()
        pass_disc(Dv, real_videos, fake_videos, stage='Gv', retain=True)
        pass_disc(Di, real_images, fake_images, stage='Gi')
        finiteCheck(Gn.parameters())
        Gn_optim.step()

        # ----------------------------------------------------
        # Logging
        
        logger.sum_counters(counts)
        fvd_metric.add_fake(fake_videos.detach())
        if i % log_period == 0:
            mse, psnr = loss.MSE_PSNR(fake_videos, real_videos)
            ssim = loss.SSIM(fake_videos, real_videos)
            fvd = fvd_metric.compute_fvd()
            logger.add_entry(i, mse, psnr, ssim, fvd)
        if i > start_iter and i % check_period == 0:
            deployer.deploy_iter(i, Gn, T)
            deployer.checkout_iter(i, Gn)

    logger.finish()

# -----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Audio to Video GAN, the model will learn to generate video frames from an audio track.')
    parser.add_argument('--train_source',   type=str,                               help='Path to the training source: video or URMP+split+instrument.')
    parser.add_argument('--gpus',           type=str,   default='0',                help='Comma-separated GPU index list.')
    parser.add_argument('--trial_name',     type=str,   default='trial',            help='A brief description of the training trial.')
    parser.add_argument('--resume',         type=str,   default='',                 help='Path to deploy folder to resume training.')

    parser.add_argument('--seq_len',        type=int,   default=32,                 help='Video sequence length [2,4,6,8,...]')
    parser.add_argument('--fps',            type=int,   default=20,                 help='Video frame rate.')
    parser.add_argument('--chunk_len',      type=float, default=0.085,              help='Duration of audio chunks in seconds (>=0.16).')

    parser.add_argument('--feat_type',      type=str,   default='lms',              help='Sound feature descriptors (mel spectrogram): mel, lms, or mfcc.')
    parser.add_argument('--mel_bands',      type=int,   default=64,                 help='Number of Mel bands, typically 64 or 128.')
    parser.add_argument('--e_motion',       type=int,   default=2,                  help='Size of motion random vector (feeds the rnn).')
    parser.add_argument('--z_content',      type=int,   default=0,                  help='Size of noise content (feeds the generator).')
    parser.add_argument('--sound_route',    type=str,   default='gen',              help='Sound features will be routed to "rnn", "gen" (generator) or "rnngen" for both.')    
    parser.add_argument('--motion_layers',  type=int,   default=1,                  help='Number of motion encoder layers.')    
    parser.add_argument('--motion_type',    type=str,   default='basic',            help='Recurrent motion encoding type: basic or feedback.')

    parser.add_argument('--image_size',     type=int,   default=256,                help='Video frame size.')
    parser.add_argument('--crop',           type=str,   default=None,               help='Video frame cropping coordinates at loading.')
    parser.add_argument('--g_type',         type=str,   default='2d',               help='Generator type: 2d (residual) or 3d.')
    parser.add_argument('--g_arch',         type=str,   default='residual',         help='Generator architecture: basic, skip, or residual.')
    parser.add_argument('--d_arch',         type=str,   default='skip',             help='Discriminators architecture: basic, skip, or residual.')
    parser.add_argument('--vid_pred',       type=str, default='',nargs='?',const='',help='Video prediction: None (empty), basic, or dir (directional).')
    parser.add_argument('--double_finest',  action=argparse.BooleanOptionalAction,  help='If given, the finest (outer) G and D layers double its channel capacity.')
    parser.add_argument('--cond_gen',       action=argparse.BooleanOptionalAction,  help='If given, activates generator\'s conditional instance normalization.')

    parser.add_argument('--total_iter',     type=int,   default=50000,              help='How many iterations to train in total, the value is in assumption that init step is 1.')
    parser.add_argument('--loss_type',      type=str,   default='wgan-gp',          help='Type of cost function: wgan-gp (Wasserstein-GP), wgan, hinge, ns (non-saturating), ls (least-squares), bce (BCELogistics).')
    parser.add_argument('--lr_g',           type=float, default=1e-4,               help='Learning rate for G.')
    parser.add_argument('--lr_d',           type=float, default=4e-4,               help='Learning rate for D.')
    parser.add_argument('--lambda_L1',      type=float, default=0,                  help='Weight factor for L1 loss term (set 0 to disable)..')
    parser.add_argument('--lambda_pl',      type=float, default=10,                 help='Weight factor for perceptual loss (set 0 to disable).')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.loss_type = args.loss_type.lower()
    print('\n' + str(args) + '\n')

    # Multi-GPU
    split_gpu = len(args.gpus.split(',')) > 1
    device0 = torch.device('cuda:0') if split_gpu else torch.device('cuda')
    device1 = torch.device('cuda:1') if split_gpu else torch.device('cuda')    

    T = args.seq_len

    from converter import AudioConverter
    aconverter = AudioConverter(chunk_len_sec=args.chunk_len, feat_type=args.feat_type, mel_bands=args.mel_bands)

    from modules import GenNet, Discriminator
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

    Di = Discriminator(1, args.image_size, architecture=args.d_arch, double_finest=args.double_finest)
    Dv = Discriminator(T, args.image_size, architecture=args.d_arch, double_finest=args.double_finest)
    
    Gn = Gn.to(device0)
    Di = Di.to(device1)
    Dv = Dv.to(device1)

    # Optimizers
    Gn_optim = torch.optim.Adam(Gn.parameters(), lr=args.lr_g, betas=(0.3, 0.999))
    Di_optim = torch.optim.Adam(Di.parameters(), lr=args.lr_d, betas=(0.3, 0.999))
    Dv_optim = torch.optim.Adam(Dv.parameters(), lr=args.lr_d, betas=(0.3, 0.999))

    # Dataloader
    import torch.utils.data.dataloader as data
    from datasets import VideoLoader, SubURMP
    if args.crop is not None: args.crop = [int(c) for c in args.crop.split(',')]

    if os.path.isfile(args.train_source):
        ds = VideoLoader(args.train_source,
                         audio_converter=aconverter,
                         split='train',
                         seq_len=T,
                         target_fps=args.fps,
                         image_size=args.image_size,
                         crop=args.crop)
    else:
        SubURMP_path, SubURMP_split, SubURMP_instrument = args.train_source.split('+')
        ds = SubURMP(SubURMP_path,
                     SubURMP_split,
                     SubURMP_instrument,
                     audio_converter=aconverter,
                     seq_len=T,
                     image_size=args.image_size,                
                     crop=args.crop)

    # Something may not go well if batch size isn't 1
    dataloader = data.DataLoader(dataset=ds,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=3,
                                 pin_memory=False)

    # Loss
    from losses import Loss
    loss = Loss(args.loss_type,
                load_pl=args.lambda_pl>0,
                device=device1)

    # Output folder
    if args.resume == '':
        from datetime import datetime
        start_time = datetime.now()
        time_stamp = str(start_time.date()).replace('-','') + f'{start_time.hour:02d}{start_time.minute:02d}{start_time.second:02d}'
        out_path = '' + time_stamp + f'-{args.trial_name}'
        os.mkdir(out_path)
        start_iter = 1
    else:
        out_path = args.resume
        assert os.path.exists(out_path), 'Resume folder not found.'
        checkpoints = os.listdir(os.path.join(out_path, 'checkpoints'))
        checkpoints.sort()
        iter_str = checkpoints[-1].split('_')[0]
        Gn.load_state_dict(torch.load(iter_str + '_Gn.model'))
        Di.load_state_dict(torch.load(iter_str + '_Di.model'))
        Dv.load_state_dict(torch.load(iter_str + '_Dv.model'))
        start_iter = int(iter_str)

    # Logger
    from logger import Logger
    log_period = 200
    logger = Logger(out_path, str(args) +'\n[Dataset]\n'+ ds.conf)
    counts = logger.get_counters()

    # FVD
    from fvd import FVD
    fvd_metric = FVD(ds.getRandomSeqs(20, T), 20, device1)

    # Deploy
    from deploy import Deployer
    check_period = 10000
    deployer = Deployer(out_path,
                        aconverter,
                        args.fps)
    
    # Train
    print(f'Logging period: {log_period}')
    print(f'Checkout period: {check_period}')
    trainer()
