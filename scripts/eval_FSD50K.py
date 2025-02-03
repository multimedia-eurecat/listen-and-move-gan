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
import random
import torch.utils.data.dataloader as data
from modules import GenNet
from converter import AudioConverter
from datasets import VideoLoader
from fid import FID

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Performs assessment of video quality over time.')
    parser.add_argument('--gpus',           type=str,   default='0',                help='GPUs.')
    parser.add_argument('--train_source',   type=str,                               help='Path to the training source: video or URMP+split+instrument.')    
    parser.add_argument('--checkpoint',     type=str,   default=None,               help='Path to the checkpoint.')

    parser.add_argument('--seq_len',        type=int,   default=32,                 help='Video sequence length [2,4,6,8,...]')
    parser.add_argument('--fps',            type=int,   default=20,                 help='Video frame rate.')
    parser.add_argument('--chunk_len',      type=float, default=0.085,               help='Duration of audio chunks in seconds (>=0.16).')

    parser.add_argument('--feat_type',      type=str,   default='mel',              help='Sound feature descriptors (mel spectrogram): mel or mfcc.')
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
    parser.add_argument('--vid_pred',       type=str, default='',nargs='?',const='',help='Video prediction: None (empty), basic, or dir (directional).')
    parser.add_argument('--double_finest',  action=argparse.BooleanOptionalAction,  help='If given, the finest (outer) G and D layers double its channel capacity.')
    parser.add_argument('--cond_gen',       action=argparse.BooleanOptionalAction,  help='If given, activates generator\'s conditional instance normalization.')
    args = parser.parse_args()
    
    device = 'cuda'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    T = args.seq_len

    from converter import AudioConverter
    aconverter = AudioConverter(chunk_len_sec=args.chunk_len, feat_type=args.feat_type, mel_bands=args.mel_bands)

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

    print(f'Loading checkpoint: {args.checkpoint}')
    Gn.load_state_dict(torch.load(args.checkpoint))
    Gn = Gn.to(device)
    Gn.eval()

    fid = FID(ds.getRandomFrames(500), device)

    num_audio_samples = 500
    max_len_sec = 5
    min_len_sec = 1.0
    min_frames = int(min_len_sec * args.fps)

    # fsd50k = '/media/ssd/datasets/FSD50K/FSD50K.dev_audio'
    fsd50k = '/media/ssd2/FSD50K/FSD50K.dev_audio'
    files = os.listdir(fsd50k)
    random.shuffle(files)

    num_vid_frames = 0
    fids = []
    i = 0
    while len(fids) < num_audio_samples or i >= len(files):
        filename = os.path.join(fsd50k, files[i])
        features = aconverter.file2features(filename, args.fps, duration=max_len_sec)
        i += 1
        
        # ignore too short audio samples
        if len(features) < min_frames:  continue

        Gn.reset_eval()
        with torch.no_grad():
            feats_batch = torch.stack(features).unsqueeze(0).to(device)
            fake_videos = Gn(feats_batch).squeeze(0)
            fake_frames = [fake_videos[i] for i in range(fake_videos.size(0))]
            fids.append(fid.calculate_FID(fake_frames))
            num_vid_frames += fake_videos.size(0)
        
        print(f'Processed files: {len(fids)}/{num_audio_samples}')
        
    print(f'Finished with checkpoint: {args.checkpoint}')
    print(f'FID: ' +\
          f'  mean={np.mean(fids):.2f}' +\
          f'  std={np.std(fids):.2f}' +\
          f'  total frames={num_vid_frames}')