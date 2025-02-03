
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

from matplotlib import cm
import os
import os.path as op
import random
import torch
from torchvision.utils import save_image
import torchvision
from datetime import datetime
from utils import range01

deploy_files = ['./assets/seyeon_jung-JSBach_cello_suite_No6_10s.wav',
                # './assets/MakeYouMoveVoice_0_25s_fps20.wav',
                # './assets/nirvana-come_as_you_are.wav',
                './assets/pau_casals-el_cant_dels_ocells.wav',
                # './assets/MEAD_M31_right_60_Excerpt.wav',
                './assets/MEAD_W33_left_60_Excerpt.wav',
                # './assets/Belen_Lopez_en_Corral_de_la_Moreria_CUT-Excerpt.wav',
                # './assets/MLK-IHaveAdream.wav',
                # './assets/Obama-Weekly_Address_Excerpt.wav',
                # './assets/JACOB_COLLIER-Excerpt.wav',
                './assets/Once_I_Saw_CUT_Excerpt.wav',
                './assets/Istanbul_Agop_16_Xist_Ion_Crash_Cymbal-Brilliant-cut_last_end.wav',
                './assets/195149__flcellogrl__cello-c2chromatic-scale-12Tup.wav',
                # './assets/203494__tesabob2001__piano-chromatic-scale_CUT.wav',
                # './assets/AuSep_2_tbn_21_Rejouissance_CUT.wav',
                # './assets/AuSep_2_vc_11_Maria_CUT.wav'
                ]

class Deployer():
    def __init__(self,
                deploy_path,                # output path
                audio_converter,            # Audio converter instance
                fps,                        # Frames per Second
                audio_files=deploy_files    # List of audio file paths
                ) -> None:

        # Log folders
        self.check_dir = op.join(deploy_path, 'checkpoints')
        os.makedirs(self.check_dir, exist_ok=True)
        self.specs_dir = op.join(deploy_path, 'specs')
        os.makedirs(self.specs_dir, exist_ok=True)
        self.videos_dir = op.join(deploy_path, 'videos')
        os.makedirs(self.videos_dir, exist_ok=True)

        # Validation spectrograms
        self.fps = fps
        self.fs_hz = audio_converter.fs_hz # copy
        self.validation_audio = []
        self.validation_feats = []
        for file in audio_files:
            # audio
            audio = audio_converter.read_file(file)
            self.validation_audio.append(audio)
            # spectrograms
            specs = audio_converter.waveform2spectrograms(audio, self.fps)
            for c, s in enumerate(specs):
                filename = op.splitext(op.split(file)[-1])[0]
                self.__save_spectrogram__(s.squeeze(0), filename + f'-chunk{c:04d}.jpg')
            # audio features
            features = [audio_converter.spec2features(s) for s in specs]
            self.validation_feats.append(features)

    def __save_spectrogram__(self, spectrogram, filename, map=cm.get_cmap('magma')):
        spectrogram = 0.5 + spectrogram / 12  # heuristic adjustment
        magnitude_color = torch.tensor(map(spectrogram.numpy())).permute(2, 0, 1)
        save_image(magnitude_color[:3], os.path.join(self.specs_dir, filename))

    @torch.no_grad()
    def gen_random_frames(self, Gn, seq_len, num_seq=5, device='cuda'):
        ''' Generate video sequences from a random number of
         audio sequences from the validation set'''
        Gn.eval()
        frames = []
        for _ in range(num_seq):
            Gn.reset_eval()
            # v = random.randint(0, len(self.validation_feats)-1)
            v = 0   # Warning: just the first audio track is currently used
            f = random.randint(0, len(self.validation_feats[v])-seq_len-1)
            feats_batch = torch.stack(self.validation_feats[v][f:f+seq_len]).unsqueeze(0).to(device)
            images = Gn(feats_batch).squeeze(0)
            frames += [images[i] for i in range(images.size(0))]
        return frames

    @staticmethod
    def get_iter_str(iter):
        return str(iter).zfill(7)

    def deploy_iter(self, iter, Gn, seq_len, audio_files=deploy_files):
        iter_str = self.get_iter_str(iter)
        self.deploy(iter_str, Gn, seq_len, audio_files=audio_files)

    @torch.no_grad()
    def deploy(self, name, Gn, seq_len, audio_files=deploy_files, device='cuda', verbose=False):
        Gn.eval()
        for feats, audio, filename in zip(self.validation_feats, self.validation_audio, audio_files):
            if verbose: print(f'Processing audio: {filename}'); start_time = datetime.now()
            Gn.reset_eval()
            frames = []
            F = len(feats)
            for start in range(0, F, seq_len):
                end = min(F, start + seq_len)
                feats_batch = torch.stack(feats[start:end]).unsqueeze(0).to(device)
                images = Gn(feats_batch).squeeze(0)
                # images = Gn(feats_batch)[1]['raw_rgb'].squeeze(0).repeat([1,3,1,1])
                # images = Gn(feats_batch)[1]['raw_rgb'].squeeze(0)
                frames += [images[i] for i in range(images.size(0))]
            frames = torch.stack(frames).permute((0, 2, 3, 1))                      # [T, H, W, C]
            frames = torch.clamp(range01(frames), min=0, max=1)                     # clamp [0, 1] see datasets.py
            frames = (255 * frames).type(torch.uint8).cpu()                         # uint8 0-255
            if verbose: total_time = datetime.now() - start_time
            export_name = op.splitext(op.split(filename)[-1])[0] + f'_{name}.mp4'
            if len(audio.size()) < 2: audio = audio.repeat(2,1)                     # 2 audio channels
            audio = audio[:, :(self.fs_hz * len(frames)) // self.fps]               # video duration = audio duration
            torchvision.io.write_video(op.join(self.videos_dir, export_name),
                                    video_array=frames,
                                    fps=self.fps,
                                    video_codec='libx264',
                                    audio_array=audio,
                                    audio_fps=self.fs_hz,
                                    audio_codec='aac')
            if verbose:
                print(f'Generated video: {export_name}\n'
                      f'>> Target: {self.fps} fps, Averaged Inference: {len(frames)/total_time.total_seconds():.2f} fps')

    def checkout_iter(self, iter, Gn=None, Di=None, Dv=None):
        iter_str = self.get_iter_str(iter)
        if Gn is not None:
            torch.save(Gn.state_dict(), op.join(self.check_dir, iter_str + '_Gn.model'))
        if Di is not None:
            torch.save(Di.state_dict(), op.join(self.check_dir, iter_str + '_Di.model'))
        if Dv is not None:            
            torch.save(Dv.state_dict(), op.join(self.check_dir, iter_str + '_Dv.model'))


# -----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Audio to Video GAN, the model will generate video frames from a list of audio files.')
    parser.add_argument('--gpus',           type=str,   default='0',                help='GPUs.')
    parser.add_argument('--checkpoint',     type=str,   default=None,               help='Path to the checkpoint.')

    parser.add_argument('--seq_len',        type=int,   default=32,                 help='Video sequence length [2,4,6,8,...]')
    parser.add_argument('--fps',            type=int,   default=20,                 help='Video frame rate.')
    parser.add_argument('--chunk_len',      type=float, default=0.085,              help='Duration of audio chunks in seconds (>=0.16).')

    parser.add_argument('--feat_type',      type=str,   default='mel',              help='Sound feature descriptors (mel spectrogram): mel or mfcc.')
    parser.add_argument('--mel_bands',      type=int,   default=64,                 help='Number of Mel bands, typically 64 or 128.')
    parser.add_argument('--e_motion',       type=int,   default=2,                  help='Size of motion random vector (feeds the rnn).')
    parser.add_argument('--z_content',      type=int,   default=0,                  help='Size of noise content (feeds the generator).')
    parser.add_argument('--sound_route',    type=str,   default='gen',              help='Sound features will be routed to "rnn", "gen" (generator) or "rnngen" for both.')    
    parser.add_argument('--motion_layers',  type=int,   default=1,                  help='Number of motion encoder layers.')    
    parser.add_argument('--motion_type',    type=str,   default='basic',            help='Recurrent motion encoding type: basic or feedback.')

    parser.add_argument('--image_size',     type=int,   default=256,                help='Video frame size.')
    parser.add_argument('--g_type',         type=str,   default='2d',               help='Generator type: 2d (residual) or 3d.')
    parser.add_argument('--g_arch',         type=str,   default='residual',         help='Generator architecture: basic, skip, or residual.')
    parser.add_argument('--vid_pred',       type=str, default='',nargs='?',const='',help='Video prediction: None (empty), basic, or dir (directional).')
    parser.add_argument('--double_finest',  action=argparse.BooleanOptionalAction,  help='If given, the finest (outer) G and D layers double its channel capacity.')
    parser.add_argument('--cond_gen',       action=argparse.BooleanOptionalAction,  help='If given, activates generator\'s conditional instance normalization.')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print('\n' + str(args) + '\n')

    # Audio Converter
    from converter import AudioConverter
    aconverter = AudioConverter(chunk_len_sec=args.chunk_len, feat_type=args.feat_type, mel_bands=args.mel_bands)

    # Generator
    from modules import GenNet
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
    Gn.load_state_dict(torch.load(args.checkpoint)) 
    Gn.cuda().eval()

    # Paths and names
    check_path, basename = os.path.split(args.checkpoint)
    deploy_path = os.path.split(check_path)[0]

    # Deploy
    from deploy import Deployer
    deployer = Deployer(deploy_path,
                        aconverter,
                        args.fps)

    deploy_name = os.path.splitext(basename)[0] + str(
    f'-is{args.image_size}'
    f'-sl{args.seq_len}'
    f'-fs{args.fps}'
    f'-cl{args.chunk_len}'
    )

    print(f'Deploy name {deploy_name}')
    deployer.deploy(deploy_name, Gn, args.seq_len, verbose=True)
