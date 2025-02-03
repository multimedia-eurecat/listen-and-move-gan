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

import os
from os import listdir
from os.path import isfile, join
import shutil
import random
import numpy as np
import torch
import torch.utils.data as data
from torchvision.utils import save_image
import torchvision.transforms as T
from PIL import Image

class BaseLoader(data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.offset = 0
        # Child should implement
        self.frame_files = []
        self.transform = []
        self.features = []
        return

    def __debug_spectrograms__(self, spectrograms):
        os.makedirs('./debug/', exist_ok=True)
        for i, f in enumerate(self.features):
            f = torch.tensor(f).view(-1,1).repeat(1, 64)
            s = torch.permute(spectrograms[i], (1,0))
            fs = torch.cat(((f+8)/16,(s+8)/16), dim=1)
            save_image(fs, os.path.join('./debug/', str(i) + '.png')) 

    def __len__(self):
        return (len(self.frame_files) // self.seq_len) - 1 # offset guard

    def open_files(self, files):
        return [self.transform(Image.open(f)) for f in files]

    def __getitem__(self, index):
        start = index * self.seq_len + self.offset
        end = start + self.seq_len
        frames_seq = torch.stack(self.open_files(self.frame_files[start:end]), dim=0)
        feats_seq  = torch.tensor(np.stack(self.features[start:end], axis=0), dtype=torch.float)
        return feats_seq, frames_seq

    def getRandomFrames(self, num_samples):
        num_samples = min(num_samples, len(self.frame_files))
        random_samples = random.sample(self.frame_files, num_samples)
        return self.open_files(random_samples)

    def getRandomSeqs(self, num_seqs, seq_len):
        random_seqs = []
        for _ in range(num_seqs):
            start = random.randint(0, len(self.frame_files)-seq_len)
            end = start + seq_len
            seq = torch.stack(self.open_files(self.frame_files[start:end]), dim=0)
            random_seqs.append(seq)
        return torch.stack(random_seqs, dim=0)

class VideoLoader(BaseLoader):
    def __init__(self,
                video_file,                 # path to the training input video
                audio_converter,            # instance of the audio converter
                split,                      # train or validation
                seq_len         = 2,        # length of the video sequences
                target_fps      = 10,       # video framerate (will resample frames from source fps)
                image_size      = 256,      # target image size
                max_frames      = -1,       # maximum number of video frames
                crop            = None,     # cropping coordinates [left, top, right, bottom]
                force_export    = False     # If true, the export folder with video and audio files is repopulated
        ) -> None:
        super().__init__()
        print('[Video Loader]')

        self.seq_len    = seq_len
        self.fps        = target_fps

        # Resize and Normalize (remember to adjust conveniently utils.py)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(image_size),
            T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

        # Export video frames
        export_path = os.path.splitext(video_file)[0] + f'-{image_size}' + f'-{self.fps}'
        if force_export and os.path.exists(export_path): shutil.rmtree(export_path)
        if not os.path.exists(export_path):
            print(f'Exporting frames to {export_path}')
            os.mkdir(export_path)
            w, h, x, y = crop[2]-crop[0], crop[3]-crop[1], crop[0], crop[1]
            export_files = os.path.join(export_path, 'frame-%06d.png')
            os.system(f'ffmpeg -i {video_file} -filter:v "crop={w}:{h}:{x}:{y},scale={image_size}:{image_size},fps={self.fps}" -loglevel error -y {export_files}')

        # Video frames read-out
        self.frame_files = [join(export_path, f) for f in listdir(export_path) if isfile(join(export_path, f))]
        self.frame_files = sorted(self.frame_files)

        # Export audio track
        audio_file = os.path.splitext(video_file)[0] + '.wav'
        if not os.path.exists(audio_file) or force_export:
            os.system(f'ffmpeg -i {video_file} -loglevel error -y {audio_file}')

        # Generate features
        features = audio_converter.file2features(audio_file, self.fps)
        self.features = [feat.numpy() for feat in features]

        # Train/validation splits
        if max_frames == -1: max_frames = 100000
        max_frames = min(len(self.frame_files), len(self.features), max_frames)
        if split == 'train':
            start_frame = 0
            end_frame = int(max_frames * 0.8)
        else:
            start_frame = int(max_frames * 0.8)
            end_frame = max_frames
        self.frame_files = self.frame_files[start_frame:end_frame]
        self.features = self.features[start_frame:end_frame]

        # self.__debug_spectrograms__(spectrograms)
        
        self.conf = str(
        f'    video source....... {video_file}\n'
        f'    target fps......... {self.fps}\n'
        f'    sequence length.... {self.seq_len}\n'
        f'    video frames....... {len(self.frame_files)}\n'
        f'    num sequences...... {self.__len__()}\n'
        f'    image size......... {image_size}\n'
        f'    crop area.......... {crop}\n'
        f'    spectrogram size... {audio_converter.get_mel_size()}\n'
        f'    features size...... {self.features[0].shape}')
        print(self.conf)
        return

class SubURMP(BaseLoader):
    def __init__(self,
                path,                       # path to the training Sub-URMP dataset
                split,                      # 'train' or 'validation'
                instrument,                 # instrument: 'cello', 'trombone',...
                audio_converter,            # instance of the audio converter
                max_frames      = -1,       # maximum number of video frames (-1 loads all)
                seq_len         = 2,        # length of the video sequences
                image_size      = 256,      # target image size
                crop            = None,     # cropping coordinates [left, top, right, bottom]
        ) -> None:
        super().__init__()
        print('[SubURMP Loader]')

        self.seq_len    = seq_len
        self.fps        = 10

        # Resize and Normalize (remember to adjust conveniently utils.py)
        w, h, x, y = crop[2]-crop[0], crop[3]-crop[1], crop[0], crop[1]
        self.transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda img: T.functional.crop(img, y, x, h, w)),
            T.Resize(image_size),
            T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

        # Video frames read-out
        # Note: if more than one song, audio-visual incoherence is induced between 
        # songs transitions, i.e. the last and first batches of two consecutive songs
        self.frame_files = []
        self.chunk_files = []
        self.start_song = []
        image_path = os.path.join(path, 'img',   split, instrument)
        chunk_path = os.path.join(path, 'chunk', split, instrument)
        for song in range(0,44):
            start = 500
            stride = 100
            song_frames = []
            song_chunks = []
            for f in range(100000):
                image = os.path.join(image_path, f'{instrument}{song:02}_{start + f * stride}.jpg')
                chunk = os.path.join(chunk_path, f'{instrument}{song:02}_{start + f * stride}.wav')
                if not os.path.exists(image): break
                song_frames.append(image)
                song_chunks.append(chunk)

                if len(self.frame_files) + len(song_frames) > max_frames and max_frames > 0: break

            if not song_frames: continue

            self.start_song += [len(self.frame_files)]
            song_batches = len(song_frames) // self.seq_len
            num_strides = song_batches * self.seq_len
            self.frame_files += song_frames[:num_strides]
            self.chunk_files += song_chunks[:num_strides]

            if len(self.frame_files) + self.seq_len > max_frames and max_frames > 0: break
        
        # Generate features
        # URMP chunks are 0.5s length, only the 1st audio frame is taken, i.e. 1st audio_converter's chunk
        from tqdm import tqdm
        self.features = []
        for c in tqdm(range(len(self.chunk_files)), desc='Extracting audio features'):
            chunk_feats = audio_converter.file2features(self.chunk_files[c], self.fps, duration=audio_converter.chunk_len_sec, verbose=False)
            self.features += [chunk_feats[0].numpy()]
        
        self.conf = str(
        f'    URM instrument..... {instrument}\n'
        f'    URM split.......... {split}\n'
        f'    number of songs.... {len(self.start_song)} ({self.start_song})\n'
        f'    target fps......... {self.fps}\n'
        f'    sequence length.... {self.seq_len}\n'
        f'    video frames....... {len(self.frame_files)}\n'
        f'    num sequences...... {self.__len__()}\n'
        f'    image size......... {image_size}\n'
        f'    crop area.......... {crop}\n'
        f'    spectrogram size... {audio_converter.get_mel_size()}\n'
        f'    features size...... {self.features[0].shape}')
        print(self.conf)
        return

# ----------------------------------------------------------------------------


if __name__ == '__main__':

    from converter import AudioConverter
    audio_converter = AudioConverter(chunk_len_sec=0.150, feat_type='lms')

    ds = VideoLoader('./assets/Once_I_Saw_CUT.mp4',
                    audio_converter,
                    split='train',
                    seq_len=16,
                    target_fps=20,
                    image_size=256,
                    max_frames=5000,
                    force_export=True,
                    crop=[290, 0, 290+720, 720])

    # ds = SubURMP('/media/ssd/datasets/Sub-URMP',
    #             'train',
    #             'cello',
    #             audio_converter,
    #             max_frames=100,
    #             seq_len=16,
    #             image_size=256,
    #             crop=[220, 0, 1080+220, 1080])

    dataloader = data.DataLoader(dataset=ds,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=3,
                                 pin_memory=True)
    dataset = iter(dataloader)

    for i in range(5):
        real_feats, real_videos = next(dataset)
        save_image((real_videos[0][0]+1.0)*0.5,f'./tmp{i}.png')
        print(real_feats.shape)
        print(real_videos.shape)
        print(f'vid max {torch.max(real_videos)}')
        print(f'vid min {torch.min(real_videos)}')
        print(f'feat min {torch.min(real_feats)}')
        print(f'feat max {torch.max(real_feats)}')
