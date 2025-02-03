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
import torchaudio
import numpy as np
import numpy.lib.stride_tricks as stricks
import librosa
import librosa.feature

STFT_WIN_LEN_SECS = 0.025
STFT_HOP_SECS = 0.01
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
MEL_BANDS = 64
NUM_MFCC = 20

class AudioConverter():
    '''
    A class used for audio processing in Pytorch to compute Mel, LMS or MFCC descriptors 
    extracted from an audio file or a list of spectrograms built upon the Librosa library.
    Descriptors are computed as average value of each frequency bin over time
    with a pre-defined window length and stride (hop).

    ...

    Attributes
    ----------
    chunk_len_sec : float
        audio chunk length in seconds
    feat_type : str
        Type of audio features: mel, lms, or mfcc.
    mel_bands : int
        number of Mel bands
    '''
    def __init__(self,
                 chunk_len_sec  = 0.085,    # audio chunk length in seconds
                 feat_type      = 'mel',    # Type of audio features: mel, lms, or mfcc.
                 mel_bands      = MEL_BANDS # number of Mel bands
                ) -> None:
        print(f'[Audio Converter]')
        self.feat_type = feat_type.lower()
        if not any(self.feat_type == s for s in ['mel', 'lms', 'mfcc']):
            raise ValueError("Unknown type of features")

        self.mel_bands      = mel_bands
        self.fs_hz          = 64000
        self.chunk_len_sec  = chunk_len_sec
        self.num_frames     = int(round(chunk_len_sec / STFT_HOP_SECS))

        print(f'    chunk length....... {chunk_len_sec}s')
        print(f'    spectrogram bands.. {self.mel_bands}')
        print(f'    spectrogram frames. {self.num_frames}')
        print(f'    sample rate........ {self.fs_hz}')
        print(f'    feat type.......... {self.feat_type}')

    def spec2features(self, spectrogram):
        '''Extracts sound features from a power mel-spectrogram:
            'mel': takes the value at each frequency bin averaged across audio frames
            'lms': takes the log-value at each frequency bin averaged across audio frames
            'mfcc': takes the log-mel frequency cepstral coefficients averaged across audio frames

        Parameters
        ----------
        spectrogram : tensor or numpy
            A 2-dimensional tensor/array.

        Returns
        -------
        Audio features as 1D tensor of size MEL_BANDS or NUM_MFCC
        '''
        if 'mfcc' == self.feat_type:
            if isinstance(spectrogram, torch.Tensor):
                spectrogram = spectrogram.numpy()
            mfcc = librosa.feature.mfcc(S=librosa.power_to_db(spectrogram), n_mfcc=NUM_MFCC)
            features = torch.from_numpy(mfcc)
        elif 'lms' == self.feat_type:
            if isinstance(spectrogram, torch.Tensor):
                spectrogram = spectrogram.numpy()
            lms = librosa.power_to_db(spectrogram, ref=np.max)
            features = torch.from_numpy(lms)
        else:
            features = spectrogram
        
        return torch.mean(features, dim=1)

    def file2features(self, file, fps, duration=-1, verbose=True):
        '''Reads a waveform from audio file and returns audio features using a sliding window processing.

        Parameters
        ----------
        file : str
            The audio file location
        fps : float
            Frames per second to determine the stride of the listening window as hop=1/fps
        duration : float
            Read-out length in seconds. If -1 all audio samples are loaded.
        verbose : boolean
            Set False to mute console info prompts.

        Returns
        -------
        A list of 1D tensors containing the audio features of each audio chunk (sliding window).
        '''        
        spectrograms = self.file2spectrograms(file, fps, duration, verbose)
        return [self.spec2features(s) for s in spectrograms]

    def file2spectrograms(self, file, fps, duration=-1, verbose=True):
        '''Reads a waveform from audio file and returns a list of spectrograms for each sliding window.

        Parameters
        ----------
        file : str
            The audio file location
        fps : float
            Frames per second to determine the stride of the listening window as hop=1/fps
        duration : float
            Read-out length in seconds. If -1 all audio samples are loaded.
        verbose : boolean
            Set False to mute console info prompts.

        Returns
        -------
        A list of 2D tensors containing the spectrograms of each audio chunk (sliding window).
        ''' 
        if verbose: print(f'Reading audio file: {file}')
        audio = self.read_file(file, duration)
        return self.waveform2spectrograms(audio, fps)

    def read_file(self, file, duration=-1):
        '''Reads a waveform from audio file and resample to a base sample rate.
        Also stereo are converted to mono channel.

        Parameters
        ----------
        file : str
            The audio file location
        duration : float
            Read-out length in seconds. If -1 all audio samples are loaded.

        Returns
        -------
        A 1D tensor with samples of the waveform.
        '''
        num_frames = int(torchaudio.info(file).sample_rate * duration) if duration > 0 else -1
        audio, fs_hz = torchaudio.load(file, channels_first=True, num_frames=num_frames)    # normalized by default
        if len(audio.shape) > 1: audio = torch.mean(audio, axis=0)                          # to mono
        return torchaudio.transforms.Resample(fs_hz, self.fs_hz)(audio)

    def get_num_features(self):
        '''Returns the number of configured audio features.'''        
        if 'mfcc' == self.feat_type:
            return NUM_MFCC
        else:
            return self.mel_bands

    def get_mel_size(self):
        '''Returns configured spectrogram size, i.e. number of audio features and audio frame size.'''
        return self.mel_bands, self.num_frames

    def waveform2spectrograms(self, audio, fps):
        '''Converts a waveform into a series of spectrograms computing the
        Short Time Fourier Transform (STFT) through a sliding window processing.

        Parameters
        ----------
        audio : tensor
            Waveform samples
        fps : float
            Frames per second to determine the stride of the listening window as hop=1/fps

        Returns
        -------
        A list of 2D tensor with the spectrograms of each audio chunk (sliding window).
        '''
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()

        # Mel-STFT settings
        window_length = int(round(self.fs_hz * STFT_WIN_LEN_SECS))
        hop_length = int(round(self.fs_hz * STFT_HOP_SECS))
        fft_length = 2 ** int(np.ceil(np.log(window_length) / np.log(2.0)))

        spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.fs_hz,
            n_fft=fft_length,
            hop_length=hop_length,
            win_length=window_length,
            n_mels=self.mel_bands,
            fmin=MEL_MIN_HZ,
            fmax=MEL_MAX_HZ)

        # Chunks
        CHUNK_WIN_SECS = self.chunk_len_sec
        CHUNK_HOP_SECS = 1.0 / fps

        chunk_win_len = int(round(CHUNK_WIN_SECS / STFT_HOP_SECS))
        chunk_hop_len = int(round(CHUNK_HOP_SECS / STFT_HOP_SECS))
        mel_chunks = stricks.sliding_window_view(spec, (spec.shape[0], chunk_win_len))[0,::chunk_hop_len,:,:]

        # Chunks to tensors
        mel_tensors = []
        for e in range(mel_chunks.shape[0]):
            t = torch.tensor(mel_chunks[e])
            mel_tensors.append(t.float())

        return mel_tensors
