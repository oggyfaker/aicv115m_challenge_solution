import sys
import timm
import audioread
import logging
import os
import random
import time
import warnings
import librosa
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.nn.parameter import Parameter

""" =============== Preprocess ============= """

class DFTBase(nn.Module):
    def __init__(self):
        super(DFTBase, self).__init__()

    def dft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(-2 * np.pi * 1j / n)
        W = np.power(omega, x * y)
        return W

    def idft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(2 * np.pi * 1j / n)
        W = np.power(omega, x * y)
        return W
    
class STFT(DFTBase):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='reflect', freeze_parameters=True):
        super(STFT, self).__init__()
        assert pad_mode in ['constant', 'reflect']
        self.n_fft = n_fft
        self.center = center
        self.pad_mode = pad_mode
        if win_length is None:
            win_length = n_fft
        if hop_length is None:
            hop_length = int(win_length // 4)
        fft_window = librosa.filters.get_window(window, win_length, fftbins=True)
        fft_window = librosa.util.pad_center(fft_window, n_fft)
        self.W = self.dft_matrix(n_fft)
        out_channels = n_fft // 2 + 1
        self.conv_real = nn.Conv1d(in_channels=1, out_channels=out_channels, 
            kernel_size=n_fft, stride=hop_length, padding=0, dilation=1, 
            groups=1, bias=False)
        self.conv_imag = nn.Conv1d(in_channels=1, out_channels=out_channels, 
            kernel_size=n_fft, stride=hop_length, padding=0, dilation=1, 
            groups=1, bias=False)
        self.conv_real.weight.data = torch.Tensor(
            np.real(self.W[:, 0 : out_channels] * fft_window[:, None]).T)[:, None, :]
        self.conv_imag.weight.data = torch.Tensor(
            np.imag(self.W[:, 0 : out_channels] * fft_window[:, None]).T)[:, None, :]
        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, input):
        x = input[:, None, :]   # (batch_size, channels_num, data_length)
        if self.center:
            x = F.pad(x, pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)
        real = self.conv_real(x)
        imag = self.conv_imag(x)
        # (batch_size, n_fft // 2 + 1, time_steps)
        real = real[:, None, :, :].transpose(2, 3)
        imag = imag[:, None, :, :].transpose(2, 3)
        # (batch_size, 1, time_steps, n_fft // 2 + 1)
        return real, imag
     
class Spectrogram(nn.Module):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, 
        window='hann', center=True, pad_mode='reflect', power=2.0, 
        freeze_parameters=True):
        super(Spectrogram, self).__init__()
        self.power = power
        self.stft = STFT(n_fft=n_fft, hop_length=hop_length, 
            win_length=win_length, window=window, center=center, 
            pad_mode=pad_mode, freeze_parameters=True)
    def forward(self, input):
        (real, imag) = self.stft.forward(input)
        # (batch_size, n_fft // 2 + 1, time_steps)
        spectrogram = real ** 2 + imag ** 2
        if self.power == 2.0:
            pass
        else:
            spectrogram = spectrogram ** (power / 2.0)
        return spectrogram
    
class LogmelFilterBank(nn.Module):
    def __init__(self, sr=32000, n_fft=2048, n_mels=64, fmin=50, fmax=14000, is_log=True, 
        ref=1.0, amin=1e-10, top_db=80.0, freeze_parameters=True):
        super(LogmelFilterBank, self).__init__()
        self.is_log = is_log
        self.ref = ref
        self.amin = amin
        self.top_db = top_db
        self.melW = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
            fmin=fmin, fmax=fmax).T
        # (n_fft // 2 + 1, mel_bins)
        self.melW = nn.Parameter(torch.Tensor(self.melW))
        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, input):
        mel_spectrogram = torch.matmul(input, self.melW)
        if self.is_log:
            output = self.power_to_db(mel_spectrogram)
        else:
            output = mel_spectrogram
        return output
    def power_to_db(self, input):
        ref_value = self.ref
        log_spec = 10.0 * torch.log10(torch.clamp(input, min=self.amin, max=np.inf))
        log_spec -= 10.0 * np.log10(np.maximum(self.amin, ref_value))
        if self.top_db is not None:
            if self.top_db < 0:
                raise ParameterError('top_db must be non-negative')
            log_spec = torch.clamp(log_spec, min=log_spec.max().item() - self.top_db, max=np.inf)
        return log_spec

class DropStripes(nn.Module):
    def __init__(self, dim, drop_width, stripes_num):
        super(DropStripes, self).__init__()
        assert dim in [2, 3]    # dim 2: time; dim 3: frequency
        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num
    def forward(self, input):
        assert input.ndimension() == 4
        if self.training is False:
            return input
        else:
            batch_size = input.shape[0]
            total_width = input.shape[self.dim]
            for n in range(batch_size):
                self.transform_slice(input[n], total_width)
            return input
    def transform_slice(self, e, total_width):
        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]
            if self.dim == 2:
                e[:, bgn : bgn + distance, :] = 0
            elif self.dim == 3:
                e[:, :, bgn : bgn + distance] = 0
               
class SpecAugmentation(nn.Module):
    def __init__(self, time_drop_width, time_stripes_num, freq_drop_width, 
        freq_stripes_num):
        super(SpecAugmentation, self).__init__()
        self.time_dropper = DropStripes(dim=2, drop_width=time_drop_width, 
            stripes_num=time_stripes_num)
        self.freq_dropper = DropStripes(dim=3, drop_width=freq_drop_width, 
            stripes_num=freq_stripes_num)
    def forward(self, input):
        x = self.time_dropper(input)
        x = self.freq_dropper(x)
        return x



""" ================== ARCH =================== """
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)

class EffV2_GemPooling(nn.Module):
    def __init__(self, arch:str, pretrained:bool, channels:int, num_classes:int, sample_rate:int, window_size:int, hop_size:int, mel_bins:int, fmin:int, fmax:int):
        super().__init__()
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.interpolate_ratio = 32  # Downsampled ratio
        
        self.arch = arch
        self.num_classes = num_classes
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, 
                                                 hop_length=hop_size, 
                                                 win_length=window_size, 
                                                 window=window, 
                                                 center=center, 
                                                 pad_mode=pad_mode, 
                                                 freeze_parameters=True)
        
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, 
                                                 n_fft=window_size, 
                                                 n_mels=mel_bins, 
                                                 fmin=fmin, 
                                                 fmax=fmax, 
                                                 ref=ref, 
                                                 amin=amin, 
                                                 top_db=top_db, 
                                                 freeze_parameters=True)
        
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, 
                                               time_stripes_num=2, 
                                               freq_drop_width=8, 
                                               freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(mel_bins)
        
       
        self.base_model = timm.create_model('tf_efficientnetv2_s_in21k', pretrained=pretrained, in_chans=channels) 
        n_features = self.base_model.classifier.in_features
        self.base_model.global_pool = nn.Identity()
        self.base_model.classifier = nn.Identity()
        self.GeM_Pooling = GeM(p=3, eps=1e-6) # Pooling New 
        self.linear = nn.Linear(n_features, num_classes)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.linear)
    

    def preprocess(self, input, SpecAugment=False):
        x = self.spectrogram_extractor(input) #[batch,channel=1,time_steps, freq_bins]
        x = self.logmel_extractor(x)  #[batch, 1, time_steps, mel_bins]
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if SpecAugment:
            x = self.spec_augmenter(x)
        return x

    def forward(self, x, SpecAugment=False):
        batch_size = x.shape[0] # [batch_size, data_length]
        x = self.preprocess(x, SpecAugment) # (batch_size, channels, time, frequency)
        x = self.base_model(x)  # (batch_size, channels, time, frequency)
        pooled_feature = self.GeM_Pooling(x)
        out = self.linear(pooled_feature.view(batch_size, -1))
        return out


if __name__ == "__main__":
    model_config = {"arch": "EfficentNetV2S", 
                    "pretrained" :True, 
                    "channels": 1,
                    "num_classes" :1,
                    "sample_rate": 48000, 
                    "window_size": 1024, 
                    "hop_size": 320, 
                    "mel_bins": 128, 
                    "fmin": 50, 
                    "fmax": 24000,
                }
    model = EffV2_GemPooling(**model_config)
    example = torch.randn(8, 48000*15).cuda()
    model.eval()
    model.cuda()
    out= model(example)
    print(out.shape)

    print("Number parameter: ",sum(p.numel() for p in model.parameters()))
    