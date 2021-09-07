import warnings
warnings.filterwarnings('ignore')

import sys
import random
import os
import logging
import librosa 
import numpy as np
import pandas as pd
from librosa.core import resample, to_mono

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class CoughDataset_Submit(Dataset):
    def __init__(self,
                df,
                SR_Effv2, 
                PERIOD_Effv2, 
                SR_PANN,  
                PERIOD_PANN,):
        
        self.df = df
        self.file_names = self.df['file_path'].values
        self.labels = self.df["assessment_result"].values
        self.SR_Effv2 = SR_Effv2
        self.PERIOD_Effv2 = PERIOD_Effv2
        self.SR_PANN = SR_PANN
        self.PERIOD_PANN = PERIOD_PANN

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        wav_path= self.file_names[idx]
        
        # Effv2 Dataloader 
        y, sr = librosa.load(wav_path, sr=self.SR_Effv2, mono=True,res_type="kaiser_fast")
        start = 0
        end = self.SR_Effv2 * self.PERIOD_Effv2
        y_batch1 = y[start:end].astype(np.float32)
        new_audio = np.zeros(self.SR_Effv2 * self.PERIOD_Effv2, dtype=np.float32)
        if self.SR_Effv2 * self.PERIOD_Effv2 != len(y_batch1):            
            new_audio[:len(y_batch1)] = y_batch1
        else:
            new_audio = y_batch1

        # PANN Dataloaders
        audio,_ = librosa.load(wav_path, sr=self.SR_PANN, mono=True,res_type="kaiser_fast")
        start = 0
        end = self.SR_PANN * self.PERIOD_PANN
        y_batch = audio[start:end].astype(np.float32)
        new_audio_PANN = np.zeros(self.PERIOD_PANN * self.SR_PANN, dtype=np.float32)
        if self.SR_PANN * self.PERIOD_PANN != len(y_batch):            
            new_audio_PANN[:len(y_batch)] = y_batch
        else:
            new_audio_PANN = y_batch        
        return new_audio, new_audio_PANN, torch.tensor(self.labels[idx]).float()