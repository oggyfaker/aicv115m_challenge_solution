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
                SR_PANN,
                PERIOD_fold0_2,
                PERIOD_fold1,
                PERIOD_fold3,
                PERIOD_fold4,
                PERIOD_PANN):
        self.df = df
        self.file_names = self.df['file_path'].values
        self.labels = self.df["assessment_result"].values
        
        self.SR_PANN = SR_PANN
        self.PERIOD_PANN = PERIOD_PANN
        
        self.SR_Effv2 = SR_Effv2
        self.PERIOD_fold0_2 = PERIOD_fold0_2
        self.PERIOD_fold1 = PERIOD_fold1
        self.PERIOD_fold3 = PERIOD_fold3
        self.PERIOD_fold4 = PERIOD_fold4

    
    def __len__(self):
        return len(self.df)
    
    def get_audio_for_each_fold(self, audio, SR, PERIOD):
        start = 0 
        end = SR*PERIOD
        y_tmp = audio[start:end].astype(np.float32)
        new_audio = np.zeros(SR*PERIOD, dtype=np.float32)
        if SR*PERIOD != len(y_tmp):
            new_audio[:len(y_tmp)] = y_tmp
        else:
            new_audio = y_tmp
        return new_audio
        

    def __getitem__(self, idx: int):
        wav_path = self.file_names[idx]
        
        # Fold DataLoader 
        y, _ = librosa.load(wav_path, sr=self.SR_Effv2, mono=True,res_type="kaiser_fast")
        new_audio_fold0_2 = self.get_audio_for_each_fold(y, self.SR_Effv2, self.PERIOD_fold0_2)
        new_audio_fold1 = self.get_audio_for_each_fold(y, self.SR_Effv2, self.PERIOD_fold1)
        new_audio_fold3 = self.get_audio_for_each_fold(y, self.SR_Effv2, self.PERIOD_fold3)
        new_audio_fold4 = self.get_audio_for_each_fold(y, self.SR_Effv2, self.PERIOD_fold4)
        
        # PANN Dataloaders
        start = 0
        end = self.SR_PANN * self.PERIOD_PANN
        audio,_ = librosa.load(wav_path, sr=self.SR_PANN, mono=True,res_type="kaiser_fast")
        y_batch = audio[start:end].astype(np.float32)
        new_audio_PANN = np.zeros(self.PERIOD_PANN * self.SR_PANN, dtype=np.float32)
        if self.SR_PANN * self.PERIOD_PANN != len(y_batch):            
            new_audio_PANN[:len(y_batch)] = y_batch
        else:
            new_audio_PANN = y_batch

        # Label 
        label = torch.tensor(self.labels[idx]).float()
        
        return new_audio_fold0_2, new_audio_fold1, new_audio_fold3, new_audio_fold4, new_audio_PANN, label