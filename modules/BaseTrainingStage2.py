import warnings
warnings.filterwarnings('ignore')

import sys
from .AngularGrad.myoptims.tanangulargrad import tanangulargrad
from .AngularGrad.myoptims.cosangulargrad import cosangulargrad

import random
import os
import logging
import librosa 
import numpy as np
import pandas as pd
from librosa.core import resample, to_mono
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.metrics import roc_auc_score

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import GPUStatsMonitor
torch.autograd.set_detect_anomaly(True)



# ================ Dataset ================== 
class CoughDataset(Dataset):
    def __init__(self, df, SR_Effv2, PERIOD_Effv2, SR_PANN, PERIOD_PANN):
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

# ================ Dataloader =================
class CoughDataLightning(LightningDataModule):
    def __init__(self, 
                data_csv, 
                fold, 
                SR_Effv2, 
                PERIOD_Effv2, 
                SR_PANN, 
                PERIOD_PANN, 
                dataloader, 
                train_batch_size,
                train_num_workers,
                val_batch_size,
                val_num_workers,
                seed):
                
        super().__init__()
        self.train_csv = data_csv[data_csv['fold']!=fold].reset_index(drop=True)
        self.val_csv = data_csv[data_csv['fold']==fold].reset_index(drop=True)
        self.dataset_process = dataloader

        self.SR_Effv2 = SR_Effv2
        self.PERIOD_Effv2 = PERIOD_Effv2
        self.SR_PANN = SR_PANN
        self.PERIOD_PANN = PERIOD_PANN

        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.val_batch_size = val_batch_size
        self.val_num_workers = val_num_workers
        self.seed = seed
    
    def prepare_data(self):
        pass 
    @property
    def num_classes(self) -> int:
        return 1 
    def setup(self, *_, **__) -> None:
        self.train_dataset = self.dataset_process(df=self.train_csv, SR_Effv2=self.SR_Effv2, PERIOD_Effv2=self.PERIOD_Effv2, SR_PANN=self.SR_PANN, PERIOD_PANN=self.PERIOD_PANN)
        logging.info(f"training dataset: {len(self.train_dataset)}")

        self.val_dataset = self.dataset_process(df=self.val_csv, SR_Effv2=self.SR_Effv2, PERIOD_Effv2=self.PERIOD_Effv2, SR_PANN=self.SR_PANN, PERIOD_PANN=self.PERIOD_PANN)
        logging.info(f"validation dataset: {len(self.val_dataset)}")
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size ,
                num_workers=self.train_num_workers,
                pin_memory=True,
                drop_last=True, 
                shuffle=True,
                worker_init_fn=np.random.seed(self.seed))
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
                self.val_dataset,
                batch_size=self.val_batch_size ,
                num_workers=self.val_num_workers,
                pin_memory=True,
                drop_last=False,
                shuffle=False,
                )

# ================== Base Training ===================
class BaseTraining(LightningModule):
    def __init__(self, model, lr: float = 1e-4, min_lr: float = 1e-5):
        super().__init__()
        self.model = model
        self.arch = self.model.arch
        self.num_classes = self.model.num_classes
        self.learn_rate = lr
        self.min_lr = min_lr
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")
    
    # ------------ Compute Loss -------------  
    def compute_loss(self, y_hat, y):
        return self.loss(y_hat.view(-1), y.to(float))
    
    # ------------- Training Step ----------------
    def forward(self, x1, x2, SpecAugment=False):
        return self.model(x1, x2, SpecAugment)

    def training_step(self, batch, batch_idx):
        x1, x2, y = batch 
        _,y_hat = self(x1, x2, SpecAugment=False)
        
        loss = self.compute_loss(y_hat, y)
        self.log("loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x1, x2, y = batch 
        _,y_hat = self(x1, x2, SpecAugment=False)
        return {"preds":y_hat.sigmoid().clone().detach().tolist(), 
                "truth":y.clone().detach().tolist()} 
    
    def validation_epoch_end(self, outputs):
        predict_list = [] 
        target_list = []
        for out in outputs:
            predict_list.extend(out['preds'])
            target_list.extend(out['truth'])
        predict_list = np.array([item for sublist in predict_list for item in sublist])
        target_list  = np.array(target_list)
        predict_list = np.nan_to_num(predict_list)
        try:
            AUC_SCORE = roc_auc_score(target_list, predict_list)
            self.log("AUC_SCORE", AUC_SCORE,sync_dist=True,on_epoch=True,prog_bar=True)
        except ValueError:
            self.log("AUC_SCORE", 0.0,sync_dist=True,on_epoch=True,prog_bar=True)
            pass
        
    def configure_optimizers(self):
        optimizer = cosangulargrad(self.model.parameters(), lr=self.learn_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, self.min_lr)
        return [optimizer] #, [scheduler]
    
    def Custom_save_model_pretrained(self,PATH,fold):
        torch.save(self.model.state_dict(), os.path.join(PATH) + f"/fold{fold}.pth")
        print("--> Saved: " + os.path.join(PATH) +  f"/fold{fold}.path")
