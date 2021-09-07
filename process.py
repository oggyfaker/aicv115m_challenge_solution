import os
import random
import numpy as np
import torch
import pandas as pd
import yaml
import argparse
import joblib
from tqdm import tqdm
from  soundfile import SoundFile
from pydub import AudioSegment
import librosa
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
torch.autograd.set_detect_anomaly(True)
from modules.BaseTrainingStage2 import CoughDataset, CoughDataLightning, BaseTraining


def convert_to_wav(file_path):
    """
    This function is to convert an audio file to .wav file
    Args:
        file_path (str): paths of audio file needed to be convert to .wav file
    Returns:
        new path of .wav file
    """
    ext = file_path.split(".")[-1]
    assert ext in ["mp4", "mp3", "acc"], "The current API does not support handling {} files".format(ext)

    sound = AudioSegment.from_file(file_path, ext)
    wav_file_path = ".".join(file_path.split(".")[:-1]) + ".wav"
    sound.export(wav_file_path, format="wav")

    os.remove(file_path)
    return wav_file_path

## Read info of audio
def Get_Audio_Info(filepath):
    with SoundFile(filepath) as f:
        sr = f.samplerate
        frames = f.frames
        duration = float(frames)/sr
    return {"frames": frames, "sample_rate": sr, "duration": duration}


## Cut the audio with period
def Cut_Audio(audio, SR, Period):
    start = 0
    end = SR*Period
    y_batch = audio[start:end].astype(np.float32)
    new_audio = np.zeros(SR*Period, dtype=np.float32)
    if SR*Period != len(y_batch):
        new_audio[:len(y_batch)] = y_batch
    else:
        new_audio = y_batch
    return new_audio

## Resample and Cut Audio
def Convert_Audio_For_Model(audio_path, audio_info_dict):
    if audio_info_dict["sample_rate"] < 8000:
        audio1, sr1 = librosa.load(audio_path, sr=4000, mono=True,res_type="kaiser_fast")
        audio2, sr2 = librosa.load(audio_path, sr=8000, mono=True,res_type="kaiser_fast")

        new_audio1 = Cut_Audio(audio=audio1, SR=sr1, Period=60)
        new_audio2 = Cut_Audio(audio=audio2, SR=sr2, Period=35)

        return new_audio1, new_audio2
    
    elif audio_info_dict["sample_rate"] >= 8000 and audio_info_dict["sample_rate"] < 48000:
        audio1, sr1 = librosa.load(audio_path, sr=8000, mono=True,res_type="kaiser_fast")
        audio2, sr2 = librosa.load(audio_path, sr=8000, mono=True,res_type="kaiser_fast")

        new_audio1 = Cut_Audio(audio=audio1, SR=sr1, Period=35)
        new_audio2 = Cut_Audio(audio=audio2, SR=sr2, Period=35)

        return new_audio1, new_audio2
    
    else:
        audio1, sr1 = librosa.load(audio_path, sr=48000, mono=True,res_type="kaiser_fast")
        audio2, sr2 = librosa.load(audio_path, sr=32000, mono=True,res_type="kaiser_fast")

        new_audio0 = Cut_Audio(audio=audio1, SR=sr1, Period=35)
        new_audio1 = Cut_Audio(audio=audio1, SR=sr1, Period=15)
        new_audio2 = Cut_Audio(audio=audio1, SR=sr1, Period=20)
        new_audio3 = Cut_Audio(audio=audio1, SR=sr1, Period=25)

        new_audio4 = Cut_Audio(audio=audio2, SR=sr2, Period=35)

        # Fold 0 - 1 - 2 - 3 - 4 - Pann
        return new_audio0, new_audio1, new_audio0, new_audio2, new_audio3, new_audio4


## Create Model
def Get_Model(mode):
    
    if mode == "4hz":
        from modules.models.Gem4hz_8hz import Hybrid_GemP_CNN14_8hz
        config = yaml.load(open("./configs/Gem4hz_CNN8hz.yaml"), Loader=yaml.FullLoader) 
        config["Effv2_Config"]['arch'] = "Gem4hz" 

        Hybrid = Hybrid_GemP_CNN14_8hz( arch= "Hybrid_4hz_8hz_PANNnets",
                                        Effv2_Config=config["Effv2_Config"],
                                        PANN_Config=config["PANN_Config"],
                                        WeightsEffv2=None,
                                        WeightsPANN=config['pretrained_CNN14_8hz'],
                                        num_classes=1)
        model = BaseTraining(model=Hybrid, lr=config['lr'])
        all_ckpt_path = [os.path.join(config["OldWeight_4hz_8hz"],item) for item in sorted(os.listdir(config["OldWeight_4hz_8hz"]))]
        all_ckpt = [torch.load(path) for path in  all_ckpt_path]
        return model.to("cuda").eval(), all_ckpt


    elif mode == "8hz":
        from modules.models.Gem8hz_8hz import Hybrid_GemP_CNN14_8hz
        config = yaml.load(open("./configs/Gem8hz_CNN8hz.yaml"), Loader=yaml.FullLoader) 
        config["Effv2_Config"]['arch'] = "Gem8hz"

        Hybrid = Hybrid_GemP_CNN14_8hz(arch = "Hybrid_8hz_8hz_PANNnets",
                                                Effv2_Config=config["Effv2_Config"],
                                                PANN_Config=config["PANN_Config"],
                                                WeightsEffv2=None,
                                                WeightsPANN=config['pretrained_CNN14_8hz'],
                                                num_classes=1)
        model = BaseTraining(model=Hybrid, lr=config['lr'])
        all_ckpt_path = [os.path.join(config["OldWeight_8hz_8hz"],item) for item in sorted(os.listdir(config["OldWeight_8hz_8hz"]))]
        all_ckpt = [torch.load(path) for path in  all_ckpt_path]
        return model.to("cuda").eval(), all_ckpt


    else:
        from modules.models.Gem48hz_32hz import Hybrid_Effv2GemP_WaveLogMelCNN14
        config = yaml.load(open("./configs/Gem48hz_Wave32hz.yaml"), Loader=yaml.FullLoader)
        config["Effv2_Config"]['arch'] = "Gem48hz"

        Hybrid = Hybrid_Effv2GemP_WaveLogMelCNN14(arch = "Hybrid_48hz_32hz_PANNnets",
                                                    Effv2_Config=config["Effv2_Config"],
                                                    PANN_Config=config["PANN_Config"],
                                                    WeightsEffv2=None,
                                                    WeightsPANN=config['pretrained_WaveLogMel32hz'],
                                                    num_classes=1)
        model = BaseTraining(model=Hybrid, lr=config['lr'])
        all_ckpt_path = [os.path.join(config["OldWeight_48hz_32hz"],item) for item in sorted(os.listdir(config["OldWeight_48hz_32hz"]))]
        all_ckpt = [torch.load(path) for path in  all_ckpt_path]
        return model.to("cuda").eval(), all_ckpt



def predict(audio_path):
    audio_info_dict = Get_Audio_Info(audio_path)
    
    # 48hz 
    if audio_info_dict["sample_rate"] >=48000:
        model, all_ckpt = Get_Model(mode="48hz")
        audio0, audio1, audio2, audio3, audio4, audio_PANN = Convert_Audio_For_Model(audio_path, audio_info_dict)
        
        audio0 = torch.from_numpy(audio0).unsqueeze(0).to("cuda")
        audio1 = torch.from_numpy(audio1).unsqueeze(0).to("cuda")
        audio2 = torch.from_numpy(audio2).unsqueeze(0).to("cuda")
        audio3 = torch.from_numpy(audio3).unsqueeze(0).to("cuda")
        audio4 = torch.from_numpy(audio4).unsqueeze(0).to("cuda")
        audio_PANN = torch.from_numpy(audio_PANN).unsqueeze(0).to("cuda")

        predict_each_ckpt = []
        for fold_id,checkpoint in enumerate(all_ckpt):
            model.load_state_dict(checkpoint['state_dict'])
            with torch.no_grad():
                if fold_id == 0:
                    _,y_preds = model(audio0, audio_PANN, SpecAugment=False)
                elif fold_id == 1:
                    _,y_preds = model(audio1, audio_PANN, SpecAugment=False)
                elif fold_id == 2:
                    _,y_preds = model(audio2, audio_PANN, SpecAugment=False)
                elif fold_id == 3:
                    _,y_preds = model(audio3, audio_PANN, SpecAugment=False)
                else: 
                    _,y_preds = model(audio4, audio_PANN, SpecAugment=False)      
            predict_each_ckpt.append(y_preds.sigmoid().to('cpu').numpy()**0.05)
        
        return np.mean(predict_each_ckpt, axis=0).item()

    # 4hz
    elif audio_info_dict["sample_rate"] < 8000:
        model, all_ckpt = Get_Model(mode="4hz")
        audio, Pann_audio = Convert_Audio_For_Model(audio_path, audio_info_dict)

        audio = torch.from_numpy(audio).unsqueeze(0).to("cuda")
        Pann_audio = torch.from_numpy(Pann_audio).unsqueeze(0).to("cuda")

        predict_each_ckpt = []
        for fold_id,checkpoint in enumerate(all_ckpt):
            model.load_state_dict(checkpoint['state_dict'])
            with torch.no_grad():
                _,y_preds = model(audio,Pann_audio ,SpecAugment=False)
            predict_each_ckpt.append(y_preds.sigmoid().to('cpu').numpy()**0.02)
        return np.mean(predict_each_ckpt, axis=0).item() 
    
    # 8hz
    else:
        model, all_ckpt = Get_Model(mode="8hz")
        audio, Pann_audio = Convert_Audio_For_Model(audio_path, audio_info_dict)
        
        audio = torch.from_numpy(audio).unsqueeze(0).to("cuda")
        Pann_audio = torch.from_numpy(Pann_audio).unsqueeze(0).to("cuda")

        predict_each_ckpt = []
        for fold_id,checkpoint in enumerate(all_ckpt):
            model.load_state_dict(checkpoint['state_dict'])
            with torch.no_grad():
                _,y_preds = model(audio,Pann_audio ,SpecAugment=False)
            predict_each_ckpt.append(y_preds.sigmoid().to('cpu').numpy()**0.4)
        return np.mean(predict_each_ckpt, axis=0).item()

# if __name__ == "__main__":
#     score = predict("/home/fruitai/Desktop/aicv115m/data/aicv115m_final_private_test/private_test_audio_files/49b57e0d-2087-48b8-81a4-c7f1d62ff608.wav")
#     print(score)