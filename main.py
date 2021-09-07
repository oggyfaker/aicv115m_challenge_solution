# Build file main for training and inference
# author: Nguyen Hoang Long 
# 30 Aug 2021 
# Edited by: FruitAI team

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
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
torch.autograd.set_detect_anomaly(True)

def create_args():
    my_parser = argparse.ArgumentParser()
    # Mode and Backbone
    my_parser.add_argument('--Mode', 
                            type=str, 
                            choices=['Train', "Save_Base_Weights", "Create_NewSubmission", "Create_OldSubmission", "CombineNewCsv", "CombineOldCsv"], 
                            help='Mode train or create submission',)
    my_parser.add_argument('--Stage', 
                            type=int, 
                            choices=[1,2],  
                            help='Number Of Training Stagem. Only use stage 2 for submission')
    my_parser.add_argument('--Backbone', 
                            type=str, 
                            choices=['Gem4hz', "Gem8hz", "Gem48hz", "Gem4hz+8hz", "Gem8hz+8hz", "Gem48hz+32hz"], 
                            help='Type Hz backbone',)
    
    # Fold training config 
    my_parser.add_argument('--N_folds', 
                            type=int,  
                            default=5,
                            help='Number Of Training Fold')
    my_parser.add_argument('--Fold', 
                            type=int,
                            default=0,
                            help='Number Of Training Fold')
    
    # Batch size config 
    my_parser.add_argument('--Train_batch_size', 
                            type=int,
                            default=12,
                            help='Number Of Training Fold')
    my_parser.add_argument('--Train_num_workers', 
                            type=int,
                            default=3,
                            help='Number Of Training Fold')
    my_parser.add_argument('--Val_batch_size', 
                            type=int,
                            default=6,
                            help='Number Of Training Fold')
    my_parser.add_argument('--Val_num_workers', 
                            type=int,
                            default=3,
                            help='Number Of Training Fold')
    
    # GPU config
    my_parser.add_argument('--GPU_id', 
                            type=list,
                            default=[0,1],
                            help='GPU id for training. 2 GPUS will give good results')

    return my_parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ==== Get Path of audio folder ==== 
def get_train_path(uuid):
    data_path = yaml.load(open("./configs/Data.yaml"), Loader=yaml.FullLoader)
    return os.path.join(data_path["TRAIN_AUDIO"], f'{uuid}.wav')

def get_test_path(uuid):
    data_path = yaml.load(open("./configs/Data.yaml"), Loader=yaml.FullLoader)
    return os.path.join(data_path["TEST_AUDIO"], f'{uuid}.wav')

def get_private_path(uuid):
    data_path = yaml.load(open("./configs/Data.yaml"), Loader=yaml.FullLoader)
    return os.path.join(data_path["PRIVATE_AUDIO"], f'{uuid}.wav')

# ==== Fold Validation ====  
def StratifiedK_Fold(csv_path,n_folds):
    file_csv= pd.read_csv(csv_path)
    file_csv['file_path'] = file_csv['uuid'].apply(get_train_path) # Add path to csv 
    Fold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=64) # Fixed seed
    for n, (train_index, val_index) in enumerate(Fold.split(file_csv, file_csv["assessment_result"])):
        file_csv.loc[val_index, 'fold'] = int(n)
    file_csv['fold'] = file_csv['fold'].astype(int) # Create the columm fold 
    return file_csv # return csv with columm fold and path 

# ===== Private CSV Processing =====
def get_audio_info(filepath):
    with SoundFile(filepath) as f:
        sr = f.samplerate
        frames = f.frames
        duration = float(frames)/sr
    return {"frames": frames, "sample_rate": sr, "duration": duration}

def Get_Private_Csv():
    private= pd.read_csv(data_path["PRIVATE_TEST"])
    private['file_path'] = private['uuid'].apply(get_private_path)

    thread = joblib.Parallel(4)
    mapper = joblib.delayed(get_audio_info)
    tasks = [mapper(file_path) for file_path in private.file_path]
    private = pd.concat([private, pd.DataFrame(thread(tqdm(tasks)))], axis=1, sort=False)
    return private

# ========= Create Fold Submission ========
def Create_Folder_Submission():
    data_path = yaml.load(open("./configs/Data.yaml"), Loader=yaml.FullLoader)
    if not os.path.exists(data_path['OLD_SUBMISION']):
        os.makedirs(data_path['OLD_SUBMISION'])

    if not os.path.exists(data_path['NEW_SUBMISION']):
        os.makedirs(data_path['NEW_SUBMISION'])


if __name__ == "__main__":
    args = create_args()
    Create_Folder_Submission()

    # ======== Gem4hz Stage-1 Training ========
    if args.Backbone=='Gem4hz' and args.Stage==1:
        from modules.BaseTrainingStage1 import CoughDataset, CoughDataLightning, BaseTraining
        from modules.models.Effv2_GemPooling import *
        data_path = yaml.load(open("./configs/Data.yaml"), Loader=yaml.FullLoader)
        data_csv = StratifiedK_Fold(csv_path=data_path["TRAIN_CSV"], n_folds=args.N_folds) # Load train file csv for training 
    
        # Load yaml file
        config = yaml.load(open("./configs/Gem4hz.yaml"), Loader=yaml.FullLoader) 
        seed = config["seed"][args.Fold]
        seed_everything(seed)

        ## Data Modules
        model_config = config["audio_config"]
        model_config['arch'] = "Base_Gem4hz"
        DataModule = CoughDataLightning(data_csv=data_csv,
                                        fold=args.Fold, 
                                        SR=config["SR"], 
                                        PERIOD=config["PERIOD"] , 
                                        dataloader=CoughDataset, 
                                        train_batch_size=args.Train_batch_size, 
                                        train_num_workers=args.Train_num_workers, 
                                        val_batch_size=args.Val_batch_size, 
                                        val_num_workers=args.Val_num_workers,
                                        seed=seed)
        DataModule.setup()

        ## Setup model 
        effv2s_Gem4hz = EffV2_GemPooling(**model_config)
        model = BaseTraining(model=effv2s_Gem4hz, lr=config['lr'])
        
        if args.Mode=="Train":
            logger = pl.loggers.CSVLogger(save_dir='./weights/logs/', name=f'{model.arch}_{config["PERIOD"]}_fold{args.Fold}')
            StochasticWeightAverage = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0.9, swa_lrs=2e-4)
            ckpt = pl.callbacks.ModelCheckpoint(monitor='AUC_SCORE',
                                                save_top_k=1,
                                                save_last=True,     
                                                filename='checkpoint/{epoch:02d}-{AUC_SCORE:.4f}',
                                                mode='max',)
            trainer = pl.Trainer(
                            accelerator=config["accelerator"],
                            move_metrics_to_cpu=config["move_metrics_to_cpu"],
                            gpus=config["gpus"],
                            plugins=[DDPPlugin(find_unused_parameters=config["find_unused_parameters"]),],
                            callbacks=[ckpt, StochasticWeightAverage],
                            logger=logger,
                            max_epochs=config["max_epochs"],
                            precision=config["precision"], # 16bit autocast 
                            accumulate_grad_batches=config["accumulate_grad_batches"],
                            check_val_every_n_epoch=config["check_val_every_n_epoch"],
                            progress_bar_refresh_rate=1,
                            weights_summary='top',
                            # resume_from_checkpoint=""
                            )
            trainer.fit(model=model, datamodule=DataModule)
        
        elif args.Mode=="Save_Base_Weights":
            directory_save = os.path.join('./weights/1.OnlyModelBase/', 'GemP_4hz')
            if not os.path.exists(directory_save):
                os.makedirs(directory_save)
            
            ## Check any version in folder ? 
            ver_dir = os.path.join('./weights/logs/', f'{model.arch}_{config["PERIOD"]}_fold{args.Fold}')
            if len(os.listdir(ver_dir)) != 0:
                latest_version = sorted(os.listdir(ver_dir))[-1] # Choose latest version
                all_weights_path = os.path.join(os.path.join(ver_dir, latest_version), "checkpoints/checkpoint")
                
                ## Check any weights in version ?
                if len(os.listdir(all_weights_path)) != 0:
                    best_model = os.listdir(all_weights_path)[-1] # Choose the best checkpoint
                    ckpt_path = os.path.join(all_weights_path, best_model)
                    print(f'>> START SAVING PRETRAINED: {ckpt_path}')
                    checkpoint = torch.load(ckpt_path)
                    model.load_state_dict(checkpoint['state_dict'])
                    model.Custom_save_model_pretrained(directory_save, args.Fold)
                    print("\n")
                else:
                    print(f'Please Check Saved Weights In: {all_weights_path}')
            else:
                print(f'No Version In: ./weights/logs/{model.arch}_{config["PERIOD"]}_fold{args.Fold}')

        
    # ======== Gem8hz Stage-1 Training ========
    elif args.Backbone=='Gem8hz' and args.Stage==1:
        from modules.BaseTrainingStage1 import CoughDataset, CoughDataLightning, BaseTraining
        from modules.models.Effv2_GemPooling import *
        
        data_path = yaml.load(open("./configs/Data.yaml"), Loader=yaml.FullLoader)
        data_csv = StratifiedK_Fold(csv_path=data_path["TRAIN_CSV"], n_folds=args.N_folds) # Load train file csv for training  

        # Load yaml file
        config = yaml.load(open("./configs/Gem8hz.yaml"), Loader=yaml.FullLoader) 
        seed = config["seed"][args.Fold]
        seed_everything(seed)

        # Data modules
        model_config = config["audio_config"]
        model_config['arch'] = "Base_Gem8hz"
        DataModule = CoughDataLightning(data_csv=data_csv,
                                        fold=args.Fold, 
                                        SR= config["SR"], 
                                        PERIOD=config["PERIOD"] , 
                                        dataloader=CoughDataset, 
                                        train_batch_size=args.Train_batch_size, 
                                        train_num_workers=args.Train_num_workers, 
                                        val_batch_size=args.Val_batch_size, 
                                        val_num_workers=args.Val_num_workers,
                                        seed=seed)
        DataModule.setup()
        
        ## Setup model 
        effv2s_Gem8hz = EffV2_GemPooling(**model_config)
        model = BaseTraining(model=effv2s_Gem8hz, lr=config['lr'])
        logger = pl.loggers.CSVLogger(save_dir='./weights/logs/', name=f'{model.arch}_{config["PERIOD"]}_fold{args.Fold}')
        StochasticWeightAverage = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0.9, swa_lrs=2e-4)
        ckpt = pl.callbacks.ModelCheckpoint(monitor='AUC_SCORE',
                                            save_top_k=1,
                                            save_last=True,     
                                            filename='checkpoint/{epoch:02d}-{AUC_SCORE:.4f}',
                                            mode='max',)
        if args.Mode=="Train":
            trainer = pl.Trainer(
                        # fast_dev_run=True,
                        accelerator=config["accelerator"],
                        move_metrics_to_cpu=config["move_metrics_to_cpu"],
                        gpus=config["gpus"],
                        plugins=[DDPPlugin(find_unused_parameters=config["find_unused_parameters"]),],
                        callbacks=[ckpt, StochasticWeightAverage],
                        logger=logger,
                        max_epochs=config["max_epochs"],
                        precision=config["precision"], # 16bit autocast 
                        accumulate_grad_batches=config["accumulate_grad_batches"],
                        check_val_every_n_epoch=config["check_val_every_n_epoch"],
                        progress_bar_refresh_rate=1,
                        weights_summary='top',
                        # resume_from_checkpoint=""
                        )
            trainer.fit(model=model, datamodule=DataModule)
        
        elif args.Mode=="Save_Base_Weights":
            ## After training --> Save only model weights
            directory_save = os.path.join('./weights/1.OnlyModelBase/', 'GemP_8hz')
            if not os.path.exists(directory_save):
                os.makedirs(directory_save)
            
            ## Check any version in folder ? 
            ver_dir = os.path.join('./weights/logs/', f'{model.arch}_{config["PERIOD"]}_fold{args.Fold}')
            if len(os.listdir(ver_dir)) != 0:
                latest_version = sorted(os.listdir(ver_dir))[-1] # Choose latest version
                all_weights_path = os.path.join(os.path.join(ver_dir, latest_version), "checkpoints/checkpoint")
                
                ## Check any weights in version ?
                if len(os.listdir(all_weights_path)) != 0:
                    best_model = os.listdir(all_weights_path)[-1] # Choose the best checkpoint
                    ckpt_path = os.path.join(all_weights_path, best_model)
                    print(f'>> START SAVING PRETRAINED: {ckpt_path}')
                    checkpoint = torch.load(ckpt_path)
                    model.load_state_dict(checkpoint['state_dict'])
                    model.Custom_save_model_pretrained(directory_save, args.Fold)
                    print("\n")
                else:
                    print(f'Please Check Saved Weights In: {all_weights_path}')

            else:
                print(f'No Version In: ./weights/logs/{model.arch}_{config["PERIOD"]}_fold{args.Fold}')
        

    # ======== Gem48hz Stage-1 Training ========
    elif args.Backbone=='Gem48hz' and args.Stage==1:
        from modules.BaseTrainingStage1 import CoughDataset, CoughDataLightning, BaseTraining
        from modules.models.Effv2_GemPooling import *
        data_path = yaml.load(open("./configs/Data.yaml"), Loader=yaml.FullLoader)
        data_csv = StratifiedK_Fold(csv_path=data_path["TRAIN_CSV"], n_folds=args.N_folds) # Load train file csv for training  

        # Load yaml file
        config = yaml.load(open("./configs/Gem48hz.yaml"), Loader=yaml.FullLoader) 
        seed = config["seed"][args.Fold]
        seed_everything(seed)

        # Datamodules
        model_config = config["audio_config"]
        model_config['arch'] = "Base_Gem48hz"
        DataModule = CoughDataLightning(data_csv=data_csv,
                                        fold=args.Fold, 
                                        SR=config["SR"], 
                                        PERIOD=config["PERIOD"] , 
                                        dataloader=CoughDataset, 
                                        train_batch_size=args.Train_batch_size, 
                                        train_num_workers=args.Train_num_workers, 
                                        val_batch_size=args.Val_batch_size, 
                                        val_num_workers=args.Val_num_workers,
                                        seed=seed)
        DataModule.setup()

        # Setup model 
        effv2s_Gem48hz = EffV2_GemPooling(**model_config)
        model = BaseTraining(model=effv2s_Gem48hz, lr=config['lr'])
        
        
        logger = pl.loggers.CSVLogger(save_dir='./weights/logs/', name=f'{model.arch}_{config["PERIOD"]}_fold{args.Fold}')
        StochasticWeightAverage = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0.9, swa_lrs=2e-4)
        ckpt = pl.callbacks.ModelCheckpoint(monitor='AUC_SCORE',
                                        save_top_k=1,
                                        save_last=True,     
                                        filename='checkpoint/{epoch:02d}-{AUC_SCORE:.4f}',
                                        mode='max',)
        if args.Mode=="Train":
            trainer = pl.Trainer(
                        # fast_dev_run=True,
                        accelerator=config["accelerator"],
                        move_metrics_to_cpu=config["move_metrics_to_cpu"],
                        gpus=config["gpus"],
                        plugins=[DDPPlugin(find_unused_parameters=config["find_unused_parameters"]),],
                        callbacks=[ckpt, StochasticWeightAverage],
                        logger=logger,
                        max_epochs=config["max_epochs"],
                        precision=config["precision"], # 16bit autocast 
                        accumulate_grad_batches=config["accumulate_grad_batches"],
                        check_val_every_n_epoch=config["check_val_every_n_epoch"],
                        progress_bar_refresh_rate=1,
                        weights_summary='top',
                        # resume_from_checkpoint=""
                        )
            trainer.fit(model=model, datamodule=DataModule)
        
        elif args.Mode=="Save_Base_Weights":
            ## Save only model weights
            directory_save = os.path.join('./weights/1.OnlyModelBase/', 'GemP_48hz')
            if not os.path.exists(directory_save):
                os.makedirs(directory_save)
            
            ## Check any version in folder ? 
            ver_dir = os.path.join('./weights/logs/', f'{model.arch}_{config["PERIOD"]}_fold{args.Fold}')
            if len(os.listdir(ver_dir)) != 0:
                latest_version = sorted(os.listdir(ver_dir))[-1] # Choose latest version
                all_weights_path = os.path.join(os.path.join(ver_dir, latest_version), "checkpoints/checkpoint")
                
                ## Check any weights in version ?
                if len(os.listdir(all_weights_path)) != 0:
                    best_model = os.listdir(all_weights_path)[-1] # Choose the best checkpoint
                    ckpt_path = os.path.join(all_weights_path, best_model)
                    print(f'>> START SAVING PRETRAINED: {ckpt_path}')
                    checkpoint = torch.load(ckpt_path)
                    model.load_state_dict(checkpoint['state_dict'])
                    model.Custom_save_model_pretrained(directory_save, args.Fold)
                    print("\n")
                else:
                    print(f'Please Check Saved Weights In: {all_weights_path}')
            else:
                print(f'No Version In: ./weights/logs/{model.arch}_{config["PERIOD"]}_fold{args.Fold}')


    # ======================  Gem4hz+8hz  Stage-2   Training ======================   
    elif args.Backbone=='Gem4hz+8hz' and args.Stage==2:
        
        from modules.BaseTrainingStage2 import CoughDataset, CoughDataLightning, BaseTraining    
        from modules.models.Gem4hz_8hz import Hybrid_GemP_CNN14_8hz
        
        data_path = yaml.load(open("./configs/Data.yaml"), Loader=yaml.FullLoader)
        data_csv = StratifiedK_Fold(csv_path=data_path["TRAIN_CSV"], n_folds=args.N_folds) # Load train file csv for training  

        # Load yaml file
        config = yaml.load(open("./configs/Gem4hz_CNN8hz.yaml"), Loader=yaml.FullLoader) 
        
        
        seed = config["seed"][args.Fold]
        seed_everything(seed)

        
        ## Setup Config
        Effv2_Config = config["Effv2_Config"]
        Effv2_Config['arch'] = "Gem4hz" # Rename arch 
        PANN_Config = config["PANN_Config"]
        pretrained_Effv2 = os.path.join(config['pretrained_Effv2'], f'fold{args.Fold}.pth')
        
        
        ## Datamodules        
        DataModule = CoughDataLightning(data_csv=data_csv,
                                        fold= args.Fold, 
                                        SR_Effv2= config["SR_Effv2"], 
                                        PERIOD_Effv2= config["PERIOD_Effv2"] ,
                                        SR_PANN=config["SR_PANN"],
                                        PERIOD_PANN= config["PERIOD_PANN"],
                                        dataloader= CoughDataset, 
                                        train_batch_size= args.Train_batch_size, 
                                        train_num_workers=args.Train_num_workers, 
                                        val_batch_size= args.Val_batch_size, 
                                        val_num_workers=args.Val_num_workers,
                                        seed=seed)
        DataModule.setup()

        if args.Mode == "Train":
            if not os.path.exists(pretrained_Effv2):
                print(f"xxx Please Train Model 4hz Fold{args.Fold} first ! xxx")
                print(f'xxx Or Put Pretrained 4hz Fold{args.Fold} In --> {pretrained_Effv2} xxx')
            else:
                ## Setup Model for training
                Hybrid = Hybrid_GemP_CNN14_8hz(arch = "Hybrid_4hz_8hz_PANNnets",
                                                Effv2_Config=Effv2_Config,
                                                PANN_Config=PANN_Config,
                                                WeightsEffv2=pretrained_Effv2,
                                                WeightsPANN=config['pretrained_CNN14_8hz'],
                                                num_classes=DataModule.num_classes)
                model = BaseTraining(model=Hybrid, lr=config['lr'])
                
                logger = pl.loggers.CSVLogger(save_dir='./weights/logs/', name=f'Hybrid_4hz_8hz_fold{args.Fold}')
                ckpt = pl.callbacks.ModelCheckpoint(monitor='AUC_SCORE',
                                                    save_top_k=1,
                                                    save_last=True,     
                                                    filename='checkpoint/{epoch:02d}-{AUC_SCORE:.4f}',
                                                    mode='max',)
                                                    
                trainer = pl.Trainer(# fast_dev_run=True,
                                    accelerator=config["accelerator"],
                                    move_metrics_to_cpu=config["move_metrics_to_cpu"],
                                    gpus=config["gpus"],
                                    plugins=[DDPPlugin(find_unused_parameters=config["find_unused_parameters"]),],
                                    callbacks=[ckpt],
                                    logger=logger,
                                    max_epochs=config["max_epochs"],
                                    precision=config["precision"],
                                    accumulate_grad_batches=config["accumulate_grad_batches"],
                                    check_val_every_n_epoch=config["check_val_every_n_epoch"],
                                    progress_bar_refresh_rate=1,
                                    weights_summary='top', ) # resume_from_checkpoint=""
                
                trainer.fit(model=model, datamodule=DataModule)
        
        # Use new trained weight for submit 
        elif args.Mode != "Train" and args.Mode != "Save_Base_Weights":
            from modules.BaseSubmit4hz import CoughDataset_Submit
            
            Hybrid = Hybrid_GemP_CNN14_8hz(arch = "Hybrid_4hz_8hz_PANNnets",
                                                Effv2_Config=Effv2_Config,
                                                PANN_Config=PANN_Config,
                                                WeightsEffv2=None,
                                                WeightsPANN=config['pretrained_CNN14_8hz'],
                                                num_classes=DataModule.num_classes)
            model = BaseTraining(model=Hybrid, lr=config['lr'])
            model.to("cuda")
            model.eval()


            private = Get_Private_Csv() # Private CSV
            test = private[private["sample_rate"]<=4000].reset_index(drop=True) # Get 4hz only
            dataloader_config = config["Submit_4hz_config"]
            dataloader_config["df"] = test
            test_dataset = CoughDataset_Submit(**dataloader_config)
            test_dataloader = DataLoader(test_dataset,
                                        batch_size=args.Val_batch_size,
                                        num_workers=args.Val_num_workers,
                                        pin_memory=True,
                                        drop_last=False,
                                        shuffle=False,)


            if args.Mode=="Create_OldSubmission":
                all_ckpt_path = [os.path.join(config["OldWeight_4hz_8hz"],item) for item in sorted(os.listdir(config["OldWeight_4hz_8hz"]))]
                all_ckpt = [torch.load(path) for path in  all_ckpt_path]

                probs = []
                for i, (image1, image2,label) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                    image1 = image1.to("cuda")
                    image2 = image2.to("cuda")

                    predict_each_ckpt = []
                    for ckpt_idx,checkpoint in enumerate(all_ckpt):
                        model.load_state_dict(checkpoint['state_dict'])

                        with torch.no_grad():
                            _,y_preds = model(image1,image2 ,SpecAugment=False)
                        predict_each_ckpt.append(y_preds.sigmoid().to('cpu').numpy()**config['OldAlpha']) # 0.03 is best good
                    
                    avg_preds = np.mean(predict_each_ckpt, axis=0) 
                    probs.append(avg_preds)
                new_probs = np.concatenate(probs)
                
                test["assessment_result"] = new_probs
                test[['uuid', 'assessment_result']].to_csv(os.path.join(data_path['OLD_SUBMISION'], 'Gem4hz_8hz.csv'), index=False)
                print(f'>> Saved {os.path.join(data_path["OLD_SUBMISION"], "Gem4hz_8hz.csv")} \n')
            
            
            elif args.Mode=="Create_NewSubmission": 
                NameFolder_4hz_Hybrid = [item for item in os.listdir("./weights/logs/") if "Hybrid_4hz_8hz" in item]
                Path_4hz_Hybrid = [os.path.join("./weights/logs/", item) for item in sorted(NameFolder_4hz_Hybrid)]
                all_ckpt = []

                print("======== Loading New Pretrained Hybrid_4hz_8hz ========")
                for path in Path_4hz_Hybrid:
                    latest_version = sorted(os.listdir(path))[-1]
                    version_folder = os.path.join(path, latest_version, "checkpoints/checkpoint")
                    if len(os.listdir(version_folder)) != 0:
                        last_model = os.listdir(version_folder)[-1]
                        ckpt = torch.load(os.path.join(version_folder, last_model))
                        all_ckpt.append(ckpt)
                        print(f'--> Using: {os.path.join(path, latest_version)}--> {last_model}')
                    else:
                        print(f'--> No Weights: {os.path.join(path, latest_version)}')
                
                # Check Enough Weights Fold ?
                if len(all_ckpt) != args.N_folds:
                    print("-----> Not Enough Fold Pretrained !! --> Please train more Fold Weights")
                
                else:
                    probs = []
                    for i, (image1, image2,label) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                        image1 = image1.to("cuda")
                        image2 = image2.to("cuda")

                        predict_each_ckpt = []
                        for ckpt_idx,checkpoint in enumerate(all_ckpt):
                            model.load_state_dict(checkpoint['state_dict'])

                            with torch.no_grad():
                                _,y_preds = model(image1,image2 ,SpecAugment=False)
                            predict_each_ckpt.append(y_preds.sigmoid().to('cpu').numpy()**config['NewAlpha']) # 0.03 is best good
                        
                        avg_preds = np.mean(predict_each_ckpt, axis=0) 
                        probs.append(avg_preds)
                    new_probs = np.concatenate(probs)
                    
                    test["assessment_result"] = new_probs
                    test[['uuid', 'assessment_result']].to_csv(os.path.join(data_path['NEW_SUBMISION'], 'Gem4hz_8hz.csv'), index=False)
                    print(f'>> Saved {os.path.join(data_path["NEW_SUBMISION"], "Gem4hz_8hz.csv")} \n')
        
        

    # ====================== Gem8hz+8hz Stage-2 Training ======================  
    elif args.Backbone=='Gem8hz+8hz' and args.Stage==2:

        from modules.BaseTrainingStage2 import CoughDataset, CoughDataLightning, BaseTraining    
        from modules.models.Gem8hz_8hz import Hybrid_GemP_CNN14_8hz

        data_path = yaml.load(open("./configs/Data.yaml"), Loader=yaml.FullLoader)
        data_csv = StratifiedK_Fold(csv_path=data_path["TRAIN_CSV"], n_folds=args.N_folds) # Load train file csv for training  

        ## Load yaml file
        config = yaml.load(open("./configs/Gem8hz_CNN8hz.yaml"), Loader=yaml.FullLoader) 
        seed = config["seed"][args.Fold]
        seed_everything(seed)

        
        ## Config model
        Effv2_Config = config["Effv2_Config"]
        Effv2_Config['arch'] = "Gem8hz" # Rename arch 
        PANN_Config = config["PANN_Config"]

        ## Datamodules        
        DataModule = CoughDataLightning(data_csv=data_csv,
                                        fold= args.Fold, 
                                        SR_Effv2= config["SR_Effv2"], 
                                        PERIOD_Effv2= config["PERIOD_Effv2"] ,
                                        SR_PANN=config["SR_PANN"],
                                        PERIOD_PANN= config["PERIOD_PANN"],
                                        dataloader= CoughDataset, 
                                        train_batch_size= args.Train_batch_size, 
                                        train_num_workers=args.Train_num_workers, 
                                        val_batch_size= args.Val_batch_size, 
                                        val_num_workers=args.Val_num_workers,
                                        seed=seed)
        DataModule.setup()
        
        if args.Mode == "Train":
            pretrained_Effv2 = os.path.join(config['pretrained_Effv2'], f'fold{args.Fold}.pth')
            if not os.path.exists(pretrained_Effv2):
                print(f"xxx Please Train Model 8hz Fold{args.Fold} first ! xxx")
                print(f'xxx Or Put Pretrained 8hz Fold{args.Fold} In --> {pretrained_Effv2} xxx')
            else:
                Hybrid = Hybrid_GemP_CNN14_8hz(arch = "Hybrid_8hz_8hz_PANNnets",
                                                Effv2_Config=Effv2_Config,
                                                PANN_Config=PANN_Config,
                                                WeightsEffv2=pretrained_Effv2,
                                                WeightsPANN=config['pretrained_CNN14_8hz'],
                                                num_classes=DataModule.num_classes)
                model = BaseTraining(model=Hybrid, lr=config['lr'])
                
                logger = pl.loggers.CSVLogger(save_dir='./weights/logs/', name=f'Hybrid_8hz_8hz_fold{args.Fold}')
                ckpt = pl.callbacks.ModelCheckpoint(monitor='AUC_SCORE',
                                                    save_top_k=1,
                                                    save_last=True,     
                                                    filename='checkpoint/{epoch:02d}-{AUC_SCORE:.4f}',
                                                    mode='max',)
                trainer = pl.Trainer(# fast_dev_run=True,
                                    accelerator=config["accelerator"],
                                    move_metrics_to_cpu=config["move_metrics_to_cpu"],
                                    gpus=config["gpus"],
                                    plugins=[DDPPlugin(find_unused_parameters=config["find_unused_parameters"]),],
                                    callbacks=[ckpt],
                                    logger=logger,
                                    max_epochs=config["max_epochs"],
                                    precision=config["precision"],
                                    accumulate_grad_batches=config["accumulate_grad_batches"],
                                    check_val_every_n_epoch=config["check_val_every_n_epoch"],
                                    progress_bar_refresh_rate=1,
                                    weights_summary='top', ) # resume_from_checkpoint=""
                trainer.fit(model=model, datamodule=DataModule)
                
        ## Create Submit ##
        elif args.Mode != "Train" and args.Mode != "Save_Base_Weights":
            from modules.BaseSubmit8hz import CoughDataset_Submit
            
            Hybrid = Hybrid_GemP_CNN14_8hz(arch = "Hybrid_8hz_8hz_PANNnets",
                                                Effv2_Config=Effv2_Config,
                                                PANN_Config=PANN_Config,
                                                WeightsEffv2=None,
                                                WeightsPANN=config['pretrained_CNN14_8hz'],
                                                num_classes=DataModule.num_classes)
            model = BaseTraining(model=Hybrid, lr=config['lr'])
            model.to("cuda")
            model.eval()

            # Private CSV
            private = Get_Private_Csv() 
            tmp = private[private["sample_rate"]>=8000].reset_index(drop=True) # Get 8000 <= hz <48000
            test = tmp[tmp["sample_rate"]<48000].reset_index(drop=True)
            dataloader_config = config["Submit_8hz_config"]
            dataloader_config["df"] = test
            test_dataset = CoughDataset_Submit(**dataloader_config)
            test_dataloader = DataLoader(test_dataset,
                                        batch_size=args.Val_batch_size,
                                        num_workers=args.Val_num_workers,
                                        pin_memory=True,
                                        drop_last=False,
                                        shuffle=False,)
                

            if args.Mode=="Create_OldSubmission":
                all_ckpt_path = [os.path.join(config["OldWeight_8hz_8hz"],item) for item in sorted(os.listdir(config["OldWeight_8hz_8hz"]))]
                all_ckpt = [torch.load(path) for path in  all_ckpt_path]

                probs = []
                for i, (image1, image2,label) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                    image1 = image1.to("cuda")
                    image2 = image2.to("cuda")

                    predict_each_ckpt = []
                    for ckpt_idx,checkpoint in enumerate(all_ckpt):
                        model.load_state_dict(checkpoint['state_dict'])
                        
                        with torch.no_grad():
                            _,y_preds = model(image1,image2 ,SpecAugment=False)
                        
                        predict_each_ckpt.append(y_preds.sigmoid().to('cpu').numpy()**config['OldAlpha']) # 0.03 is best good
                    
                    avg_preds = np.mean(predict_each_ckpt, axis=0) 
                    probs.append(avg_preds)
                new_probs = np.concatenate(probs)
                
                test["assessment_result"] = new_probs
                test[['uuid', 'assessment_result']].to_csv(os.path.join(data_path['OLD_SUBMISION'], 'Gem8hz_8hz.csv'), index=False)
                print(f'>> Saved {os.path.join(data_path["OLD_SUBMISION"], "Gem8hz_8hz.csv")} \n')


            elif args.Mode=="Create_NewSubmission": 
                NameFolder_8hz_Hybrid = [item for item in os.listdir("./weights/logs/") if "Hybrid_8hz_8hz" in item]
                Path_8hz_Hybrid = [os.path.join("./weights/logs/", item) for item in sorted(NameFolder_8hz_Hybrid)]
                all_ckpt = []

                print("======== Loading New Pretrained Hybrid_8hz_8hz ========")
                for path in  Path_8hz_Hybrid:
                    latest_version = sorted(os.listdir(path))[-1]
                    version_folder = os.path.join(path, latest_version, "checkpoints/checkpoint")
                    if len(os.listdir(version_folder)) != 0:
                        last_model = os.listdir(version_folder)[-1]
                        ckpt = torch.load(os.path.join(version_folder, last_model))
                        all_ckpt.append(ckpt)
                        print(f'--> Using: {os.path.join(path, latest_version)}--> {last_model}')
                    else:
                        print(f'--> No Weights: {os.path.join(path, latest_version)}')
                
                # Check Enough Weights Fold ?
                if len(all_ckpt) != args.N_folds:
                    print("-----> Not Enough Fold Pretrained !! --> Please train more Fold Weights")
                else:
                    probs = []
                    for i, (image1, image2,label) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                        image1 = image1.to("cuda")
                        image2 = image2.to("cuda")

                        predict_each_ckpt = []
                        for ckpt_idx,checkpoint in enumerate(all_ckpt):
                            model.load_state_dict(checkpoint['state_dict'])
                            
                            with torch.no_grad():
                                _,y_preds = model(image1,image2 ,SpecAugment=False)
                            
                            predict_each_ckpt.append(y_preds.sigmoid().to('cpu').numpy()**config['NewAlpha']) # 0.03 is best good
                        
                        avg_preds = np.mean(predict_each_ckpt, axis=0) 
                        probs.append(avg_preds)
                    new_probs = np.concatenate(probs)
                    
                    test["assessment_result"] = new_probs
                    test[['uuid', 'assessment_result']].to_csv(os.path.join(data_path['NEW_SUBMISION'], 'Gem8hz_8hz.csv'), index=False)
                    print(f'>> Saved {os.path.join(data_path["NEW_SUBMISION"], "Gem8hz_8hz.csv")} \n')

                
    
    # ====================== Gem48hz+32hz - Stage2 - Training ======================  
    elif args.Backbone=='Gem48hz+32hz' and args.Stage==2:
        
        from modules.BaseTrainingStage2 import CoughDataset, CoughDataLightning, BaseTraining    
        from modules.models.Gem48hz_32hz import Hybrid_Effv2GemP_WaveLogMelCNN14

        data_path = yaml.load(open("./configs/Data.yaml"), Loader=yaml.FullLoader)
        data_csv = StratifiedK_Fold(csv_path=data_path["TRAIN_CSV"], n_folds=args.N_folds) # Load train file csv for training  

        # Load yaml file
        config = yaml.load(open("./configs/Gem48hz_Wave32hz.yaml"), Loader=yaml.FullLoader) 
        seed = config["seed"][args.Fold]
        seed_everything(seed)

        ## Datamodules
        DataModule = CoughDataLightning(data_csv=data_csv,
                                        fold= args.Fold, 
                                        SR_Effv2= config["SR_Effv2"], 
                                        PERIOD_Effv2= config["PERIOD_Effv2"] ,
                                        SR_PANN=config["SR_PANN"],
                                        PERIOD_PANN= config["PERIOD_PANN"],
                                        dataloader= CoughDataset, 
                                        train_batch_size= args.Train_batch_size, 
                                        train_num_workers=args.Train_num_workers, 
                                        val_batch_size= args.Val_batch_size, 
                                        val_num_workers=args.Val_num_workers,
                                        seed=seed)
        DataModule.setup()

        ## Setup model
        Effv2_Config = config["Effv2_Config"]
        Effv2_Config['arch'] = "Gem48hz" # Rename arch 
        PANN_Config = config["PANN_Config"]
        pretrained_Effv2 = os.path.join(config['pretrained_Effv2'], f'fold{args.Fold}.pth')
        
        ### Training
        if args.Mode == "Train":
            if not os.path.exists(pretrained_Effv2):
                print(f"xxx Please Train Model 48hz Fold{args.Fold} first ! xxx")
                print(f'xxx Or Put Pretrained of 48hz Fold{args.Fold} in --> {pretrained_Effv2} xxx')
            else:
                Hybrid = Hybrid_Effv2GemP_WaveLogMelCNN14(arch = "Hybrid_48hz_32hz_PANNnets",
                                                    Effv2_Config=Effv2_Config,
                                                    PANN_Config=PANN_Config,
                                                    WeightsEffv2=pretrained_Effv2,
                                                    WeightsPANN=config['pretrained_WaveLogMel32hz'],
                                                    num_classes=DataModule.num_classes)
                model = BaseTraining(model=Hybrid, lr=config['lr'])

                logger = pl.loggers.CSVLogger(save_dir='./weights/logs/', name=f'Hybrid_48hz_32hz_fold{args.Fold}')
                ckpt = pl.callbacks.ModelCheckpoint(monitor='AUC_SCORE',
                                                    save_top_k=1,
                                                    save_last=True,     
                                                    filename='checkpoint/{epoch:02d}-{AUC_SCORE:.4f}',
                                                    mode='max',)
                trainer = pl.Trainer(
                        # fast_dev_run=True,
                        accelerator=config["accelerator"],
                        move_metrics_to_cpu=config["move_metrics_to_cpu"],
                        gpus=config["gpus"],
                        plugins=[DDPPlugin(find_unused_parameters=config["find_unused_parameters"]),],
                        callbacks=[ckpt],
                        logger=logger,
                        max_epochs=config["max_epochs"],
                        precision=config["precision"],
                        accumulate_grad_batches=config["accumulate_grad_batches"],
                        check_val_every_n_epoch=config["check_val_every_n_epoch"],
                        progress_bar_refresh_rate=1,
                        weights_summary='top', ) # resume_from_checkpoint=""
                trainer.fit(model=model, datamodule=DataModule)

        ## Create Submit
        elif args.Mode != "Train" and args.Mode != "Save_Base_Weights":
            
            from modules.BaseSubmit48hz import CoughDataset_Submit
            
            Hybrid = Hybrid_Effv2GemP_WaveLogMelCNN14(arch = "Hybrid_48hz_32hz_PANNnets",
                                                    Effv2_Config=Effv2_Config,
                                                    PANN_Config=PANN_Config,
                                                    WeightsEffv2=None,
                                                    WeightsPANN=config['pretrained_WaveLogMel32hz'],
                                                    num_classes=DataModule.num_classes)
            
            model = BaseTraining(model=Hybrid, lr=config['lr'])
            model.to("cuda")
            model.eval()

            # Private CSV
            private = Get_Private_Csv() 
            test = private[private["sample_rate"]>=48000].reset_index(drop=True)
            dataloader_config = config["Submit_48hz_config"]
            dataloader_config["df"] = test

            test_dataset = CoughDataset_Submit(**dataloader_config)
            test_dataloader = DataLoader(test_dataset,
                                        batch_size=args.Val_batch_size,
                                        num_workers=args.Val_num_workers,
                                        pin_memory=True,
                                        drop_last=False,
                                        shuffle=False,)

            if args.Mode=="Create_OldSubmission":
                all_ckpt_path = [os.path.join(config["OldWeight_48hz_32hz"],item) for item in sorted(os.listdir(config["OldWeight_48hz_32hz"]))]
                all_ckpt = [torch.load(path) for path in  all_ckpt_path]
                
                probs = []
                for i, (audio1,audio2, audio3, audio4, audio_PANN,label) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                    audio1 = audio1.to("cuda")
                    audio2 = audio2.to("cuda")
                    audio3 = audio3.to("cuda")
                    audio4 = audio4.to("cuda")
                    audio_PANN = audio_PANN.to("cuda")

                    predict_each_ckpt = []
                    for fold_id,checkpoint in enumerate(all_ckpt):
                        model.load_state_dict(checkpoint['state_dict'])

                        with torch.no_grad():
                            if fold_id == 0 or fold_id == 2:
                                _,y_preds = model(audio1, audio_PANN, SpecAugment=False)
                            elif fold_id == 1:
                                _,y_preds = model(audio2, audio_PANN, SpecAugment=False)
                            elif fold_id == 3:
                                _,y_preds = model(audio3, audio_PANN, SpecAugment=False)
                            else: 
                                _,y_preds = model(audio4, audio_PANN, SpecAugment=False)

                        predict_each_ckpt.append(y_preds.sigmoid().to('cpu').numpy()**config['OldAlpha'])
                
                    avg_preds = np.mean(predict_each_ckpt, axis=0) 
                    probs.append(avg_preds)

                new_probs = np.concatenate(probs)
                test["assessment_result"] = new_probs
                test[['uuid', 'assessment_result']].to_csv(os.path.join(data_path['OLD_SUBMISION'], 'Gem48hz_32hz.csv'), index=False)
                print(f'>> Saved {os.path.join(data_path["OLD_SUBMISION"], "Gem48hz_32hz.csv")} \n')


            elif args.Mode=="Create_NewSubmission": 
                NameFolder_48hz_Hybrid = [item for item in os.listdir("./weights/logs/") if "Hybrid_48hz_32hz" in item]
                Path_48hz_Hybrid = [os.path.join("./weights/logs/", item) for item in sorted(NameFolder_48hz_Hybrid)]
                all_ckpt = []

                print("======== Loading New Pretrained Hybrid_48hz_32hz ========")
                for path in  Path_48hz_Hybrid:
                    latest_version = sorted(os.listdir(path))[-1]
                    version_folder = os.path.join(path, latest_version, "checkpoints/checkpoint")
                    if len(os.listdir(version_folder)) != 0:
                        last_model = os.listdir(version_folder)[-1]
                        ckpt = torch.load(os.path.join(version_folder, last_model))
                        all_ckpt.append(ckpt)
                        print(f'--> Using: {os.path.join(path, latest_version)}--> {last_model}')
                    else:
                        print(f'--> No Weights: {os.path.join(path, latest_version)}')
                

                # Check Enough Weights Fold ?
                if len(all_ckpt) != args.N_folds:
                    print("-----> Not Enough Fold Pretrained !! --> Please train more Fold Weights")
                
                else:
                    probs = []
                    for i, (audio1,audio2, audio3, audio4, audio_PANN,label) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                        audio1 = audio1.to("cuda")
                        audio2 = audio2.to("cuda")
                        audio3 = audio3.to("cuda")
                        audio4 = audio4.to("cuda")
                        audio_PANN = audio_PANN.to("cuda")

                        predict_each_ckpt = []
                        for fold_id,checkpoint in enumerate(all_ckpt):
                            model.load_state_dict(checkpoint['state_dict'])

                            with torch.no_grad():
                                if fold_id == 0 or fold_id == 2:
                                    _,y_preds = model(audio1, audio_PANN, SpecAugment=False)
                                elif fold_id == 1:
                                    _,y_preds = model(audio2, audio_PANN, SpecAugment=False)
                                elif fold_id == 3:
                                    _,y_preds = model(audio3, audio_PANN, SpecAugment=False)
                                else: 
                                    _,y_preds = model(audio4, audio_PANN, SpecAugment=False)

                            predict_each_ckpt.append(y_preds.sigmoid().to('cpu').numpy()**config['NewAlpha'])
                    
                        avg_preds = np.mean(predict_each_ckpt, axis=0) 
                        probs.append(avg_preds)

                    new_probs = np.concatenate(probs)
                    test["assessment_result"] = new_probs
                    test[['uuid', 'assessment_result']].to_csv(os.path.join(data_path['NEW_SUBMISION'], 'Gem48hz_32hz.csv'), index=False)
                    print(f'>> Saved {os.path.join(data_path["NEW_SUBMISION"], "Gem48hz_32hz.csv")} \n')



    # Combine New Csv
    elif args.Mode == "CombineNewCsv":
        data_path = yaml.load(open("./configs/Data.yaml"), Loader=yaml.FullLoader)
        file = pd.read_csv(data_path["PRIVATE_TEST"]).reset_index(drop=True)
        file2 = [pd.read_csv(os.path.join(data_path["NEW_SUBMISION"], item)).reset_index(drop=True) for item in os.listdir(data_path["NEW_SUBMISION"])]
        result = pd.concat(file2).reset_index(drop=True)
        
        for idx in range(len(result)):
            uuid = result.iloc[idx].uuid
            label = result.iloc[idx].assessment_result
            file.loc[file['uuid']==uuid, 'assessment_result'] = label
        
        file.to_csv(os.path.join(data_path["NEW_SUBMISION"],'results.csv'), index=False)
        print(f'>> Final New Result --> {os.path.join(data_path["NEW_SUBMISION"],"results.csv")}')
    
    elif args.Mode == "CombineOldCsv":
        data_path = yaml.load(open("./configs/Data.yaml"), Loader=yaml.FullLoader)
        file = pd.read_csv(data_path["PRIVATE_TEST"]).reset_index(drop=True)
        file2 = [pd.read_csv(os.path.join(data_path["OLD_SUBMISION"], item)).reset_index(drop=True) for item in os.listdir(data_path["OLD_SUBMISION"])]
        result = pd.concat(file2).reset_index(drop=True)
        
        for idx in range(len(result)):
            uuid = result.iloc[idx].uuid
            label = result.iloc[idx].assessment_result
            file.loc[file['uuid']==uuid, 'assessment_result'] = label
        
        file.to_csv(os.path.join(data_path["OLD_SUBMISION"],'results.csv'), index=False)
        print(f'>> Final Old Result --> {os.path.join(data_path["OLD_SUBMISION"],"results.csv")}')