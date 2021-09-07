import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class Cnn14_8k(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        
        super(Cnn14_8k, self).__init__() 

        assert sample_rate == 8000
        assert window_size == 256
        assert hop_size == 80
        assert mel_bins == 64
        assert fmin == 50
        assert fmax == 4000

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        # if self.training:
        #     x = self.spec_augmenter(x)

        # # Mixup on spectrogram
        # if self.training and mixup_lambda is not None:
        #     x = do_mixup(x, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=False)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=False)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=False)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=False)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=False)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=False)
        feature_core = x  ### Change by fruit  ###
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        feature = x  ### Change by fruit ###
        x = F.dropout(x, p=0.5, training=False)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=False)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        output_dict = {"feature_core": feature_core,'embedding_nofc': feature,'clipwise_output': clipwise_output, 'embedding': embedding}
        return output_dict


from .Effv2_GemPooling import  EffV2_GemPooling
class Hybrid_GemP_CNN14_8hz(nn.Module):
    def __init__(self, 
                arch = "Hybrid_8hz_8hz_PANNnets", 
                Effv2_Config=None,
                PANN_Config=None,
                WeightsEffv2=None, 
                WeightsPANN="./weights/PANNnets/Cnn14_8k_mAP=0.416.pth", 
                num_classes=None):
        
        super().__init__()
        self.arch = arch 
        self.num_classes = num_classes
        
        # 0. Efficentnetv2 8hz
        self.Effv2GemP_Config = Effv2_Config
        self.BaseEffv2 = EffV2_GemPooling(**self.Effv2GemP_Config)  
        
        if WeightsEffv2 != None:
            self.BaseEffv2.load_state_dict(torch.load(WeightsEffv2)) # Depennd Fold num 
        else:
            print("No Loading Weights Gem8hz base !! --> Submit mode")

        features_Effv2 = self.BaseEffv2.linear.in_features
        self.BaseEffv2.linear = nn.Identity()
        for param in self.BaseEffv2.parameters():
            param.requires_grad = False
        self.BaseEffv2.eval()

        # 1. CNN14 8hz 
        self.PANN_Config= PANN_Config
        self.BasePANN = Cnn14_8k(**self.PANN_Config)
        self.BasePANN.load_state_dict(torch.load(WeightsPANN)["model"])
        features_PANN = self.BasePANN.fc_audioset.in_features
        for param in self.BasePANN.parameters():
            param.requires_grad = False
        self.BasePANN.eval()

        # 2.Linear for Combining Features
        self.Num_Feature = features_Effv2 + features_PANN
        self.Hybrid_Linear = nn.Linear(self.Num_Feature, self.num_classes)
    

    # Input 1: Use Dataloader with Aug
    # input2: Use Dataloader With No Aug
    def forward(self,input1, input2, SpecAugment=False):
        feature_1 = self.BaseEffv2(input1, SpecAugment)
        feature_2 = self.BasePANN(input2)
        embedding_combine = torch.cat((feature_1, feature_2['embedding_nofc']), 1)
        out = self.Hybrid_Linear(embedding_combine)
        return embedding_combine ,out