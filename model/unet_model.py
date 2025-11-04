# -*- coding: utf-8 -*-
""" Full assembly of the parts to form the complete network """
"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, resnet101

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class ResNetUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, resnet_type='resnet18', pretrained=False):
        super(ResNetUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        if resnet_type == 'resnet18':
            self.resnet = resnet18(pretrained=pretrained)
        elif resnet_type == 'resnet34':
            self.resnet = resnet34(pretrained=pretrained)
        elif resnet_type == 'resnet50':
            self.resnet = resnet50(pretrained=pretrained)
        elif resnet_type == 'resnet101':
            self.resnet = resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")
        
        if n_channels != 3:
            self.resnet.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.encoder1 = nn.Sequential(
            self.resnet.conv1,    
            self.resnet.bn1,   
            self.resnet.relu  
        )
        
        self.pool = self.resnet.maxpool
        
        self.encoder2 = self.resnet.layer1  
        self.encoder3 = self.resnet.layer2  
        self.encoder4 = self.resnet.layer3  
        self.encoder5 = self.resnet.layer4 
        if resnet_type in ['resnet18', 'resnet34']:
            self.enc2_channels = 64
            self.enc3_channels = 128
            self.enc4_channels = 256
            self.enc5_channels = 512
        else:
            self.enc2_channels = 256
            self.enc3_channels = 512
            self.enc4_channels = 1024
            self.enc5_channels = 2048
        
        self.decoder4 = Up(self.enc5_channels + self.enc4_channels, self.enc4_channels, bilinear)
        self.decoder3 = Up(self.enc4_channels + self.enc3_channels, self.enc3_channels, bilinear)
        self.decoder2 = Up(self.enc3_channels + self.enc2_channels, self.enc2_channels, bilinear)
        self.decoder1 = Up(self.enc2_channels + 64, 64, bilinear) 
        
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        enc1 = self.encoder1(x)      
        enc2 = self.encoder2(self.pool(enc1)) 
        enc3 = self.encoder3(enc2)   
        enc4 = self.encoder4(enc3)   
        enc5 = self.encoder5(enc4)   

        dec4 = self.decoder4(enc5, enc4)  
        dec3 = self.decoder3(dec4, enc3) 
        dec2 = self.decoder2(dec3, enc2)  
        dec1 = self.decoder1(dec2, enc1) 
        
        logits = self.outc(dec1)
        return logits
