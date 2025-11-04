# -*- coding: utf-8 -*-
import glob
import numpy as np
import torch
import os
import cv2
import argparse
from model.unet_model import UNet, ResNetUNet

if __name__ == "__main__":
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='U-Net prediction script')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'resnetunet'],
                        help='Model type: unet or resnetunet')
    parser.add_argument('--resnet-type', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'],
                        help='ResNet type (only valid when using resnetunet)')
    parser.add_argument('--model-path', type=str, default='best_model.pth',
                        help='Model weight file path')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create model
    if args.model == 'unet':
        print("Creating original UNet model...")
        net = UNet(n_channels=1, n_classes=1)
    else:
        print(f"Creating ResNetUNet model (using {args.resnet_type})...")
        net = ResNetUNet(n_channels=1, n_classes=1, resnet_type=args.resnet_type)
    
    # Move model to specified device and load weights
    net.to(device=device)
    print(f"Loading model weights: {args.model_path}")
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.eval()
    tests_path = glob.glob('data/test/*.png')
    for test_path in tests_path:
        filename = os.path.basename(test_path)
        save_res_path = os.path.join('data/test/result', filename.split('.')[0] + '_res.png')
        img = cv2.imread(test_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        pred = net(img_tensor)
        pred = np.array(pred.data.cpu()[0])[0]
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        cv2.imwrite(save_res_path, pred)
