from model.unet_model import UNet, ResNetUNet
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
import argparse

def train_net(net, device, data_path, epochs=40, batch_size=1, lr=0.00001):
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    for epoch in range(epochs):
        net.train()
        for image, label in train_loader:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = net(image)
            loss = criterion(pred, label)
            print('Loss/train', loss.item())
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='U-Net training script')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'resnetunet'],
                        help='Model type: unet or resnetunet')
    parser.add_argument('--resnet-type', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'],
                        help='ResNet type (only valid when using resnetunet)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights (only valid when using resnetunet)')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    if args.model == 'unet':
        print("Creating original UNet model...")
        net = UNet(n_channels=1, n_classes=1)
    else:
        print(f"Creating ResNetUNet model (using {args.resnet_type}, pretrained: {args.pretrained})...")
        net = ResNetUNet(n_channels=1, n_classes=1, 
                        resnet_type=args.resnet_type, 
                        pretrained=args.pretrained)
    
    net.to(device=device)
    
    data_path = "data/train/"
    
    print(f"Starting training {args.model} model...")
    train_net(net, device, data_path)