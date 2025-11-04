# -*- coding: utf-8 -*-
import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random

class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # Initialization function, read all images under data_path
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))

    def augment(self, image, flipCode):
        # Use cv2.flip for data augmentation, flipCode=1 for horizontal flip, 0 for vertical flip, -1 for horizontal+vertical flip
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # Read image according to index
        image_path = self.imgs_path[index]
        # Generate label_path based on image_path
        label_path = image_path.replace('image', 'label')
        # Read training image and label image
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # Convert data to single-channel images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # Process label, change pixel values of 255 to 1
        if label.max() > 1:
            label = label / 255
        # Random data augmentation, no processing when flipCode is 2
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label

    def __len__(self):
        # Return the size of the training set
        return len(self.imgs_path)

    
if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("data/train/")
    print("Number of data samples:", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2,
                                               shuffle=True)
