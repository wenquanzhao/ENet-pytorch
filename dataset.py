import glob
import numpy as np
import os
import random
from random import shuffle # not used

import torch
from torch.utils.data import Dataset

import cv2
import matplotlib.pyplot as plt
from PIL import Image

# parameters handle function
# from hparam import hparam as hp

def showImage(image):
    """Show image"""
    plt.imshow(image)
    #plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

class cattleDataset(Dataset):
    def __init__(self, training = False):
        self.training = training
        if self.training:
            print("[INFO:] using training dataset.\n")
            self.image_path = '../Dataset/cattleDataset/picture/*/*.png'
            self.label_path = '../Dataset/cattleDataset/mask/*/*.png'
        else:
            print("[INFO:] using testing dataset.\n")
            self.image_path = '../Dataset/cattleDataset/test/*/*.png'
            self.label_path = '../Dataset/cattleDataset/testannot/*/*.png'
            
        self.imageData = glob.glob(os.path.dirname(self.image_path))
        self.semanticData = glob.glob(os.path.dirname(self.label_path))
        
    def __len__(self):
        return len(self.imageData)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.imageData[idx]
        semantic_name = self.semanticData[idx]
        #print('='*15, 'Check image and semantic', '='*15)
        #print(image_name, semantic_name)
        #print('='*50)
        
        image = cv2.imread(image_name)
        semantic = Image.open(semantic_name) # cv2 cannot read the annot
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        semantic = np.array(semantic)
        
        # Resizing using nearest neighbor method
        # image_rgb = cv2.resize(image_rgb, (h, w), cv2.INTER_NEAREST)
        
        # Change image format to C * H * W, semantic has only one channel
        image_rgb = image_rgb.transpose((2, 0, 1))
        
        sample = {'image': image_rgb, 'semantic': semantic}
        
        return sample
    
class CamVid(Dataset):
    def __init__(self, training = True):
        self.training = training
        if self.training:
            print("I am here")
            self.image_path = './CamVid/train/*/*.png'
            self.label_path = './CamVid/trainannot/*/*.png'
        else:
            self.image_path = './CamVid/test/*/*.png'
            self.label_path = './CamVid/testannot/*/*.png'
        
        self.imageData = glob.glob(os.path.dirname(self.image_path))
        self.semanticData = glob.glob(os.path.dirname(self.label_path))
        
        # To Do:
        print("To Do: Shuffle the data set")
        # shuffle the dataset 
        # shuffle(self.imageData)
        
    def __len__(self):
        return len(self.imageData)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        image_name = self.imageData[idx]
        semantic_name = self.semanticData[idx]
        #print('='*15, 'Check image and semantic', '='*15)
        #print(image_name, semantic_name)
        #print('='*50)
        
        image = cv2.imread(image_name)
        # cv2 cannot read the annot
        # semantic = cv2.imread(semantic_name)
        semantic = Image.open(semantic_name)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #semantic_rgb = cv2.cvtColor(semantic, cv2.COLOR_BGR2RGB)
        semantic = np.array(semantic)
        
        # Resizing using nearest neighbor method
        # image_rgb = cv2.resize(image_rgb, (h, w), cv2.INTER_NEAREST)
        
        # Change image format to C * H * W, semantic has only one channel
        image_rgb = image_rgb.transpose((2, 0, 1))
        
        sample = {'image': image_rgb, 'semantic': semantic}
        
        return sample
        
if __name__=="__main__":
    #train_dataset = CamVid()
    train_dataset = cattleDataset()
    fig = plt.figure()
    print(len(train_dataset))
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        
        print('='*50)
        #print(sample['image'])
        #print(sample['semantic'], sample['semantic'].max(), sample['semantic'].min())
        print(i, sample['image'].shape, sample['semantic'].shape)
        print('='*50)
        
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        # showImage(sample['image'].transpose((1, 2, 0)))
        showImage(sample['semantic'])

        if i == 3:
            plt.show()
            break