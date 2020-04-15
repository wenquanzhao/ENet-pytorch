import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import CamVid, showImage 
from torch.utils.data import DataLoader

from ENet import ENet

import time
import os

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# training parameters
trainFlag = True
N = 10
num_workers = 4
C = 12
restore = True
lr = 0.01
epochs = 500
checkpointDir = './checkpoint'
logInterval = 1
logFile = './checkpoint/Stats'

def train(modelPath):
    trainDataset = CamVid()
    trainLoader = DataLoader(trainDataset, 
                             batch_size = N,
                             shuffle = True,
                             num_workers = num_workers,
                             drop_last = True)
    print(trainLoader)
    # instantiation the ENet
    efficientNet = ENet(C)
    if restore:
        efficientNet.load_state_dict(torch.load(modelPath))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(efficientNet.parameters(),
                                lr = lr,
                                momentum = 0.9)
    os.makedirs(checkpointDir, exist_ok = True)
    efficientNet.train()
    iteration = 0 
    for e in range(epochs):
        totalLoss = 0
        for batchID, batchData in enumerate(trainLoader):
            # get the inputs; data is a list of [image, semantic]
            inputs, labels = batchData['image'], batchData['semantic']
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = efficientNet(inputs.float())

            #########################################
            #annotation = outputs[0,:,:,:].squeeze(0)
            ## combine to one dimension
            #Annot = annotation.data.max(0)[1].cpu().numpy()
            #print(Annot, np.max(Annot), np.min(Annot))
            #plt.imshow(Annot)
            #input()
            #########################################
            
            loss = criterion(outputs, labels.long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(efficientNet.parameters(), 3.0)
            optimizer.step()
            
            # print statistis
            totalLoss += loss.item()
            iteration += 1
            
            if (batchID + 1) % logInterval == 0:
                mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.4f}\tTLoss:{6:.4f}\t\n".format(time.ctime(), e+1,
                        batchID+1, 
                        len(trainDataset)// N, 
                        iteration,
                        loss,
                        totalLoss / (batchID + 1))
                print(mesg)
                if logFile is not None:
                    with open(logFile,'a') as f:
                        f.write(mesg)
    #save model
    # Checkout to eval to save model
    # efficientNet.eval().cpu()
    print("====not eval() before save====")
    efficientNet.cpu()
    save_model_filename = "final_epoch_" + str(e + 1) + "_batch_id_" + str(batchID + 1) + ".pth"
    save_model_path = os.path.join(checkpointDir, save_model_filename)
    torch.save(efficientNet.state_dict(), save_model_path)
    
    print("\nDone, trained model saved at", save_model_path)
def test(modelPath):
    testDataset = CamVid()
    testLoader = DataLoader(testDataset, 
                             batch_size = N,
                             shuffle = True,
                             num_workers = num_workers,
                             drop_last = True)
    
    eNet = ENet(C)
    eNet.load_state_dict(torch.load(modelPath))
    # eNet.eval()
    for e in range(4):
        batch_avg_EER = 0
        for batchID, batchData in enumerate(testLoader):
            inputs, labels = batchData['image'], batchData['semantic']
            inputs, labels = inputs.to(device), labels.to(device)
            
            with torch.no_grad():
                outputs = eNet(inputs.float())
                
            plt.imshow(outputs.numpy()[0, 0,:,:])
            # outputs = eNet(inputs.float())
            '''
            Check the training process
            '''
            annotation = outputs[0,:,:,:].squeeze(0)
            # combine to one dimension
            Annot = annotation.data.max(0)[1].cpu().numpy()
            # print(Annot, np.max(Annot), np.min(Annot))
            showImage(Annot)

if __name__ == '__main__':
    modelPath = './checkpoint/final_epoch_500_batch_id_2.pth'
    if trainFlag:
        print('Training the model')
        train(modelPath)
    else:
        print('Testing the model')
        test(modelPath)