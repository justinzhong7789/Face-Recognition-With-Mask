import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import os
import pathlib
import shutil

root = 'self-built-masked-face-recognition-dataset'
maskDir = 'self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset'
nomaskDir = 'self-built-masked-face-recognition-dataset/AFDB_face_dataset'
nomaskRootAddr = 'self-built-masked-face-recognition-dataset/AFDB_face_dataset_pair'
maskRootAddr = 'self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset_pair'

# [[maskImage, nomaskImage], label]
def getPairLoader(maskRootAddr, nomaskRootAddr, batchSize=32, split=False):
    allmaskImages = datasets.ImageFolder(root=maskDir, transform=transforms.ToTensor())
    allnomaskImages = datasets.ImageFolder(root=nomaskDir, transform=transforms.ToTensor())

    maskClassStartInd = [] 
    maskClassStartInd.append(0)
    for i in range(len(allmaskImages)):
        if i+1 == len(allmaskImages): break
        if allmaskImages[i][1] != allmaskImages[i+1][1]:
            maskClassStartInd.append(i+1)

    nomaskClassStartInd = [] 
    nomaskClassStartInd.append(0)
    for i in range(len(allnomaskImages)):
        if i+1 == len(allnomaskImages): break
        if allnomaskImages[i][1] != allnomaskImages[i+1][1]:
            nomaskClassStartInd.append(i+1)

    allPairs = []
    nomaskInd = 0
    for i in range(len(maskClassStartInd)):
        maskInd = maskClassStartInd[i]
        nomaskInd = nomaskClassStartInd[i]
        for ind in range(5):
            maskImage = allmaskImages[maskInd+ind][0]
            nomaskImage = allnomaskImages[nomaskInd+ind][0]
            pair = [maskImage, nomaskImage]
            allPairs.append([pair, i])
    totalPair = len(allPairs)
    allIndices = list(range(totalPair))
    np.random.shuffle(allIndices)
    if split==True:
        split1 = int(totalPair*0.6)
        split2 = int(totalPair*0.8)
        trainSampler = SubsetRandomSampler(allIndices[:split1])
        valSampler = SubsetRandomSampler(allIndices[split1:split2])
        testSampler = SubsetRandomSampler(allIndices[split2:])
        trainLoader = torch.utils.data.DataLoader(allPairs, batch_size=batchSize, sampler=trainSampler)
        valLoader = torch.utils.data.DataLoader(allPairs, batch_size=batchSize, sampler=valSampler)
        testLoader = torch.utils.data.DataLoader(allPairs, batch_size=batchSize, sampler=testSampler)
        return trainLoader, valLoader, testLoader
    else:
        sampler = SubsetRandomSampler(allIndices)
        return torch.utils.data.DataLoader(allPairs, batch_size=batchSize, sampler=sampler)

# 
def getPairFolder(maskRootAddr, nomaskRootAddr, pairmaskRootAddr, pairnomaskRootAddr):
    if os.path.isdir(pairmaskRootAddr) == False:
        os.mkdir(pairmaskRootAddr)
        os.mkdir(pairnomaskRootAddr)
        for person in os.listdir(maskRootAddr):
            pathMask = os.path.join(pairmaskRootAddr, person)
            pathNoMask = os.path.join(pairnomaskRootAddr, person)
            os.mkdir(pathMask)
            os.mkdir(pathNoMask)
        
        for person in os.listdir(maskRootAddr):
            maskSrc, maskDest= [], []
            nomaskSrc, nomaskDest = [], []
            maskimgPath = os.path.join(maskRootAddr, person)
            maskimgDest = os.path.join(pairmaskRootAddr, person)
            nomaskimgPath = os.path.join(nomaskRootAddr, person)
            nomaskimgDest = os.path.join(pairnomaskRootAddr, person)
            n = 0
            for img in os.listdir(maskimgPath):
                if n==5: break
                maskSrc.append(os.path.join(maskimgPath, img))
                maskDest.append(os.path.join(maskimgDest, img))
                n+=1
            n=0
            for img in os.listdir(nomaskimgPath):
                if n==5: break
                nomaskSrc.append(os.path.join(nomaskimgPath, img))
                nomaskDest.append(os.path.join(nomaskimgDest, img))
            for i in range(len(maskSrc)):
                print(maskSrc[i])
                print(maskDest[i])
                shutil.copyfile(maskSrc[i], maskDest[i])
                shutil.copyfile(nomaskSrc[i], nomaskDest[i])
                


            
if __name__ == "__main__":
    
    #getPairFolder(maskDir, nomaskDir, maskRootAddr, nomaskRootAddr)
    pairLoader = getPairLoader(maskRootAddr, nomaskRootAddr)
        
    for i, data in enumerate(pairLoader):
        print(len(data))
        print(data[0][0].shape)
        print(data[1])
        break
