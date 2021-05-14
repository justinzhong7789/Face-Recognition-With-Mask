# this program is to find people whose images are not enough
# we decided we at least need 5
# for people who don't have enough images, we augment them to produce more

import numpy as np
import os
from PIL import Image
def addNoise(imageFullPath, savePath):
    img = Image.open(imageFullPath)
    img = np.array(img)
    noiseNum = np.random.randint(224*224/4)
    x = [np.random.randint(224) for i in range(noiseNum)]
    y = [np.random.randint(224) for i in range(noiseNum)]
    for i in range(len(x)):
        noiseAdd = np.random.randint(5, 1000)
        noise = np.random.randint(noiseAdd, size=(3))
        img[x[i]][y[i]] = img[x[i]][y[i]] + noise
    img = np.clip(img, 0, 255)
    img = Image.fromarray(img, "RGB")
    img.save(savePath)


def flipHorizontally(imageFullPath, savePath):
    img = Image.open(imageFullPath)
    img = np.array(img)
    flipped = np.fliplr(img)
    flipped = Image.fromarray(flipped, 'RGB')
    flipped.save(savePath)
    

def rotate(imageFullPath, savePath):
    rotationAngle = 5
    """
    rotationAngle = np.random.randint(-20, 20)
    
    """
    img = Image.open(imageFullPath)
    rotatedImg = img.rotate(rotationAngle)
    rotatedImg.save(savePath)

def darkAblock(imageFullPath, savePath):
    img = Image.open(imageFullPath)
    img = np.array(img)
    width = np.random.randint(0, 50)
    length = np.random.randint(0, 50)
    start_x = np.random.randint(0, 244)
    start_y = np.random.randint(0, 244)
    end_x = (start_x+width) if start_x+width<224 else 223
    end_y = (start_y+length) if start_y+length<224 else 223
    
    for i in range(start_x, end_x):
        for j in range(start_y, end_y):
            img[j][i] = 0 
    finalImg = Image.fromarray(img, 'RGB')
    finalImg.save(savePath)

def randomComb(imageFullPath, savePath):
    if np.random.randint(100)>50:
        addNoise(originalImage, randomName)
    if np.random.randint(100)>50:
        rotate(randomName, randomName)
    if np.random.randint(100)>50:
        flipHorizontally(randomName, randomName)
    if np.random.randint(100)>50:
        darkAblock(randomName, randomName)

if __name__ == "__main__":

# find all person whose images are not enough
    
    maskFoldrAddr = 'self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset'
    personToAugment = os.listdir(maskFoldrAddr)
    # augment
    processedNum = 0
    for person in personToAugment:
        folderFullPath = os.path.join(maskFoldrAddr, person)
        images = os.listdir(folderFullPath)
        imageNum = len(images)
        # choose what augmentation we do
        
        for image in images:
            imageFullPath = os.path.join(folderFullPath, image)
            name, extension = image.split('.')

            saveName = name+"_noise."+extension
            saveFullPath = os.path.join(folderFullPath, saveName)
            addNoise(imageFullPath, saveFullPath)
            
            saveName = name+"_flip."+extension
            saveFullPath = os.path.join(folderFullPath, saveName)
            flipHorizontally(imageFullPath, saveFullPath)

            saveName = name+"_rotate."+extension
            saveFullPath = os.path.join(folderFullPath, saveName)
            rotate(imageFullPath, saveFullPath)
            
            saveName = name+"_dark."+extension
            saveFullPath = os.path.join(folderFullPath, saveName)
            darkAblock(imageFullPath, saveFullPath)

        wantPicNum = 130
        n=1
        while len(os.listdir(folderFullPath))<wantPicNum:
            images = os.listdir(folderFullPath)    
            for image in images:
                imageFullPath = os.path.join(folderFullPath, image)
                saveName = name+"_rotate"+ str(n)+'.'+extension
                saveFullPath = os.path.join(folderFullPath, saveName)
                rotate(imageFullPath, saveFullPath)
                n+=1
                if len(os.listdir(folderFullPath))==wantPicNum:
                    break

