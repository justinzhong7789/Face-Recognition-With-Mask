
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt




# Old Model
# This model is used to do classification of faces.
# Input: features of faces derived from pretrained ResNet 50, after multiplying
# with FDM to discard mask-covered features
# Output: classification result

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # only need a single fc layer
        self.fc1 = nn.Linear(2048, 380)

    def forward(self, x):
        # Classification
        x = x.view(-1, 2048) # flatten the input
        x = self.fc1(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x


# New Model
# This is the model used to train both FDM and ANN together in one model
# Input: a pair of features derived from pretrained ResNet 50
# Output: 1. fdm: array with the same dimension as the features
#         2. a pair of filtered features
#         3. classification result

class FDMClassifier(nn.Module):
    def __init__(self):
        super(FDMClassifier, self).__init__()
        # 1 * 1 convolution to maintain the size
        self.conv1 = nn.Conv2d(2048, 2048, 1) #in_channels, out_chanels, kernel_size
        # only need a single fc layer
        self.fc1 = nn.Linear(2048, 380)

    def forward(self, mask, nomask, train):
        # generate fdm
        fdm = F.relu(self.conv1(mask - nomask))

        # apply fdm to do filtering
        mask_filtered = mask * fdm
        nomask_filtered = nomask * fdm

        if train == True:
            # training classification, using unmasked faces
            x = nomask_filtered.view(-1, 2048) # flatten the input
            x = self.fc1(x)
            x = x.squeeze(1) # Flatten to [batch_size]
            return (fdm, mask_filtered, nomask_filtered, x)

        else:
            # validation/testing classification, using masked faces
            x = mask_filtered.view(-1, 2048) # flatten the input
            x = self.fc1(x)
            x = x.squeeze(1) # Flatten to [batch_size]
            return (fdm, mask_filtered, nomask_filtered, x)

def getNewOneToOnePairLoader(maskRootAddr, nomaskRootAddr, batchSize=32, numPeople=20):
    totalClassNum = 40
    wantClass = totalClassNum-numPeople
    
    allmaskImages = datasets.ImageFolder(root=maskRootAddr, transform=transforms.ToTensor())
    # get index of the first occurence of our desired class
    wantmaskInd = wantnomaskInd = 0
    for i in range(len(allmaskImages)):
        if allmaskImages[1] == wantClass:
            wantmaskInd = i
            break
            
    allnomaskImages = datasets.ImageFolder(root=nomaskRootAddr, transform=transforms.ToTensor())
    for i in range(len(allnomaskImages)):
        if allmaskImages[1] == wantClass:
            wantnomaskInd = i
            break
            
    train_pair_list, val_pair_list, test_pair_list = [], [], []
    for clas in range(numPeople):
        maskEndInd = nomaskEndInd = 0
        for i in range(wantmaskInd, len(allmaskImages)):
            if i+1 != len(allmaskImages):                
                if allmaskImages[i][1]!=allmaskImages[i+1][1]:
                    maskEndInd = i+1
                    break
            else:
                maskEndInd = -1
                break
        classmaskImages = []        
        for i in range(wantmaskInd, maskEndInd if maskEndInd!=-1 else len(allmaskImages)):
            classmaskImages.append(allmaskImages[i])
        wantmaskInd = maskEndInd
        
        for i in range(wantnomaskInd, len(allnomaskImages)):
            if i+1 != len(allnomaskImages):                
                if allnomaskImages[i][1]!=allnomaskImages[i+1][1]:
                    nomaskEndInd = i+1
                    break
            else:
                nomaskEndInd = -1
                break
        classnomaskImages = []
        for i in range(wantnomaskInd, nomaskEndInd if nomaskEndInd!=-1 else len(allnomaskImages)):
            classnomaskImages.append(allnomaskImages[i])
        wantnomaskInd = nomaskEndInd
        # next  pair up
        i = 0 
        label = classnomaskImages[0][1]
        lenMaskImg = len(classnomaskImages)
        while i<100:
            train_pair_list.append(([classnomaskImages[i%lenMaskImg][0], classmaskImages[i][0]], label))
            i+=1
        while i<130:
            test_pair_list.append(([classnomaskImages[i%lenMaskImg][0], classmaskImages[i][0]], label))
            i+=1
        while i<160:
            val_pair_list.append(([classnomaskImages[i%lenMaskImg][0], classmaskImages[i][0]], label))
            i+=1
            
    trainInd = list(range(len(train_pair_list)))
    np.random.shuffle(trainInd)
    valInd = list(range(len(val_pair_list)))
    np.random.shuffle(valInd)
    testInd = list(range(len(test_pair_list)))
    np.random.shuffle(testInd)
    trainSampler = SubsetRandomSampler(trainInd)
    valSampler = SubsetRandomSampler(valInd)
    testSampler = SubsetRandomSampler(testInd)
    trainLoader = torch.utils.data.DataLoader(train_pair_list, batch_size=batchSize, sampler=trainSampler)
    valLoader = torch.utils.data.DataLoader(val_pair_list, batch_size=batchSize, sampler=valSampler)
    testLoader = torch.utils.data.DataLoader(test_pair_list, batch_size=batchSize, sampler=valSampler)
    return trainLoader, valLoader, testLoader

def getNewTrainANNLoader(maskAddr, nomaskAddr, batchsize=32, numPeople=20):

    allnomaskImages = datasets.ImageFolder(root=nomaskAddr, transform=transforms.ToTensor())
    wantInd = 0
    for i in range(len(allnomaskImages)):
        if allnomaskImages[i][1]==numPeople:
            wantInd = i
            break
            
    trainInd = list(range(wantInd))
    np.random.shuffle(trainInd)
    trainSampler = SubsetRandomSampler(trainInd)
    trainLoader = torch.utils.data.DataLoader(allnomaskImages, batch_size=batchsize, sampler=trainSampler)


    allmaskImages = datasets.ImageFolder(root=maskAddr, transform=transforms.ToTensor())
    for i in range(len(allmaskImages)):
        if allmaskImages[1]==numPeople:
            wantInd = i
            break
            
    allInd = list(range(wantInd))
    np.random.shuffle(allInd)
    
    valInd = allInd[:int(0.5*len(allInd))]
    testInd = allInd[int(0.5*len(allInd)):]
    valSampler = SubsetRandomSampler(valInd)
    testSampler = SubsetRandomSampler(testInd)
    
    valLoader = torch.utils.data.DataLoader(allmaskImages, batch_size=batchsize, sampler=valSampler)
    testLoader = torch.utils.data.DataLoader(allmaskImages, batch_size=batchsize, sampler=testSampler)
    return trainLoader, valLoader, testLoader


# New get_accuracy function
def get_accuracy(pretrained_cnn, fdm_classifier_model, train, data_loader, batch_size):
    correct = 0
    total = 0

    for i, data in enumerate(data_loader):
        # Get the inputs
        inputs, labels = data
        mask = inputs[0]
        nomask = inputs[1]

        if nomask.shape[0] != batch_size:
            continue


        mask_conv = pretrained_cnn(mask)
        nomask_conv = pretrained_cnn(nomask)

        #############################################
        #To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
            mask_conv = mask_conv.cuda()
            nomask_conv = nomask_conv.cuda()
            labels = labels.cuda()
        #############################################

        fdm, mask_filtered, nomask_filtered, output = fdm_classifier_model(mask_conv, nomask_conv, train)

        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += nomask.shape[0]

    return correct / total


  # get_accuracy function for the second stage of training: fix FDM and train ANN
def get_accuracy_classifier(pretrained_cnn, fdm, classifier_model, data_loader, batch_size):
    correct = 0
    total = 0

    for i, data in enumerate(data_loader):
        # Get the inputs
        inputs, labels = data

        if inputs.shape[0] != batch_size:
            continue

        conv_features = pretrained_cnn(inputs)
        filtered_features = conv_features * fdm

        output = classifier_model(filtered_features)

        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += inputs.shape[0]

    return correct / total



# Train function 1: train FDM + ANN
def train(pretrained_cnn, fdm_classifier_model, fdm, train_pair_loader, valid_pair_loader, batch_size, num_epochs = 5, learning_rate=1e-3):

    torch.manual_seed(1) # set the random seed

    criterionFDM = nn.MSELoss() # mean square error loss to train FDM
    criterionClassifier = nn.CrossEntropyLoss() # CE loss for classification of multiple classes
    optimizer = torch.optim.Adam(fdm_classifier_model.parameters(), lr = learning_rate) # use Adam optimizer

    epochs, every_iteration, fdm_losses, classifier_losses, losses, train_acc, val_acc = [], [], [], [], [], [], []
    iter = 0
    for epoch in range(num_epochs):
        print("Epoch", epoch)
        for i, data in enumerate(train_pair_loader):
            iter += 1
            print("batch", i)
            # Get the inputs
            inputs, labels = data
            inputs_mask = inputs[0]
            inputs_nomask = inputs[1]

            # Discard the last batch to make size of fdm consistent
            if inputs_mask.shape[0] != batch_size:
                continue
            

            mask_conv = pretrained_cnn(inputs_mask)
            nomask_conv = pretrained_cnn(inputs_nomask)

            #############################################
            #To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
                mask_conv = mask_conv.cuda()
                nomask_conv = nomask_conv.cuda()
                labels = labels.cuda()
            #############################################

            fdm, mask_filtered, nomask_filtered, out = fdm_classifier_model(mask_conv, nomask_conv, False)

            fdm_loss = criterionFDM(mask_filtered, nomask_filtered) # The first loss is contrastive loss

            classifier_loss = criterionClassifier(out, labels)
            print("contrastive loss is", fdm_loss)
            print("classification loss is", classifier_loss)

            every_iteration.append(iter)
            fdm_losses.append(float(fdm_loss))
            classifier_losses.append(float(classifier_loss))

            total_loss = fdm_loss * 1000 + classifier_loss
            print("total loss is", total_loss)
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # if i%10 == 2: # check accuracy for every 10 iterations
            #   train_accuracy = get_accuracy(pretrained_cnn, fdm_classifier_model, True, train_pair_loader, batch_size)
            #   valid_accuracy = get_accuracy(pretrained_cnn, fdm_classifier_model, False, valid_pair_loader, batch_size)
            #   print("train accuracy is", train_accuracy)
            #   print("validation accuracy is", valid_accuracy)

            #   # record every 10 iterations
            #   losses.append(float(total_loss))     
            #   iteration.append(iter)
            #   # train_accuracy = get_accuracy(pretrained_cnn, fdm_classifier_model, True, train_pair_loader, batch_size)
            #   # valid_accuracy = get_accuracy(pretrained_cnn, fdm_classifier_model, False, valid_pair_loader, batch_size)
            #   # print("train accuracy is", train_accuracy)
            #   # print("validation accuracy is", valid_accuracy)
            #   train_acc.append(train_accuracy)
            #   val_acc.append(valid_accuracy)

        # get accuracy for every epoch
        train_accuracy = get_accuracy(pretrained_cnn, fdm_classifier_model, False, train_pair_loader, batch_size)
        valid_accuracy = get_accuracy(pretrained_cnn, fdm_classifier_model, False, valid_pair_loader, batch_size)
        print("train accuracy is", train_accuracy)
        print("validation accuracy is", valid_accuracy)
        losses.append(float(total_loss))
        epochs.append(epoch)
        ## Add code here

    # plotting contrastive loss curve
    plt.title("Contrastive Loss Curve")
    plt.plot(every_iteration, fdm_losses, label="Contrastive Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    #plt.show()
    plt.savefig("contrastiveLoss.jpg")

    # plotting classification loss curve
    plt.title("Classification Loss Curve")
    plt.plot(every_iteration, classifier_losses, label="Classification Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    #plt.show()
    plt.savefig("classificationLoss.jpg")

    # plotting overall loss curve
    plt.title("Loss Curve")
    plt.plot(epochs, losses, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    #plt.show()
    plt.savefig("overallLoss.jpg")

    # plotting accuracy curve
    plt.title("Accuracy Curve")
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, val_acc, label="Validation")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    #plt.show()
    plt.savefig("accuracy.jpg")



# Train function 2: fix FDM, train ANN only
def trainANN(pretrained_cnn, classifier_model, fdm, train_loader, valid_loader, batch_size, num_epochs = 5, learning_rate=1e-3):

    torch.manual_seed(1) # set the random seed

    criterion = nn.CrossEntropyLoss() # CE loss for classification of multiple classes
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr = learning_rate) # use Adam optimizer

    epochs, losses, train_acc, val_acc = [], [], [], []

    for epoch in range(num_epochs):
        print("Epoch", epoch)
        for i, data in enumerate(train_loader):
            print("batch", i)
            # Get the inputs
            inputs, labels = data
            

            # Discard the last batch to make size of fdm consistent
            if inputs.shape[0] != batch_size:
                continue

            # Use unmasked faces to train
            nomask_conv = pretrained_cnn(inputs)
            # Use fixed FDM to discard some features
            nomask_filtered = nomask_conv * fdm

            #############################################
            #To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
                nomask_filtered = nomask_filtered.cuda()
                labels = labels.cuda()
            #############################################

            out = classifier_model(nomask_filtered)
            print("1")
            loss = criterion(out, labels) # Classification loss
            print("2")
            print("loss is", total_loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i%10 == 2: # check accuracy for every 10 iterations
                train_accuracy = get_accuracy_classifier(pretrained_cnn, fdm, classifier_model, train_loader, batch_size)
                valid_accuracy = get_accuracy_classifier(pretrained_cnn, fdm, classifier_model, valid_loader, batch_size)
                print("train accuracy is", train_accuracy)
                print("validation accuracy is", valid_accuracy)

    # record for every epoch
    losses.append(float(loss))     
    epochs.append(epoch)
    train_accuracy = get_accuracy_classifier(pretrained_cnn, fdm, classifier_model, train_loader, batch_size)
    valid_accuracy = get_accuracy_classifier(pretrained_cnn, fdm, classifier_model, valid_loader, batch_size)
    print("train accuracy is", train_accuracy)
    print("validation accuracy is", valid_accuracy)
    train_acc.append(train_accuracy)
    val_acc.append(valid_accuracy)

    # plotting loss curve
    plt.title("Loss Curve")
    plt.plot(losses, label="Train")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

    # plotting accuracy curve
    plt.title("Accuracy Curve")
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, val_acc, label="Validation")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()




if __name__ == "__main__":

        
    resnet50 = models.resnet50(pretrained=True)
    modules = list(resnet50.children())[:-1]
    resnet50 = nn.Sequential(*modules)
    # tell the model not to learn or modify the weights / parameters of the model
    for p in resnet50.parameters():
        p.requires_grad = False


    maskRootAddr = 'self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset'
    nomaskRootAddr = 'self-built-masked-face-recognition-dataset/AFDB_face_dataset'

    new_train_pair_loader, new_valid_pair_loader, new_test_pair_loader = getNewOneToOnePairLoader(maskRootAddr, nomaskRootAddr, 128, 20)

    # One click to train

    # Need to change the first argument when changing batch size
    fdm = torch.empty(128, 2048, 1, 1)

    fdm_classifier_model = FDMClassifier()

    #Use GPU
    use_cuda = True

    if use_cuda and torch.cuda.is_available():
        fdm_classifier_model.cuda()
        print('CUDA is available!  Training on GPU ...')
    else:
        print('CUDA is not available.  Training on CPU ...')
    
    #proper model
    train(resnet50, fdm_classifier_model, fdm, new_train_pair_loader, new_valid_pair_loader, 128, 5, 0.0005)