# Prerequisites
import time
import pickle
import random
import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score

use_gpu = torch.cuda.is_available()

'''
python3 run_chexpert.py -p ones -r 0.001 -o ensemble/experiment_01/ -s 1
nohup python3 run_chexpert.py -p ones -r 1 -o ensemble/experiment_01/ -s 1 > ensemble/experiment_01/result.txt &
'''

parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p', '--policy', required = True, help = 'Define uncertain label policy.')
parser.add_argument('-r', '--ratio', required = True, help = 'Training data ratio.')
parser.add_argument('-o', '--output_path', required = True, help = 'Output model, ROC curve path.')
parser.add_argument('-s', '--random_seed', required = True, help = 'Set random seed.')
args = parser.parse_args()

# Control Randomness
random_seed = int(args.random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Pre-define Values
# Paths to the files with training, and validation sets.
# Each file contains pairs (path to image, output vector)
Traindata = pd.read_csv('./CheXpert-v1.0-small/train.csv')
Traindata = Traindata[Traindata['Path'].str.contains("frontal")] # use only frontal images
Traindata = Traindata[500:]
Traindata.to_csv('./CheXpert-v1.0-small/train_mod.csv', index = False)
# print("Train data length:", len(Traindata))

Validdata = pd.read_csv('./CheXpert-v1.0-small/valid.csv')
Validdata = Validdata[Validdata['Path'].str.contains("frontal")] # use only frontal images
Validdata.to_csv('./CheXpert-v1.0-small/valid_mod.csv', index = False)
# print("Valid data length:", len(Validdata))

Testdata = Traindata.head(500) # use first 500 training data as test data (obs ratio is almost same!)
Testdata.to_csv('./CheXpert-v1.0-small/test_mod.csv', index = False)
# print("Test data length:", len(Testdata))

pathFileTrain = './CheXpert-v1.0-small/train_mod.csv'
pathFileValid = './CheXpert-v1.0-small/valid_mod.csv'
pathFileTest = './CheXpert-v1.0-small/test_mod.csv'

# Neural network parameters:
nnIsTrained = False     # if pre-trained by ImageNet
nnClassCount = 14       # dimension of the output

# Training settings: batch size, maximum number of epochs
trBatchSize = 16
trMaxEpoch = 3

# Parameters related to image transforms: size of the down-scaled image, cropped image
imgtransResize = (320, 320)
imgtransCrop = 224

# Class names
class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']



# Create a Dataset
class CheXpertDataSet(Dataset):
    def __init__(self, data_PATH, transform = None, policy = 'ones'):
        """
        data_PATH: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels.
        """
        image_names = []
        labels = []

        with open(data_PATH, "r") as f:
            csvReader = csv.reader(f)
            next(csvReader, None) # skip the header
            for line in csvReader:
                image_name = line[0]
                label = line[5:]
                
                for i in range(14):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1:
                            if policy == "ones":
                                label[i] = 1
                            elif policy == "zeroes":
                                label[i] = 0
                            else:
                                label[i] = 0
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0
                        
                image_names.append('./' + image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)



# Create DataLoaders
# Tranform data
# IMAGENET_MEAN = [0.485, 0.456, 0.406]  # mean of ImageNet dataset(for normalization)
# IMAGENET_STD = [0.229, 0.224, 0.225]   # std of ImageNet dataset(for normalization)
# normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

transformList = []
transformList.append(transforms.Resize((imgtransCrop, imgtransCrop))) # 224
# transformList.append(transforms.RandomResizedCrop(imgtransCrop))
# transformList.append(transforms.RandomHorizontalFlip())
transformList.append(transforms.ToTensor())
# transformList.append(normalize)
transformSequence = transforms.Compose(transformList)

# Load dataset
policy = args.policy # ones or zeroes

datasetTrain = CheXpertDataSet(pathFileTrain, transformSequence, policy = policy)
datasetValid = CheXpertDataSet(pathFileValid, transformSequence)
datasetTest = CheXpertDataSet(pathFileTest, transformSequence, policy = policy)

# Use subset of datasetTrain
train_ratio = float(args.ratio) # use subset of original training dataset
train_num = round(len(datasetTrain) * train_ratio)
datasetTrain, datasetLeft = random_split(datasetTrain, [train_num, len(datasetTrain) - train_num])
print("<<< Data Information >>>")
print("Train data length:", len(datasetTrain))
print("Valid data length:", len(datasetValid))
print("Test data length:", len(datasetTest))
print('')

# Define DataLoaders
dataLoaderTrain = DataLoader(dataset = datasetTrain, batch_size = trBatchSize, 
                             shuffle = True, num_workers = 2, pin_memory = True)
dataLoaderVal = DataLoader(dataset = datasetValid, batch_size = trBatchSize, 
                           shuffle = False, num_workers = 2, pin_memory = True)
dataLoaderTest = DataLoader(dataset = datasetTest, num_workers = 2, pin_memory = True)



# Create Train and Test Models
PATH = args.output_path
class CheXpertTrainer():

    def train(model, dataLoaderTrain, dataLoaderVal, nnClassCount, trMaxEpoch, checkpoint):
        optimizer = optim.Adam(model.parameters(), lr = 0.0001, # setting optimizer & scheduler
                               betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0) 
        loss = torch.nn.BCELoss() # setting loss function
        
        if checkpoint != None and use_gpu: # loading checkpoint
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
            
        # Train the network
        lossMIN = 100000
        train_start = []
        train_end = []
        print('<<< Training & Evaluating >>>')
        for epochID in range(0, trMaxEpoch):
            train_start.append(time.time()) # training starts
            losst = CheXpertTrainer.epochTrain(model, dataLoaderTrain, optimizer, trMaxEpoch, nnClassCount, loss)
            train_end.append(time.time()) # training ends
            lossv = CheXpertTrainer.epochVal(model, dataLoaderVal, optimizer, trMaxEpoch, nnClassCount, loss)
            print("Training loss: {:.3f},".format(losst), "Valid loss: {:.3f}".format(lossv))
            
            if lossv < lossMIN:
                lossMIN = lossv
                model_num = epochID + 1
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 
                            'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 
                            PATH + 'm-epoch_ALL' + str(epochID + 1) + '.pth.tar')
                print('Epoch ' + str(epochID + 1) + ' [save] loss = ' + str(lossv))
            else:
                print('Epoch ' + str(epochID + 1) + ' [----] loss = ' + str(lossv))
        
        train_time = np.array(train_end) - np.array(train_start)
        return model_num, train_time
       
        
    def epochTrain(model, dataLoaderTrain, optimizer, epochMax, classCount, loss):
        model.train()
        losstrain = 0

        for batchID, (varInput, target) in enumerate(dataLoaderTrain):
            optimizer.zero_grad()
            
            varTarget = target.cuda(non_blocking = True)
            varOutput = model(varInput)
            lossvalue = loss(varOutput, varTarget)
                       
            lossvalue.backward()
            optimizer.step()
            
            losstrain += lossvalue.item()
            if batchID % 1000 == 999:
                print('[Batch: %5d] loss: %.3f'%(batchID + 1, losstrain / 2000))
            
        return losstrain / len(dataLoaderTrain)
    
    
    def epochVal(model, dataLoaderVal, optimizer, epochMax, classCount, loss):
        model.eval()
        lossVal = 0

        with torch.no_grad():
            for i, (varInput, target) in enumerate(dataLoaderVal):
                
                target = target.cuda(non_blocking = True)
                varOutput = model(varInput)
                
                lossVal += loss(varOutput, target)
                
        return lossVal / len(dataLoaderVal)

    
    def computeAUROC(dataGT, dataPRED, classCount):
        # Computes area under ROC curve 
        # dataGT: ground truth data
        # dataPRED: predicted data
        outAUROC = []
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            try:
                outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            except ValueError:
                pass
        return outAUROC
    
    
    def test(model, dataLoaderTest, nnClassCount, checkpoint, class_names):
        cudnn.benchmark = True
        
        if checkpoint != None and use_gpu:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])

        if use_gpu:
            outGT = torch.FloatTensor().cuda()
            outPRED = torch.FloatTensor().cuda()
        else:
            outGT = torch.FloatTensor()
            outPRED = torch.FloatTensor()
       
        model.eval()
        
        outPROB = []
        with torch.no_grad():
            for i, (input, target) in enumerate(dataLoaderTest):

                target = target.cuda()
                outGT = torch.cat((outGT, target), 0).cuda()
                outProb = model(input) # probability
                outProb = outProb.tolist()
                outPROB.append(outProb)

                bs, c, h, w = input.size()
                varInput = input.view(-1, c, h, w)
                
                out = model(varInput)
                outPRED = torch.cat((outPRED, out), 0)
        aurocIndividual = CheXpertTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        print('<<< Model Test Results >>>')
        print('AUROC mean ', aurocMean)
        
        for i in range (0, len(aurocIndividual)):
            print(class_names[i], ' ', aurocIndividual[i])
        
        return outGT, outPRED, outPROB



# Define and Train the Model
class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained = False)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x
    
# Initialize and load the model
model = DenseNet121(nnClassCount).cuda()
model = torch.nn.DataParallel(model).cuda()

train_valid_start = time.time()
model_num, train_time = CheXpertTrainer.train(model, dataLoaderTrain, dataLoaderVal, nnClassCount, trMaxEpoch, checkpoint = None)
train_valid_end = time.time()
print('')
print("<<< Model Trained >>>")
print("m-epoch_ALL{0}.pth.tar".format(model_num), 'is the best model.')
print('')



# Test and ROC Curves
checkpoint = PATH + "m-epoch_ALL{0}.pth.tar".format(model_num)
outGT, outPRED, outPROB = CheXpertTrainer.test(model, dataLoaderTest, nnClassCount, checkpoint, class_names)

with open(PATH + "testPROB.txt", "wb") as fp: # save test outPROB
    pickle.dump(outPROB, fp)

for i in range(nnClassCount):
    fpr, tpr, threshold = metrics.roc_curve(outGT.cpu()[:,i], outPRED.cpu()[:,i])
    roc_auc = metrics.auc(fpr, tpr)
    f = plt.subplot(2, 7, i+1)

    plt.title('ROC for: ' + class_names[i])
    plt.plot(fpr, tpr, label = 'U-ones: AUC = %0.2f' % roc_auc)

    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

fig_size = plt.rcParams["figure.figsize"]
plt.rcParams["figure.figsize"] = (30, 10)

plt.savefig(PATH + "ROC.png", dpi = 1000)
# plt.show()



# Computational Stats
# Single GPU(TITAN RTX GPU) usage was about 50% while training
print('')
print('<<< Computational Stats >>>')
print(train_time.round(0), '/seconds per epoch.') # took about 3.5 min for one epoch training (35 min for whole dataset)
print('Total', round((train_valid_end - train_valid_start) / 60), 'minutes elapsed.') # took about 10 min for whole training (100 min for whole training)