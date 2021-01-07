'''Running the Model'''

###################
## Prerequisites ##
###################
import time
import pickle
import random
import csv
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from materials import CheXpertDataSet, CheXpertTrainer, DenseNet121

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



######################
## Arguments to Set ##
######################
parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p', '--policy', required = False, help = 'Define uncertain label policy.', default = 'ones')
parser.add_argument('-r', '--ratio', required = False, help = 'Training data ratio.', default = 1)
parser.add_argument('-o', '--output_path', required = False, help = 'Path to save results.', default = './results')
parser.add_argument('-s', '--random_seed', required = False, help = 'Random seed for reproduction.')
args = parser.parse_args()

# Example running commands ('nohup' command for running background on server)
'''
python3 run_chexpert.py
python3 run_chexpert.py -p ones -r 0.001 -o ensemble/experiment_01/ -s 1
nohup python3 run_chexpert.py -p ones -r 1 -o ensemble/experiment_01/ -s 1 > ensemble/experiment_01/result.txt &
'''

# Control randomness for reproduction
if args.random_seed:
    random_seed = int(args.random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



#######################
## Pre-define Values ##
#######################
'''Should have run 'run_preprocessing.py' before this part!'''
# Paths to the files with training, validation, and test sets.
pathFileTrain = './CheXpert-v1.0-small/train_mod.csv'
pathFileValid = './CheXpert-v1.0-small/valid_mod.csv'
pathFileTest = './CheXpert-v1.0-small/test_mod.csv'

# Neural network parameters
nnIsTrained = False # if pre-trained by ImageNet
nnClassCount = 14   # dimension of the output

# Training settings
trBatchSize = 16    # batch size
trMaxEpoch = 3      # maximum number of epochs

# Parameters related to image transforms: size of the down-scaled image, cropped image
imgtransResize = (320, 320)
imgtransCrop = 224

# Class names
class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']



######################
## Create a Dataset ##
######################
# Tranform data
transformList = []
transformList.append(transforms.Resize((imgtransCrop, imgtransCrop))) # 224
transformList.append(transforms.ToTensor())
transformSequence = transforms.Compose(transformList)

# Uncertain labeling policy
policy = args.policy # ones or zeroes

# Create a dataset
'''See 'materials.py' to check the class 'CheXpertDataSet'.'''
datasetTrain = CheXpertDataSet(pathFileTrain, nnClassCount, transformSequence, policy = policy)
datasetValid = CheXpertDataSet(pathFileValid, nnClassCount, transformSequence)
datasetTest = CheXpertDataSet(pathFileTest, nnClassCount, transformSequence, policy = policy)

# Use subset of datasetTrain for training
train_ratio = float(args.ratio) # use subset of original training dataset
train_num = round(len(datasetTrain) * train_ratio)
datasetTrain, datasetLeft = random_split(datasetTrain, [train_num, len(datasetTrain) - train_num])
print('<<< Data Information >>>')
print('Train data length:', len(datasetTrain))
print('Valid data length:', len(datasetValid))
print('Test data length:', len(datasetTest))
print('')

# Create DataLoaders
dataLoaderTrain = DataLoader(dataset = datasetTrain, batch_size = trBatchSize, 
                             shuffle = True, num_workers = 2, pin_memory = True)
dataLoaderVal = DataLoader(dataset = datasetValid, batch_size = trBatchSize, 
                           shuffle = False, num_workers = 2, pin_memory = True)
dataLoaderTest = DataLoader(dataset = datasetTest, num_workers = 2, pin_memory = True)



#####################
## Train the Model ##
#####################
# Initialize and load the model
'''See 'materials.py' to check the class 'DenseNet121'.'''
model = DenseNet121(nnClassCount).cuda()
model = torch.nn.DataParallel(model).cuda()

# Train the model
train_valid_start = time.time()
PATH = args.output_path
os.makedirs(PATH)
'''See 'materials.py' to check the class 'CheXpertTrainer'.'''
model_num, train_time = CheXpertTrainer.train(model, dataLoaderTrain, dataLoaderVal, nnClassCount, trMaxEpoch, checkpoint = None, PATH)
train_valid_end = time.time()
print('')
print('<<< Model Trained >>>')
print('m-epoch_ALL{0}.pth.tar'.format(model_num), 'is the best model.')
print('')



##############################
## Test and Draw ROC Curves ##
##############################
checkpoint = PATH + 'm-epoch_ALL{0}.pth.tar'.format(model_num)
'''See 'materials.py' to check the class 'CheXpertTrainer'.'''
outGT, outPRED, outPROB = CheXpertTrainer.test(model, dataLoaderTest, nnClassCount, checkpoint, class_names)

# Save the test outPROB
with open(PATH + 'testPROB.txt', 'wb') as fp:
    pickle.dump(outPROB, fp)

# Draw ROC curves
fig_size = plt.rcParams['figure.figsize']
plt.rcParams['figure.figsize'] = (30, 10)

for i in range(nnClassCount):
    fpr, tpr, threshold = metrics.roc_curve(outGT.cpu()[:,i], outPRED.cpu()[:,i])
    roc_auc = metrics.auc(fpr, tpr)
    f = plt.subplot(2, 7, i+1)

    plt.title('ROC for: ' + class_names[i])
    plt.plot(fpr, tpr, label = 'U-%s: AUC = %0.2f' % (policy, roc_auc))

    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

plt.savefig(PATH + 'ROC.png', dpi = 1000)



#########################
## Computational Stats ##
#########################
print('')
print('<<< Computational Stats >>>')
print(train_time.round(0), '/seconds per epoch.')
print('Total', round((train_valid_end - train_valid_start) / 60), 'minutes elapsed.')