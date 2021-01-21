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
parser.add_argument('--policy', '-p', help = 'Define uncertain label policy: "ones" or "zeroes".', default = 'ones')
parser.add_argument('--ratio', '-r', type = float, help = 'Training data ratio: 0 < val <= 1.', default = 1)
parser.add_argument('--output_path', '-o', help = 'Path to save results.', default = 'results/')
parser.add_argument('--random_seed', '-s', type = int, help = 'Random seed for reproduction.')
parser.add_argument('--epochs', '-e', type = int, help = 'The number of training epochs.', default = 3)
parser.add_argument('--batch_size', '-b', type = int, help = 'The number of batch size.', default = 16)
args = parser.parse_args()

# Example running commands ('nohup' command for running background on server)
'''
python3 run_chexpert.py
python3 run_chexpert.py -p ones -r 0.01 -o results/ -s 2021
nohup python3 run_chexpert.py -p ones -r 1 -o ensemble/experiment_00/ -s 0 > ensemble/printed_00.txt &
'''

# Control randomness for reproduction
if args.random_seed:
    random_seed = args.random_seed
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
pathFileTrain_frt = './CheXpert-v1.0-small/train_frt.csv'
pathFileTrain_lat = './CheXpert-v1.0-small/train_lat.csv'
pathFileValid_frt = './CheXpert-v1.0-small/valid_frt.csv'
pathFileValid_lat = './CheXpert-v1.0-small/valid_lat.csv'
pathFileTest_frt = './CheXpert-v1.0-small/test_frt.csv'
pathFileTest_lat = './CheXpert-v1.0-small/test_lat.csv'

# Neural network parameters
nnIsTrained = False # if pre-trained by ImageNet
nnClassCount = 14   # dimension of the output

# Training settings
trBatchSize = args.batch_size    # batch size
trMaxEpoch = args.epochs      # maximum number of epochs

# Parameters related to image transforms: size of the down-scaled image, cropped image
imgtransResize = 320

# Class names
class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']



######################
## Create a Dataset ##
######################
# Tranform data
transformList = []
transformList.append(transforms.Resize((imgtransResize, imgtransResize))) # 320
transformList.append(transforms.ToTensor())
transformSequence = transforms.Compose(transformList)

# Uncertain labeling policy
policy = args.policy # ones or zeroes

# Create a dataset
'''See 'materials.py' to check the class 'CheXpertDataSet'.'''
datasetTrain_frt = CheXpertDataSet(pathFileTrain_frt, nnClassCount, transformSequence, policy = policy)
datasetTrain_lat = CheXpertDataSet(pathFileTrain_lat, nnClassCount, transformSequence, policy = policy)
datasetValid_frt = CheXpertDataSet(pathFileValid_frt, nnClassCount, transformSequence)
datasetValid_lat = CheXpertDataSet(pathFileValid_lat, nnClassCount, transformSequence)
datasetTest_frt = CheXpertDataSet(pathFileTest_frt, nnClassCount, transformSequence, policy = policy)
datasetTest_lat = CheXpertDataSet(pathFileTest_lat, nnClassCount, transformSequence, policy = policy)

# Use subset of datasetTrain for training
train_num_frt = round(len(datasetTrain_frt) * args.ratio) # use subset of original training dataset
train_num_lat = round(len(datasetTrain_lat) * args.ratio) # use subset of original training dataset
datasetTrain_frt, _ = random_split(datasetTrain_frt, [train_num_frt, len(datasetTrain_frt) - train_num_frt])
datasetTrain_lat, _ = random_split(datasetTrain_lat, [train_num_lat, len(datasetTrain_lat) - train_num_lat])
print('<<< Data Information >>>')
print('Train data length(frontal):', len(datasetTrain_frt))
print('Train data length(lateral):', len(datasetTrain_lat))
print('Valid data length(frontal):', len(datasetValid_frt))
print('Valid data length(lateral):', len(datasetValid_lat))
print('Test data length(frontal):', len(datasetTest_frt))
print('Test data length(lateral):', len(datasetTest_lat))
print('')

# Create DataLoaders
dataLoaderTrain_frt = DataLoader(dataset = datasetTrain_frt, batch_size = trBatchSize, 
                                 shuffle = True, num_workers = 2, pin_memory = True)
dataLoaderTrain_lat = DataLoader(dataset = datasetTrain_lat, batch_size = trBatchSize, 
                                 shuffle = True, num_workers = 2, pin_memory = True)
dataLoaderVal_frt = DataLoader(dataset = datasetValid_frt, batch_size = trBatchSize, 
                               shuffle = False, num_workers = 2, pin_memory = True)
dataLoaderVal_lat = DataLoader(dataset = datasetValid_lat, batch_size = trBatchSize, 
                               shuffle = False, num_workers = 2, pin_memory = True)
dataLoaderTest_frt = DataLoader(dataset = datasetTest_frt, num_workers = 2, pin_memory = True)
dataLoaderTest_lat = DataLoader(dataset = datasetTest_lat, num_workers = 2, pin_memory = True)



#####################
## Train the Model ##
#####################
# Initialize and load the model
'''See 'materials.py' to check the class 'DenseNet121'.'''
model = DenseNet121(nnClassCount).cuda()
model = torch.nn.DataParallel(model).cuda()

# Train the model
PATH = args.output_path
if args.output_path[-1] != '/':
    PATH = args.output_path + '/'
else:
    PATH = args.output_path

if not os.path.exists(PATH): os.makedirs(PATH)

# Train frontal model
train_valid_start_frt = time.time()
'''See 'materials.py' to check the class 'CheXpertTrainer'.'''
model_num_frt, train_time_frt = CheXpertTrainer.train(model, dataLoaderTrain_frt, dataLoaderVal_frt, nnClassCount, trMaxEpoch, PATH, checkpoint = None)
train_valid_end_frt = time.time()
print('')

# Train lateral model
train_valid_start_lat = time.time()
'''See 'materials.py' to check the class 'CheXpertTrainer'.'''
model_num_lat, train_time_lat = CheXpertTrainer.train(model, dataLoaderTrain_lat, dataLoaderVal_lat, nnClassCount, trMaxEpoch, PATH, checkpoint = None)
train_valid_end_lat = time.time()
print('')
print('<<< Model Trained >>>')
print('For frontal model,', 'm-epoch_{0}.pth.tar'.format(model_num_frt), 'is the best model.')
print('For lateral model,', 'm-epoch_{0}.pth.tar'.format(model_num_lat), 'is the best model.')
print('')



##############################
## Test and Draw ROC Curves ##
##############################
checkpoint_frt = PATH + 'm-epoch_{0}.pth.tar'.format(model_num_frt)
checkpoint_lat = PATH + 'm-epoch_{0}.pth.tar'.format(model_num_lat)
'''See 'materials.py' to check the class 'CheXpertTrainer'.'''
outGT_frt, outPRED_frt, outPROB_frt, aurocMean_frt, aurocIndividual_frt = CheXpertTrainer.test(model, dataLoaderTest_frt, nnClassCount, checkpoint_frt, class_names)
print('')
outGT_lat, outPRED_lat, outPROB_lat, aurocMean_lat, aurocIndividual_lat = CheXpertTrainer.test(model, dataLoaderTest_lat, nnClassCount, checkpoint_lat, class_names)

# Save the test outPROB_frt
with open('{}testPROB_frt.txt'.format(PATH), 'wb') as fp:
    pickle.dump(outPROB_frt, fp)

# Save the test outPROB_lat
with open('{}testPROB_lat.txt'.format(PATH), 'wb') as fp:
    pickle.dump(outPROB_lat, fp)

# Draw ROC curves
fig_size = plt.rcParams['figure.figsize']
plt.rcParams['figure.figsize'] = (30, 10)

for i in range(nnClassCount):
    fpr, tpr, threshold = metrics.roc_curve(outGT_frt.cpu()[:,i], outPRED_frt.cpu()[:,i])
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

plt.savefig('{}ROC_frt.png'.format(PATH), dpi = 1000)


for i in range(nnClassCount):
    fpr, tpr, threshold = metrics.roc_curve(outGT_lat.cpu()[:,i], outPRED_lat.cpu()[:,i])
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

plt.savefig('{}ROC_lat.png'.format(PATH), dpi = 1000)



#########################
## Computational Stats ##
#########################
print('')
print('<<< Computational Stats >>>')
print(train_time_frt.round(0), '/seconds per epoch.')
print(train_time_lat.round(0), '/seconds per epoch.')
print('Total', round((train_valid_end_frt - train_valid_start_frt) / 60), 'minutes elapsed.')
print('Total', round((train_valid_end_lat - train_valid_start_lat) / 60), 'minutes elapsed.')



###############################
## Save some printed outputs ##
###############################
'''
with open(PATH + 'printed_outputs.txt', "w") as file:
    file.write('<<< Data Information >>> \n')
    file.write('Train data length: {} \n'.format(len(datasetTrain)))
    file.write('Valid data length: {} \n'.format(len(datasetValid)))
    file.write('Test data length: {} \n'.format(len(datasetTest)))
    file.write('\n')
    file.write('<<< Model Trained >>> \n')
    file.write('m-epoch_{0}.pth.tar is the best model. \n'.format(model_num))
    file.write('\n')
    file.write('<<< Model Test Results >>> \n')
    file.write('AUROC mean = {} \n'.format(aurocMean))
    for i in range (0, len(aurocIndividual)):
        file.write('{0} = {1} \n'.format(class_names[i], aurocIndividual[i]))
    file.write('\n')
    file.write('<<< Computational Stats >>> \n')
    file.write('{} /seconds per epoch. \n'.format(train_time.round(0)))
    file.write('Total {} minutes elapsed.'.format(round((train_valid_end - train_valid_start) / 60)))
'''