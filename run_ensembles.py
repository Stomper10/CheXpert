'''Run ensembles'''

###################
## Prerequisites ##
###################
import pickle
import random
import csv
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats 
from collections import Counter
from materials import CheXpertDataSet, CheXpertTrainer, DenseNet121, EnsemAgg

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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

use_gpu = torch.cuda.is_available()



######################
## Arguments to Set ##
######################
parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--policy', '-p', help = 'Define uncertain label policy: "ones" or "zeroes".', default = 'ones')
parser.add_argument('--output_path', '-o', help = 'Path to save results.', default = 'results/ensem_results/')
args = parser.parse_args()

# Example running commands ('nohup' command for running background on server)
'''
python3 run_ensembles.py
python3 run_ensembles.py -p ones -o results/ensem_results/
'''



#######################
## Pre-define Values ##
#######################
# Paths to the files with training, validation, and test sets.
pathFileTest = './CheXpert-v1.0-small/test_mod.csv'

# Neural network parameters
nnIsTrained = False # if pre-trained by ImageNet
nnClassCount = 14   # dimension of the output

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
datasetTest = CheXpertDataSet(pathFileTest, nnClassCount, transformSequence, policy = policy)

# Create DataLoaders
dataLoaderTest = DataLoader(dataset = datasetTest, num_workers = 2, pin_memory = True)



######################
## Define the Model ##
######################
# Initialize and load the model
'''See 'materials.py' to check the class 'DenseNet121'.'''
model = DenseNet121(nnClassCount).cuda()
model = torch.nn.DataParallel(model).cuda()



#############################
## Test Result Aggregation ##
#############################
# Load each experiment's test probability
ENSEM_DIR = 'ensembles/'
experiment_dirs = [f for f in os.listdir(ENSEM_DIR) if not f.startswith('.') and os.path.isdir(os.path.join(ENSEM_DIR, f))]

results = []
for i in range(len(experiment_dirs)):
    exPATH = './ensembles/{}/'.format(experiment_dirs[i])
    with open(exPATH + 'testPROB.txt', 'rb') as fp:
        result = pickle.load(fp)
        results.append(result)

# Average test probabilities
images_mean = []
for i in range(len(datasetTest)):
    image = []
    for j in range(nnClassCount):
        obs = 0
        for k in range(len(experiment_dirs)):
            obs += results[k][i][0][j]
        obs_mean = obs / len(experiment_dirs)
        image.append(obs_mean)
    images_mean.append(image)



##############################
## Test and Draw ROC Curves ##
##############################
# Apply EnsemAgg function for aggregation
EnsemTest = images_mean
'''See 'materials.py' to check the function 'EnsemAgg'.'''
outGT, outPRED, aurocMean, aurocIndividual = EnsemAgg(EnsemTest, dataLoaderTest, nnClassCount, class_names)

# Draw ROC curves
fig_size = plt.rcParams["figure.figsize"]
plt.rcParams["figure.figsize"] = (30, 10)

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

# Save ensemble results
PATH = args.output_path
if args.output_path[-1] != '/':
    PATH = args.output_path + '/'
else:
    PATH = args.output_path
    
if not os.path.exists(PATH): os.makedirs(PATH)
plt.savefig(PATH + 'ROC_ensem_mean.png', dpi = 1000)



###############################
## Save some printed outputs ##
###############################
with open(PATH + 'printed_outputs.txt', "w") as file:
    file.write('<<< Ensembles Test Results >>> \n')
    file.write('AUROC mean = {} \n'.format(aurocMean))
    for i in range (0, len(aurocIndividual)):
        file.write('{0} = {1} \n'.format(class_names[i], aurocIndividual[i]))