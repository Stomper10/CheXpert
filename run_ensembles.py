'''Run ensembles'''

###################
## Prerequisites ##
###################
import json
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
from easydict import EasyDict as edict
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
parser.add_argument('cfg_path', metavar='CFG_PATH', type = str, help = 'Path to the config file in yaml format.')
parser.add_argument('--output_path', '-o', help = 'Path to save results.', default = 'results/ensem_results/')
args = parser.parse_args()
with open(args.cfg_path) as f:
    cfg = edict(json.load(f))

# Example running commands ('nohup' command for running background on server)
'''
python3 run_ensembles.py configuration.json
python3 run_ensembles.py configuration.json -o results/ensem_results/
'''



#######################
## Pre-define Values ##
#######################
# Paths to the files with test set.
if cfg.image_type == 'small':
    img_type = '-small'
else:
    img_type = ''
pathFileTest = './CheXpert-v1.0{0}/test_200.csv'.format(img_type)

# Neural network parameters
nnIsTrained = cfg.pre_trained # if pre-trained by ImageNet

# Parameters related to image transforms: size of the down-scaled image, cropped image
imgtransResize = cfg.imgtransResize

# Class names
nnClassCount = cfg.nnClassCount   # dimension of the output - 5: only competition obs.
if nnClassCount == 5:
    class_names = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
else:
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

# Create a dataset
'''See 'materials.py' to check the class 'CheXpertDataSet'.'''
datasetTest = CheXpertDataSet(pathFileTest, nnClassCount, transformSequence)

# Create DataLoaders
dataLoaderTest = DataLoader(dataset = datasetTest, num_workers = 2, pin_memory = True)



######################
## Define the Model ##
######################
# Initialize and load the model
'''See 'materials.py' to check the class 'DenseNet121'.'''
model = DenseNet121(nnClassCount, nnIsTrained).cuda()
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
    with open(exPATH + 'testPROB_all.txt', 'rb') as fp:
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
if nnClassCount <= 7:
    nrows = 1
    ncols = nnClassCount
else:
    nrows = 2
    ncols = 7

fig, ax = plt.subplots(nrows = nrows, ncols = ncols)
fig.set_size_inches((ncols * 10, 10))
for i in range(nnClassCount):
    fpr, tpr, threshold = metrics.roc_curve(outGT.cpu()[:, i], outPRED.cpu()[:, i])
    roc_auc = metrics.auc(fpr, tpr)
    
    ax[i].plot(fpr, tpr, label = 'AUC = %0.2f' % (roc_auc))
    ax[i].set_title('ROC for: ' + class_names[i])
    ax[i].legend(loc = 'lower right')
    ax[i].plot([0, 1], [0, 1],'r--')
    ax[i].set_xlim([0, 1])
    ax[i].set_ylim([0, 1])
    ax[i].set_ylabel('True Positive Rate')
    ax[i].set_xlabel('False Positive Rate')

# Save ensemble results
PATH = args.output_path
if args.output_path[-1] != '/':
    PATH = args.output_path + '/'
else:
    PATH = args.output_path
    
if not os.path.exists(PATH): os.makedirs(PATH)
plt.savefig(PATH + 'ROC_ensem_mean.png', dpi = 100)



###############################
## Save some printed outputs ##
###############################
with open(PATH + 'printed_outputs.txt', "w") as file:
    file.write('<<< Ensembles Test Results >>> \n')
    file.write('AUROC mean = {} \n'.format(aurocMean))
    for i in range (0, len(aurocIndividual)):
        file.write('{0} = {1} \n'.format(class_names[i], aurocIndividual[i]))