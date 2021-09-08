'''Running the Model'''

###################
## Prerequisites ##
###################
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3' # should do this before importing torch modules!
import time
import json
import pickle
import random
import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
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

use_gpu = torch.cuda.is_available()



######################
## Arguments to Set ##
######################
parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('cfg_path', metavar = 'CFG_PATH', type = str, help = 'Path to the config file in yaml format.')
parser.add_argument('--output_path', '-o', help = 'Path to save results.', default = 'results/')
parser.add_argument('--random_seed', '-s', type = int, help = 'Random seed for reproduction.')
args = parser.parse_args()
with open(args.cfg_path) as f:
    cfg = edict(json.load(f))

# Example running commands ('nohup' command for running background on server)
'''
python3 run_chexpert.py configuration.json 
python3 run_chexpert.py configuration.json -o results/ -s 2021
nohup python3 run_chexpert.py configuration.json -o ensembles/experiment_00/ -s 0 > ensemble/printed_00.txt &
'''

# Control randomness for reproduction
if args.random_seed != None:
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)



#######################
## Pre-define Values ##
#######################
'''Should have run 'run_preprocessing.py' before this part!'''
# Paths to the files with training, validation, and test sets.
if cfg.image_type == 'small':
    img_type = '-small'
else:
    img_type = ''

Traindata_frt = pd.read_csv('./CheXpert-v1.0{0}/train_frt.csv'.format(img_type)) ###
Traindata_frt = Traindata_frt.sort_values('Path').reset_index(drop=True)
Traindata_frt.to_csv('./CheXpert-v1.0{0}/train_frt.csv'.format(img_type), index = False) ###
Traindata_lat = pd.read_csv('./CheXpert-v1.0{0}/train_lat.csv'.format(img_type)) ###
Traindata_lat = Traindata_lat.sort_values('Path').reset_index(drop=True)
Traindata_lat.to_csv('./CheXpert-v1.0{0}/train_lat.csv'.format(img_type), index = False) ###

pathFileTrain_frt = './CheXpert-v1.0{0}/train_frt.csv'.format(img_type) ###
pathFileTrain_lat = './CheXpert-v1.0{0}/train_lat.csv'.format(img_type) ###
pathFileValid_frt = './CheXpert-v1.0{0}/valid_frt.csv'.format(img_type)
pathFileValid_lat = './CheXpert-v1.0{0}/valid_lat.csv'.format(img_type)
pathFileTest_frt = './CheXpert-v1.0{0}/test_frt.csv'.format(img_type)
pathFileTest_lat = './CheXpert-v1.0{0}/test_lat.csv'.format(img_type)
pathFileTest_agg = './CheXpert-v1.0{0}/test_agg.csv'.format(img_type)

# Neural network parameters
nnIsTrained = cfg.pre_trained # if pre-trained by ImageNet

# Training settings
trBatchSize = cfg.batch_size # batch size
trMaxEpoch = cfg.epochs      # maximum number of epochs

# Parameters related to image transforms: size of the down-scaled image, cropped image
imgtransResize = cfg.imgtransResize

# Class names
nnClassCount = cfg.nnClassCount   # dimension of the output - 5: only competition obs.
class_names = ["Card", "Edem", "Cons", "Atel", "PlEf"]



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
datasetTrain_frt = CheXpertDataSet(pathFileTrain_frt, nnClassCount, cfg.policy, transformSequence)
datasetTrain_lat = CheXpertDataSet(pathFileTrain_lat, nnClassCount, cfg.policy, transformSequence)
datasetValid_frt = CheXpertDataSet(pathFileValid_frt, nnClassCount, cfg.policy, transformSequence)
datasetValid_lat = CheXpertDataSet(pathFileValid_lat, nnClassCount, cfg.policy, transformSequence)
datasetTest_frt = CheXpertDataSet(pathFileTest_frt, nnClassCount, cfg.policy, transformSequence)
datasetTest_lat = CheXpertDataSet(pathFileTest_lat, nnClassCount, cfg.policy, transformSequence)
datasetTest_agg = CheXpertDataSet(pathFileTest_agg, nnClassCount, cfg.policy, transformSequence)

# Use subset of datasetTrain for training ###
train_num_frt = round(len(datasetTrain_frt) * cfg.train_ratio) # use subset of original training dataset
train_num_lat = round(len(datasetTrain_lat) * cfg.train_ratio) # use subset of original training dataset
datasetTrain_frt, _ = random_split(datasetTrain_frt, [train_num_frt, len(datasetTrain_frt) - train_num_frt])
datasetTrain_lat, _ = random_split(datasetTrain_lat, [train_num_lat, len(datasetTrain_lat) - train_num_lat])
print('<<< Data Information >>>')
print('Train data (frontal):', len(datasetTrain_frt))
print('Train data (lateral):', len(datasetTrain_lat))
print('Valid data (frontal):', len(datasetValid_frt))
print('Valid data (lateral):', len(datasetValid_lat))
print('Test data (frontal):', len(datasetTest_frt))
print('Test data (lateral):', len(datasetTest_lat))
print('Test data (study):', len(datasetTest_agg), '\n')

# Create DataLoaders
dataLoaderTrain_frt = DataLoader(dataset = datasetTrain_frt, batch_size = trBatchSize, 
                                 shuffle = True, num_workers = 2, pin_memory = True) ###
dataLoaderTrain_lat = DataLoader(dataset = datasetTrain_lat, batch_size = trBatchSize, 
                                 shuffle = True, num_workers = 2, pin_memory = True) ###
dataLoaderVal_frt = DataLoader(dataset = datasetValid_frt, batch_size = trBatchSize, 
                               shuffle = False, num_workers = 2, pin_memory = True)
dataLoaderVal_lat = DataLoader(dataset = datasetValid_lat, batch_size = trBatchSize, 
                               shuffle = False, num_workers = 2, pin_memory = True)
dataLoaderTest_frt = DataLoader(dataset = datasetTest_frt, num_workers = 2, pin_memory = True)
dataLoaderTest_lat = DataLoader(dataset = datasetTest_lat, num_workers = 2, pin_memory = True)
dataLoaderTest_agg = DataLoader(dataset = datasetTest_agg, num_workers = 2, pin_memory = True)



#####################
## Train the Model ##
#####################
# Initialize and load the model
'''See 'materials.py' to check the class 'DenseNet121'.'''
model = DenseNet121(nnClassCount, nnIsTrained).cuda()
model = torch.nn.DataParallel(model).cuda()

# Train the model
PATH = args.output_path
if args.output_path[-1] != '/':
    PATH = PATH + '/'
else:
    PATH = PATH

if not os.path.exists(PATH): os.makedirs(PATH)

# Train frontal model
train_valid_start_frt = time.time()
'''See 'materials.py' to check the class 'CheXpertTrainer'.'''
model_num_frt, model_num_frt_each, train_time_frt = CheXpertTrainer.train(model, dataLoaderTrain_frt, dataLoaderVal_frt, class_names,
                                                                          nnClassCount, trMaxEpoch, PATH, 'frt', checkpoint = None, cfg = cfg)
train_valid_end_frt = time.time()

# Train lateral model
train_valid_start_lat = time.time()
'''See 'materials.py' to check the class 'CheXpertTrainer'.'''
model_num_lat, model_num_lat_each, train_time_lat = CheXpertTrainer.train(model, dataLoaderTrain_lat, dataLoaderVal_lat, class_names,
                                                                          nnClassCount, trMaxEpoch, PATH, 'lat', checkpoint = None, cfg = cfg)
train_valid_end_lat = time.time()
print('<<< Model Trained >>>')
print('For frontal model,', 'm-epoch_{0}_frt.pth.tar'.format(model_num_frt), 'is the best overall.')
for i in range(5):
    print('For frontal {0},'.format(class_names[i]), 'm-epoch_{0}_frt.pth.tar'.format(model_num_frt_each[i]), 'is the best.')
print('')
print('For lateral model,', 'm-epoch_{0}_lat.pth.tar'.format(model_num_lat), 'is the best overall.')
for i in range(5):
    print('For lateral {0},'.format(class_names[i]), 'm-epoch_{0}_lat.pth.tar'.format(model_num_lat_each[i]), 'is the best.')
print('')



##############################
## Test and Draw ROC Curves ##
##############################
checkpoint_frt = PATH + 'm-epoch_{0}_frt.pth.tar'.format(trMaxEpoch) # Use the last model ('model_num_frt' if use valid set for model decision)
checkpoint_lat = PATH + 'm-epoch_{0}_lat.pth.tar'.format(trMaxEpoch) # Use the last model ('model_num_lat' if use valid set for model decision)
'''See 'materials.py' to check the class 'CheXpertTrainer'.'''
outGT_frt, outPRED_frt, outPROB_frt, aurocMean_frt, aurocIndividual_frt = CheXpertTrainer.test(model, dataLoaderTest_frt, nnClassCount, checkpoint_frt, class_names, 'frt')
outGT_lat, outPRED_lat, outPROB_lat, aurocMean_lat, aurocIndividual_lat = CheXpertTrainer.test(model, dataLoaderTest_lat, nnClassCount, checkpoint_lat, class_names, 'lat')

# Save the test outPROB_frt
with open('{}testPROB_frt.txt'.format(PATH), 'wb') as fp:
    pickle.dump(outPROB_frt, fp)

# Save the test outPROB_lat
with open('{}testPROB_lat.txt'.format(PATH), 'wb') as fp:
    pickle.dump(outPROB_lat, fp)

test_frt = pd.read_csv(pathFileTest_frt)
test_lat = pd.read_csv(pathFileTest_lat)

column_names = ['Path'] + class_names
df = pd.DataFrame(0, index = np.arange(len(test_frt) + len(test_lat)), columns = column_names)
test_frt_list = list(test_frt['Path'].copy())
test_lat_list = list(test_lat['Path'].copy())

for i in range(len(test_frt_list)):
    df.iloc[i, 0] = test_frt_list[i].split('/')[2] + '/' + test_frt_list[i].split('/')[3]

for i in range(len(test_lat_list)):
    df.iloc[len(test_frt_list) + i, 0] = test_lat_list[i].split('/')[2] + '/' + test_frt_list[i].split('/')[3]

for i in range(len(outPROB_frt)):
    for j in range(len(class_names)):
        df.iloc[i, j + 1] = outPROB_frt[i][0][j]
        
for i in range(len(outPROB_lat)):
    for j in range(len(class_names)):
        df.iloc[len(outPROB_frt) + i, j + 1] = outPROB_lat[i][0][j]

df_agg = df.groupby('Path').agg({'Card' : 'min',
                                 'Edem' : 'max',
                                 'Cons' : 'min',
                                 'Atel' : 'min',
                                 'PlEf' : 'min'}).reset_index()
df_agg = df_agg.sort_values('Path')
results = df_agg.drop(['Path'], axis = 1).values.tolist()

# Save the test outPROB_all
outPROB_all = []
for i in range(len(results)):
    outPROB_all.append([results[i]])

with open('{}testPROB_all.txt'.format(PATH), 'wb') as fp:
    pickle.dump(outPROB_all, fp)

# Draw ROC curves
EnsemTest = results
'''See 'materials.py' to check the function 'EnsemAgg'.'''
outGT, outPRED, aurocMean, aurocIndividual = EnsemAgg(EnsemTest, dataLoaderTest_agg, nnClassCount, class_names)

fig, ax = plt.subplots(nrows = 1, ncols = nnClassCount)
ax = ax.flatten()
fig.set_size_inches((nnClassCount * 10, 10))
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

plt.savefig('{0}ROC_{1}.png'.format(PATH, nnClassCount), dpi = 100)
plt.close()



#########################
## Computational Stats ##
#########################
print('<<< Computational Stats >>>')
print(train_time_frt.round(0), '/seconds per epoch. (frt)')
print('Total', round((train_valid_end_frt - train_valid_start_frt) / 60), 'minutes elapsed.')
print(train_time_lat.round(0), '/seconds per epoch. (lat)')
print('Total', round((train_valid_end_lat - train_valid_start_lat) / 60), 'minutes elapsed.')