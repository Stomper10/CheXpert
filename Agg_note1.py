###################
## Prerequisites ##
###################
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3' # should do this before importing torch modules!
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
'''parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('cfg_path', metavar = 'CFG_PATH', type = str, help = 'Path to the config file in yaml format.')
parser.add_argument('--output_path', '-o', help = 'Path to save results.', default = 'results/')
parser.add_argument('--random_seed', '-s', type = int, help = 'Random seed for reproduction.')
args = parser.parse_args()
with open(args.cfg_path) as f:
    cfg = edict(json.load(f))'''

# Example running commands ('nohup' command for running background on server)
'''
python3 run_chexpert.py configuration.json 
python3 run_chexpert.py configuration.json -o results/ -s 2021
nohup python3 run_chexpert.py configuration.json -o ensembles/experiment_00/ -s 0 > ensemble/printed_00.txt &
'''

# Control randomness for reproduction
'''if args.random_seed != None:
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)'''



#######################
## Pre-define Values ##
#######################
'''Should have run 'run_preprocessing.py' before this part!'''
# Paths to the files with training, validation, and test sets.
'''if cfg.image_type == 'small':
    img_type = '-small'
else:
    img_type = ''
'''

img_type = '-small' ###
PATH = './results/' ###

with open('{}210729/testPROB_frt.txt'.format(PATH), 'rb') as f:
    outPROB_frt = pickle.load(f)

with open('{}210729/testPROB_lat.txt'.format(PATH), 'rb') as f:
    outPROB_lat = pickle.load(f)

test_frt = pd.read_csv('./CheXpert-v1.0{0}/test_frt.csv'.format(img_type))
test_lat = pd.read_csv('./CheXpert-v1.0{0}/test_lat.csv'.format(img_type))

test_frt1 = test_frt[['Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA', 
                      'Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']].copy()
test_lat1 = test_lat[['Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA', 
                      'Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']].copy()

test_frt1['Card'] = 0
test_frt1['Edem'] = 0
test_frt1['Cons'] = 0
test_frt1['Atel'] = 0
test_frt1['PlEf'] = 0

for i in range(len(outPROB_frt)):
    for j in range(5):
        test_frt1.iloc[i, j+10] = outPROB_frt[i][0][j]
test_frt1 = test_frt1.rename(columns={'Pleural Effusion': 'Pleural_Effusion'})

test_frt1['Card_diff'] = test_frt1.apply(lambda row: abs(row.Card - row.Cardiomegaly), axis = 1)
test_frt1['Edem_diff'] = test_frt1.apply(lambda row: abs(row.Edem - row.Edema), axis = 1)
test_frt1['Cons_diff'] = test_frt1.apply(lambda row: abs(row.Cons - row.Consolidation), axis = 1)
test_frt1['Atel_diff'] = test_frt1.apply(lambda row: abs(row.Atel - row.Atelectasis), axis = 1)
test_frt1['PlEf_diff'] = test_frt1.apply(lambda row: abs(row.PlEf - row.Pleural_Effusion), axis = 1)

test_frt1['Card_flag'] = test_frt1.apply(lambda row: row.Card_diff > 0.9, axis = 1) # or 0.5
test_frt1['Edem_flag'] = test_frt1.apply(lambda row: row.Edem_diff > 0.9, axis = 1) # or 0.5
test_frt1['Cons_flag'] = test_frt1.apply(lambda row: row.Cons_diff > 0.9, axis = 1) # or 0.5
test_frt1['Atel_flag'] = test_frt1.apply(lambda row: row.Atel_diff > 0.9, axis = 1) # or 0.5
test_frt1['PlEf_flag'] = test_frt1.apply(lambda row: row.PlEf_diff > 0.9, axis = 1) # or 0.5

test_frt1['Age'] = test_frt1.apply(lambda row: str(row.Age)[0], axis = 1) # convert to age range


test_lat1['Card'] = 0
test_lat1['Edem'] = 0
test_lat1['Cons'] = 0
test_lat1['Atel'] = 0
test_lat1['PlEf'] = 0

for i in range(len(outPROB_lat)):
    for j in range(5):
        test_lat1.iloc[i, j+10] = outPROB_lat[i][0][j]
test_lat1 = test_lat1.rename(columns={'Pleural Effusion': 'Pleural_Effusion'})

test_lat1['Card_diff'] = test_lat1.apply(lambda row: abs(row.Card - row.Cardiomegaly), axis = 1)
test_lat1['Edem_diff'] = test_lat1.apply(lambda row: abs(row.Edem - row.Edema), axis = 1)
test_lat1['Cons_diff'] = test_lat1.apply(lambda row: abs(row.Cons - row.Consolidation), axis = 1)
test_lat1['Atel_diff'] = test_lat1.apply(lambda row: abs(row.Atel - row.Atelectasis), axis = 1)
test_lat1['PlEf_diff'] = test_lat1.apply(lambda row: abs(row.PlEf - row.Pleural_Effusion), axis = 1)

test_lat1['Card_flag'] = test_lat1.apply(lambda row: row.Card_diff > 0.9, axis = 1) # or 0.5
test_lat1['Edem_flag'] = test_lat1.apply(lambda row: row.Edem_diff > 0.9, axis = 1) # or 0.5
test_lat1['Cons_flag'] = test_lat1.apply(lambda row: row.Cons_diff > 0.9, axis = 1) # or 0.5
test_lat1['Atel_flag'] = test_lat1.apply(lambda row: row.Atel_diff > 0.9, axis = 1) # or 0.5
test_lat1['PlEf_flag'] = test_lat1.apply(lambda row: row.PlEf_diff > 0.9, axis = 1) # or 0.5

test_lat1['Age'] = test_lat1.apply(lambda row: str(row.Age)[0], axis = 1) # convert to age range



######################
## Confusion Matrix ##
######################
print(test_frt1.Card_flag.value_counts())
print(test_frt1.Edem_flag.value_counts())
print(test_frt1.Cons_flag.value_counts())
print(test_frt1.Atel_flag.value_counts())
print(test_frt1.PlEf_flag.value_counts())

test_frt1[test_frt1.Card_flag == 1]
test_frt1[test_frt1.Edem_flag == 1]
test_frt1[test_frt1.Cons_flag == 1]

# Card (0.9, 0.5) with Sex -> not significant
test_frt1.Sex.value_counts().sort_index()
test_frt1[test_frt1.Card_flag == 1].Sex.value_counts().sort_index()

# Edem (0.9, 0.5) with Sex -> not significant
test_frt1.Sex.value_counts().sort_index()
test_frt1[test_frt1.Edem_flag == 1].Sex.value_counts().sort_index()

# Cons (0.9, 0.5) with Sex -> female is twice more than male / female is more than male
test_frt1.Sex.value_counts().sort_index()
test_frt1[test_frt1.Cons_flag == 1].Sex.value_counts().sort_index()

# Atel (0.5) with Sex -> not significant
test_frt1.Sex.value_counts().sort_index()
test_frt1[test_frt1.Atel_flag == 1].Sex.value_counts().sort_index()

# PlEf (0.5) with Sex -> not significant
test_frt1.Sex.value_counts().sort_index()
test_frt1[test_frt1.PlEf_flag == 1].Sex.value_counts().sort_index()



# Card (0.9, 0.5) with Age -> not significant
test_frt1.Age.value_counts().sort_index()
test_frt1[test_frt1.Card_flag == 1].Age.value_counts().sort_index()

# Edem (0.9, 0.5) with Age -> not significant
test_frt1.Age.value_counts().sort_index()
test_frt1[test_frt1.Edem_flag == 1].Age.value_counts().sort_index()

# Cons (0.9, 0.5) with Age -> most in 80s
test_frt1.Age.value_counts().sort_index()
test_frt1[test_frt1.Cons_flag == 1].Age.value_counts().sort_index()

# Atel (0.5) with Sex -> not significant
test_frt1.Sex.value_counts().sort_index()
test_frt1[test_frt1.Atel_flag == 1].Age.value_counts().sort_index()

# PlEf (0.5) with Sex -> not significant
test_frt1.Sex.value_counts().sort_index()
test_frt1[test_frt1.PlEf_flag == 1].Age.value_counts().sort_index()



# Card (0.9, 0.5) with AP/PA -> AP dominates / AP dominates
test_frt1['AP/PA'].value_counts().sort_index()
test_frt1[test_frt1.Card_flag == 1]['AP/PA'].value_counts().sort_index()

# Edem (0.9, 0.5) with AP/PA -> not significant / AP dominates
test_frt1['AP/PA'].value_counts().sort_index()
test_frt1[test_frt1.Edem_flag == 1]['AP/PA'].value_counts().sort_index()

# Cons (0.9, 0.5) with AP/PA -> AP dominates / AP dominates
test_frt1['AP/PA'].value_counts().sort_index()
test_frt1[test_frt1.Cons_flag == 1]['AP/PA'].value_counts().sort_index()

# Atel (0.5) with AP/PA -> AP dominates / AP dominates
test_frt1['AP/PA'].value_counts().sort_index()
test_frt1[test_frt1.Atel_flag == 1]['AP/PA'].value_counts().sort_index()

# Cons (0.5) with AP/PA -> AP dominates / AP dominates
test_frt1['AP/PA'].value_counts().sort_index()
test_frt1[test_frt1.PlEf_flag == 1]['AP/PA'].value_counts().sort_index()


# lat images
print(test_lat1.Card_flag.value_counts())
print(test_lat1.Edem_flag.value_counts())
print(test_lat1.Cons_flag.value_counts())
print(test_lat1.Atel_flag.value_counts())
print(test_lat1.PlEf_flag.value_counts())

test_lat1[test_lat1.Card_flag == 1]
test_lat1[test_lat1.Edem_flag == 1]
test_lat1[test_lat1.PlEf_flag == 1]


# Appendix
cm_Card_sex = pd.crosstab(test_frt1['Card_flag'], test_frt1['Sex'], rownames=['Card'], colnames=['Sex'])
print(cm_Card_sex)



###################################
## Train with only failure modes ##
###################################
# frt 0.7
data_frt = pd.read_csv('./CheXpert-v1.0{0}/data_frt_boost1.csv'.format(img_type))

data_frt['Card_diff'] = data_frt.apply(lambda row: abs(row.Card - row.Cardiomegaly), axis = 1)
data_frt['Edem_diff'] = data_frt.apply(lambda row: abs(row.Edem - row.Edema), axis = 1)
data_frt['Cons_diff'] = data_frt.apply(lambda row: abs(row.Cons - row.Consolidation), axis = 1)
data_frt['Atel_diff'] = data_frt.apply(lambda row: abs(row.Atel - row.Atelectasis), axis = 1)
data_frt['PlEf_diff'] = data_frt.apply(lambda row: abs(row.PlEf - row.Pleural_Effusion), axis = 1)

data_frt['Card_flag'] = data_frt.apply(lambda row: row.Card_diff > 0.7, axis = 1)
data_frt['Edem_flag'] = data_frt.apply(lambda row: row.Edem_diff > 0.7, axis = 1)
data_frt['Cons_flag'] = data_frt.apply(lambda row: row.Cons_diff > 0.7, axis = 1)
data_frt['Atel_flag'] = data_frt.apply(lambda row: row.Atel_diff > 0.7, axis = 1)
data_frt['PlEf_flag'] = data_frt.apply(lambda row: row.PlEf_diff > 0.7, axis = 1)

data_frt['Total_flag'] = data_frt.apply(lambda row: (row.Card_flag + row.Edem_flag + row.Cons_flag + row.Atel_flag + row.PlEf_flag) > 0, axis = 1)

data_frt2 = data_frt[data_frt['Total_flag'] == True] # 0.5: 122671, 0.7: 62468, 0.9: 15158

Traindata_frt = pd.read_csv('./CheXpert-v1.0{0}/train_frt.csv'.format(img_type)) ###
Traindata_frt = Traindata_frt.sort_values('Path').reset_index(drop=True)

data_frt3 = Traindata_frt[Traindata_frt['Path'].isin(list(data_frt2['Path']))].reset_index(drop=True)
data_frt3.to_csv('./CheXpert-v1.0{0}/train_frt0.7.csv'.format(img_type), index = False)


# lat 0.7
data_lat = pd.read_csv('./CheXpert-v1.0{0}/data_lat_boost1.csv'.format(img_type))

data_lat['Card_diff'] = data_lat.apply(lambda row: abs(row.Card - row.Cardiomegaly), axis = 1)
data_lat['Edem_diff'] = data_lat.apply(lambda row: abs(row.Edem - row.Edema), axis = 1)
data_lat['Cons_diff'] = data_lat.apply(lambda row: abs(row.Cons - row.Consolidation), axis = 1)
data_lat['Atel_diff'] = data_lat.apply(lambda row: abs(row.Atel - row.Atelectasis), axis = 1)
data_lat['PlEf_diff'] = data_lat.apply(lambda row: abs(row.PlEf - row.Pleural_Effusion), axis = 1)

data_lat['Card_flag'] = data_lat.apply(lambda row: row.Card_diff > 0.7, axis = 1)
data_lat['Edem_flag'] = data_lat.apply(lambda row: row.Edem_diff > 0.7, axis = 1)
data_lat['Cons_flag'] = data_lat.apply(lambda row: row.Cons_diff > 0.7, axis = 1)
data_lat['Atel_flag'] = data_lat.apply(lambda row: row.Atel_diff > 0.7, axis = 1)
data_lat['PlEf_flag'] = data_lat.apply(lambda row: row.PlEf_diff > 0.7, axis = 1)

data_lat['Total_flag'] = data_lat.apply(lambda row: (row.Card_flag + row.Edem_flag + row.Cons_flag + row.Atel_flag + row.PlEf_flag) > 0, axis = 1)

data_lat2 = data_lat[data_lat['Total_flag'] == True] # 0.5: 16686, 0.7: 10738, 0.9: 3197

Traindata_lat = pd.read_csv('./CheXpert-v1.0{0}/train_lat.csv'.format(img_type))
Traindata_lat = Traindata_lat.sort_values('Path').reset_index(drop=True)

data_lat3 = Traindata_lat[Traindata_lat['Path'].isin(list(data_lat2['Path']))].reset_index(drop=True)
data_lat3.to_csv('./CheXpert-v1.0{0}/train_lat0.7.csv'.format(img_type), index = False)



# frt 0.5
data_frt = pd.read_csv('./CheXpert-v1.0{0}/data_frt_boost1.csv'.format(img_type))

data_frt['Card_diff'] = data_frt.apply(lambda row: abs(row.Card - row.Cardiomegaly), axis = 1)
data_frt['Edem_diff'] = data_frt.apply(lambda row: abs(row.Edem - row.Edema), axis = 1)
data_frt['Cons_diff'] = data_frt.apply(lambda row: abs(row.Cons - row.Consolidation), axis = 1)
data_frt['Atel_diff'] = data_frt.apply(lambda row: abs(row.Atel - row.Atelectasis), axis = 1)
data_frt['PlEf_diff'] = data_frt.apply(lambda row: abs(row.PlEf - row.Pleural_Effusion), axis = 1)

data_frt['Card_flag'] = data_frt.apply(lambda row: row.Card_diff > 0.5, axis = 1)
data_frt['Edem_flag'] = data_frt.apply(lambda row: row.Edem_diff > 0.5, axis = 1)
data_frt['Cons_flag'] = data_frt.apply(lambda row: row.Cons_diff > 0.5, axis = 1)
data_frt['Atel_flag'] = data_frt.apply(lambda row: row.Atel_diff > 0.5, axis = 1)
data_frt['PlEf_flag'] = data_frt.apply(lambda row: row.PlEf_diff > 0.5, axis = 1)

data_frt['Total_flag'] = data_frt.apply(lambda row: (row.Card_flag + row.Edem_flag + row.Cons_flag + row.Atel_flag + row.PlEf_flag) > 0, axis = 1)

data_frt2 = data_frt[data_frt['Total_flag'] == True] # 0.5: 122671, 0.7: 62468, 0.9: 15158

Traindata_frt = pd.read_csv('./CheXpert-v1.0{0}/train_frt.csv'.format(img_type)) ###
Traindata_frt = Traindata_frt.sort_values('Path').reset_index(drop=True)

data_frt3 = Traindata_frt[Traindata_frt['Path'].isin(list(data_frt2['Path']))].reset_index(drop=True)
data_frt3.to_csv('./CheXpert-v1.0{0}/train_frt0.5.csv'.format(img_type), index = False)


# lat 0.5
data_lat = pd.read_csv('./CheXpert-v1.0{0}/data_lat_boost1.csv'.format(img_type))

data_lat['Card_diff'] = data_lat.apply(lambda row: abs(row.Card - row.Cardiomegaly), axis = 1)
data_lat['Edem_diff'] = data_lat.apply(lambda row: abs(row.Edem - row.Edema), axis = 1)
data_lat['Cons_diff'] = data_lat.apply(lambda row: abs(row.Cons - row.Consolidation), axis = 1)
data_lat['Atel_diff'] = data_lat.apply(lambda row: abs(row.Atel - row.Atelectasis), axis = 1)
data_lat['PlEf_diff'] = data_lat.apply(lambda row: abs(row.PlEf - row.Pleural_Effusion), axis = 1)

data_lat['Card_flag'] = data_lat.apply(lambda row: row.Card_diff > 0.5, axis = 1)
data_lat['Edem_flag'] = data_lat.apply(lambda row: row.Edem_diff > 0.5, axis = 1)
data_lat['Cons_flag'] = data_lat.apply(lambda row: row.Cons_diff > 0.5, axis = 1)
data_lat['Atel_flag'] = data_lat.apply(lambda row: row.Atel_diff > 0.5, axis = 1)
data_lat['PlEf_flag'] = data_lat.apply(lambda row: row.PlEf_diff > 0.5, axis = 1)

data_lat['Total_flag'] = data_lat.apply(lambda row: (row.Card_flag + row.Edem_flag + row.Cons_flag + row.Atel_flag + row.PlEf_flag) > 0, axis = 1)

data_lat2 = data_lat[data_lat['Total_flag'] == True] # 0.5: 16686, 0.7: 10738, 0.9: 3197

Traindata_lat = pd.read_csv('./CheXpert-v1.0{0}/train_lat.csv'.format(img_type))
Traindata_lat = Traindata_lat.sort_values('Path').reset_index(drop=True)

data_lat3 = Traindata_lat[Traindata_lat['Path'].isin(list(data_lat2['Path']))].reset_index(drop=True)
data_lat3.to_csv('./CheXpert-v1.0{0}/train_lat0.5.csv'.format(img_type), index = False)



# frt 0.9
data_frt = pd.read_csv('./CheXpert-v1.0{0}/data_frt_boost1.csv'.format(img_type))

data_frt['Card_diff'] = data_frt.apply(lambda row: abs(row.Card - row.Cardiomegaly), axis = 1)
data_frt['Edem_diff'] = data_frt.apply(lambda row: abs(row.Edem - row.Edema), axis = 1)
data_frt['Cons_diff'] = data_frt.apply(lambda row: abs(row.Cons - row.Consolidation), axis = 1)
data_frt['Atel_diff'] = data_frt.apply(lambda row: abs(row.Atel - row.Atelectasis), axis = 1)
data_frt['PlEf_diff'] = data_frt.apply(lambda row: abs(row.PlEf - row.Pleural_Effusion), axis = 1)

data_frt['Card_flag'] = data_frt.apply(lambda row: row.Card_diff > 0.9, axis = 1)
data_frt['Edem_flag'] = data_frt.apply(lambda row: row.Edem_diff > 0.9, axis = 1)
data_frt['Cons_flag'] = data_frt.apply(lambda row: row.Cons_diff > 0.9, axis = 1)
data_frt['Atel_flag'] = data_frt.apply(lambda row: row.Atel_diff > 0.9, axis = 1)
data_frt['PlEf_flag'] = data_frt.apply(lambda row: row.PlEf_diff > 0.9, axis = 1)

data_frt['Total_flag'] = data_frt.apply(lambda row: (row.Card_flag + row.Edem_flag + row.Cons_flag + row.Atel_flag + row.PlEf_flag) > 0, axis = 1)

data_frt2 = data_frt[data_frt['Total_flag'] == True] # 0.5: 122671, 0.7: 62468, 0.9: 15158

Traindata_frt = pd.read_csv('./CheXpert-v1.0{0}/train_frt.csv'.format(img_type)) ###
Traindata_frt = Traindata_frt.sort_values('Path').reset_index(drop=True)

data_frt3 = Traindata_frt[Traindata_frt['Path'].isin(list(data_frt2['Path']))].reset_index(drop=True)
data_frt3.to_csv('./CheXpert-v1.0{0}/train_frt0.9.csv'.format(img_type), index = False)


# lat 0.9
data_lat = pd.read_csv('./CheXpert-v1.0{0}/data_lat_boost1.csv'.format(img_type))

data_lat['Card_diff'] = data_lat.apply(lambda row: abs(row.Card - row.Cardiomegaly), axis = 1)
data_lat['Edem_diff'] = data_lat.apply(lambda row: abs(row.Edem - row.Edema), axis = 1)
data_lat['Cons_diff'] = data_lat.apply(lambda row: abs(row.Cons - row.Consolidation), axis = 1)
data_lat['Atel_diff'] = data_lat.apply(lambda row: abs(row.Atel - row.Atelectasis), axis = 1)
data_lat['PlEf_diff'] = data_lat.apply(lambda row: abs(row.PlEf - row.Pleural_Effusion), axis = 1)

data_lat['Card_flag'] = data_lat.apply(lambda row: row.Card_diff > 0.9, axis = 1)
data_lat['Edem_flag'] = data_lat.apply(lambda row: row.Edem_diff > 0.9, axis = 1)
data_lat['Cons_flag'] = data_lat.apply(lambda row: row.Cons_diff > 0.9, axis = 1)
data_lat['Atel_flag'] = data_lat.apply(lambda row: row.Atel_diff > 0.9, axis = 1)
data_lat['PlEf_flag'] = data_lat.apply(lambda row: row.PlEf_diff > 0.9, axis = 1)

data_lat['Total_flag'] = data_lat.apply(lambda row: (row.Card_flag + row.Edem_flag + row.Cons_flag + row.Atel_flag + row.PlEf_flag) > 0, axis = 1)

data_lat2 = data_lat[data_lat['Total_flag'] == True] # 0.5: 16686, 0.7: 10738, 0.9: 3197

Traindata_lat = pd.read_csv('./CheXpert-v1.0{0}/train_lat.csv'.format(img_type))
Traindata_lat = Traindata_lat.sort_values('Path').reset_index(drop=True)

data_lat3 = Traindata_lat[Traindata_lat['Path'].isin(list(data_lat2['Path']))].reset_index(drop=True)
data_lat3.to_csv('./CheXpert-v1.0{0}/train_lat0.9.csv'.format(img_type), index = False)



######################################
## Find Failure Success & Base Fail ## 210804
######################################
img_type = '-small' ###
PATH = './results/' ###

with open('{}210729/testPROB_all.txt'.format(PATH), 'rb') as f:
    outPROB_all = pickle.load(f)

test_agg = pd.read_csv('./CheXpert-v1.0{0}/test_agg.csv'.format(img_type))

test_agg1 = test_agg[['Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA', 
                      'Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']].copy()

test_agg1['Card_base'] = 0
test_agg1['Edem_base'] = 0
test_agg1['Cons_base'] = 0
test_agg1['Atel_base'] = 0
test_agg1['PlEf_base'] = 0

for i in range(len(outPROB_all)):
    for j in range(5):
        test_agg1.iloc[i, j+10] = outPROB_all[i][0][j]
test_agg1 = test_agg1.rename(columns={'Pleural Effusion': 'Pleural_Effusion'})

test_agg1['Card_diff_base'] = test_agg1.apply(lambda row: abs(row.Card_base - row.Cardiomegaly), axis = 1)
test_agg1['Edem_diff_base'] = test_agg1.apply(lambda row: abs(row.Edem_base - row.Edema), axis = 1)
test_agg1['Cons_diff_base'] = test_agg1.apply(lambda row: abs(row.Cons_base - row.Consolidation), axis = 1)
test_agg1['Atel_diff_base'] = test_agg1.apply(lambda row: abs(row.Atel_base - row.Atelectasis), axis = 1)
test_agg1['PlEf_diff_base'] = test_agg1.apply(lambda row: abs(row.PlEf_base - row.Pleural_Effusion), axis = 1)

test_agg1['Card_flag_base'] = test_agg1.apply(lambda row: row.Card_diff_base > 0.5, axis = 1) # fixed
test_agg1['Edem_flag_base'] = test_agg1.apply(lambda row: row.Edem_diff_base > 0.5, axis = 1) # fixed
test_agg1['Cons_flag_base'] = test_agg1.apply(lambda row: row.Cons_diff_base > 0.5, axis = 1) # fixed
test_agg1['Atel_flag_base'] = test_agg1.apply(lambda row: row.Atel_diff_base > 0.5, axis = 1) # fixed
test_agg1['PlEf_flag_base'] = test_agg1.apply(lambda row: row.PlEf_diff_base > 0.5, axis = 1) # fixed

test_agg1['Age'] = test_agg1.apply(lambda row: str(row.Age)[0], axis = 1) # convert to age range
test_agg_base = test_agg1.copy()





with open('{}210802_0.9/testPROB_all.txt'.format(PATH), 'rb') as f:
    outPROB_agg = pickle.load(f)

test_agg = pd.read_csv('./CheXpert-v1.0{0}/test_agg.csv'.format(img_type))

test_agg1 = test_agg[['Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA', 
                      'Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']].copy()

test_agg1['Card_05'] = 0
test_agg1['Edem_05'] = 0
test_agg1['Cons_05'] = 0
test_agg1['Atel_05'] = 0
test_agg1['PlEf_05'] = 0

for i in range(len(outPROB_agg)):
    for j in range(5):
        test_agg1.iloc[i, j+10] = outPROB_agg[i][0][j]
test_agg1 = test_agg1.rename(columns={'Pleural Effusion': 'Pleural_Effusion'})

test_agg1['Card_diff_05'] = test_agg1.apply(lambda row: abs(row.Card_05 - row.Cardiomegaly), axis = 1)
test_agg1['Edem_diff_05'] = test_agg1.apply(lambda row: abs(row.Edem_05 - row.Edema), axis = 1)
test_agg1['Cons_diff_05'] = test_agg1.apply(lambda row: abs(row.Cons_05 - row.Consolidation), axis = 1)
test_agg1['Atel_diff_05'] = test_agg1.apply(lambda row: abs(row.Atel_05 - row.Atelectasis), axis = 1)
test_agg1['PlEf_diff_05'] = test_agg1.apply(lambda row: abs(row.PlEf_05 - row.Pleural_Effusion), axis = 1)

test_agg1['Card_flag_05'] = test_agg1.apply(lambda row: row.Card_diff_05 > 0.5, axis = 1)
test_agg1['Edem_flag_05'] = test_agg1.apply(lambda row: row.Edem_diff_05 > 0.5, axis = 1)
test_agg1['Cons_flag_05'] = test_agg1.apply(lambda row: row.Cons_diff_05 > 0.5, axis = 1)
test_agg1['Atel_flag_05'] = test_agg1.apply(lambda row: row.Atel_diff_05 > 0.5, axis = 1)
test_agg1['PlEf_flag_05'] = test_agg1.apply(lambda row: row.PlEf_diff_05 > 0.5, axis = 1)

test_agg1['Age'] = test_agg1.apply(lambda row: str(row.Age)[0], axis = 1) # convert to age range
test_agg_05 = test_agg1.copy()


test_frt_base_05 = pd.concat([test_agg_base, test_agg_05[['Card_flag_05', 'Edem_flag_05', 'Cons_flag_05', 'Atel_flag_05', 'PlEf_flag_05']]], axis = 1)
test_frt_base_05[test_frt_base_05.Card_flag_base == True][test_frt_base_05.Card_flag_05 == False] # 18 / 65
test_frt_base_05[test_frt_base_05.Edem_flag_base == True][test_frt_base_05.Edem_flag_05 == False] # 17 / 26
test_frt_base_05[test_frt_base_05.Cons_flag_base == True][test_frt_base_05.Cons_flag_05 == False] #  0 / 32
test_frt_base_05[test_frt_base_05.Atel_flag_base == True][test_frt_base_05.Atel_flag_05 == False] # 16 / 51
test_frt_base_05[test_frt_base_05.PlEf_flag_base == True][test_frt_base_05.PlEf_flag_05 == False] #  8 / 30
# Checked out possibility of performance improvement



#########################################
## Agg base model & failure only model ## ############################
#########################################

img_type = '-small' ###
PATH = './results/' ###

with open('{}210729/testPROB_all.txt'.format(PATH), 'rb') as f:
    outPROB_base = pickle.load(f)

with open('{}210802_0.5/testPROB_all.txt'.format(PATH), 'rb') as f:
    outPROB_05 = pickle.load(f)

outPROB_base_05 = []
'''
for i in range(len(outPROB_base)):
    outPROB_base_05.append(list(map(min, zip(outPROB_base[i][0], outPROB_05[i][0])))) # min, max
'''
for i in range(len(outPROB_base)):
    outPROB_base_05.append(list(map(lambda x, y: (10*x + 0*y) / 10, outPROB_base[i][0], outPROB_05[i][0]))) # min, max


# Tranform data
imgtransResize = 320
transformList = []
transformList.append(transforms.Resize((imgtransResize, imgtransResize))) # 320
transformList.append(transforms.ToTensor())
transformSequence = transforms.Compose(transformList)

class_names = ["Card", "Edem", "Cons", "Atel", "PlEf"]
nnClassCount = 5
policy = "diff"
pathFileTest_agg = './CheXpert-v1.0{0}/test_agg.csv'.format(img_type)
datasetTest_agg = CheXpertDataSet(pathFileTest_agg, nnClassCount, policy, transformSequence)
dataLoaderTest_agg = DataLoader(dataset = datasetTest_agg, num_workers = 2, pin_memory = True)



# Draw ROC curves
EnsemTest = outPROB_base_05
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

#plt.savefig('{0}ROC_{1}.png'.format(PATH, nnClassCount), dpi = 100)
plt.close()

'''
BASE
<<< Ensembles Test Results: AUROC >>>
MEAN : 0.8757
Card : 0.8357
Edem : 0.9024
Cons : 0.9115
Atel : 0.8111
PlEf : 0.9180



FO_05
<<< Ensembles Test Results: AUROC >>>
MEAN : 0.8363
Card : 0.8033
Edem : 0.8657
Cons : 0.8744
Atel : 0.7471
PlEf : 0.8912

with MIN agg metric
<<< Ensembles Test Results: AUROC >>>
MEAN : 0.8716
Card : 0.8393
Edem : 0.9027
Cons : 0.9122
Atel : 0.7929
PlEf : 0.9110



FO_07
<<< Ensembles Test Results: AUROC >>>
MEAN : 0.7368
Card : 0.7933
Edem : 0.8415
Cons : 0.8244
Atel : 0.4640
PlEf : 0.7610

with MIN agg metric
<<< Ensembles Test Results: AUROC >>>
MEAN : 0.8599
Card : 0.8368
Edem : 0.8998
Cons : 0.9122
Atel : 0.7545
PlEf : 0.8960



FO_09
<<< Ensembles Test Results: AUROC >>>
MEAN : 0.7479
Card : 0.7154
Edem : 0.8118
Cons : 0.7494
Atel : 0.7044
PlEf : 0.7585

with MIN agg metric
<<< Ensembles Test Results: AUROC >>>
MEAN : 0.8580
Card : 0.8357
Edem : 0.8989
Cons : 0.9115
Atel : 0.7613
PlEf : 0.8825
'''



########################################
## Finding Base model best agg metric ##
########################################
img_type = '-small' ###
PATH = './results/' ###

with open('{}210729/testPROB_frt.txt'.format(PATH), 'rb') as f:
    outPROB_frt = pickle.load(f)

with open('{}210729/testPROB_lat.txt'.format(PATH), 'rb') as f:
    outPROB_lat = pickle.load(f)

# Tranform data
imgtransResize = 320
transformList = []
transformList.append(transforms.Resize((imgtransResize, imgtransResize))) # 320
transformList.append(transforms.ToTensor())
transformSequence = transforms.Compose(transformList)

class_names = ["Card", "Edem", "Cons", "Atel", "PlEf"]
nnClassCount = 5
policy = "diff"
pathFileTest_agg = './CheXpert-v1.0{0}/test_agg.csv'.format(img_type)
datasetTest_agg = CheXpertDataSet(pathFileTest_agg, nnClassCount, policy, transformSequence)
dataLoaderTest_agg = DataLoader(dataset = datasetTest_agg, num_workers = 2, pin_memory = True)

pathFileTest_frt = './CheXpert-v1.0{0}/test_frt.csv'.format(img_type)
pathFileTest_lat = './CheXpert-v1.0{0}/test_lat.csv'.format(img_type)
pathFileTest_agg = './CheXpert-v1.0{0}/test_agg.csv'.format(img_type)

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
                                 'PlEf' : 'min'}).reset_index() # max -> mean -> min
df_agg = df_agg.sort_values('Path')
results = df_agg.drop(['Path'], axis = 1).values.tolist()

# Save the test outPROB_all
outPROB_all = []
for i in range(len(results)):
    outPROB_all.append([results[i]])

with open('{}210729/testPROB_all.txt'.format(PATH), 'wb') as fp:
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

#plt.savefig('{0}210729/ROC_{1}.png'.format(PATH, nnClassCount), dpi = 100)
plt.close()



'''
{'Card' : 'min',
 'Edem' : 'max',
 'Cons' : 'min',
 'Atel' : 'mean', # 'max'
 'PlEf' : 'max'}
'''


##########################
## Sample Discriminator ##
##########################
img_type = '-small' ###
PATH = './results/' ###

data_frt = pd.read_csv('./CheXpert-v1.0{0}/data_frt_boost1.csv'.format(img_type))
data_frt['Card_diff'] = data_frt.apply(lambda row: abs(row.Card - row.Cardiomegaly), axis = 1)
data_frt['Edem_diff'] = data_frt.apply(lambda row: abs(row.Edem - row.Edema), axis = 1)
data_frt['Cons_diff'] = data_frt.apply(lambda row: abs(row.Cons - row.Consolidation), axis = 1)
data_frt['Atel_diff'] = data_frt.apply(lambda row: abs(row.Atel - row.Atelectasis), axis = 1)
data_frt['PlEf_diff'] = data_frt.apply(lambda row: abs(row.PlEf - row.Pleural_Effusion), axis = 1)
data_frt['Card_flag'] = data_frt.apply(lambda row: float(row.Card_diff > 0.5), axis = 1) # 1 means bad sample
data_frt['Edem_flag'] = data_frt.apply(lambda row: float(row.Edem_diff > 0.5), axis = 1) # 0 means good sample
data_frt['Cons_flag'] = data_frt.apply(lambda row: float(row.Cons_diff > 0.5), axis = 1)
data_frt['Atel_flag'] = data_frt.apply(lambda row: float(row.Atel_diff > 0.5), axis = 1)
data_frt['PlEf_flag'] = data_frt.apply(lambda row: float(row.PlEf_diff > 0.5), axis = 1)

train_frt = pd.read_csv('./CheXpert-v1.0-small/train_frt.csv')
train_frt['Cardiomegaly'] = data_frt['Card_flag']
train_frt['Edema'] = data_frt['Edem_flag']
train_frt['Consolidation'] = data_frt['Cons_flag']
train_frt['Atelectasis'] = data_frt['Atel_flag']
train_frt['Pleural Effusion'] = data_frt['PlEf_flag']

train_frt.to_csv('./CheXpert-v1.0{0}/train_frt_SD.csv'.format(img_type), index = False)



data_lat = pd.read_csv('./CheXpert-v1.0{0}/data_lat_boost1.csv'.format(img_type))
data_lat['Card_diff'] = data_lat.apply(lambda row: abs(row.Card - row.Cardiomegaly), axis = 1)
data_lat['Edem_diff'] = data_lat.apply(lambda row: abs(row.Edem - row.Edema), axis = 1)
data_lat['Cons_diff'] = data_lat.apply(lambda row: abs(row.Cons - row.Consolidation), axis = 1)
data_lat['Atel_diff'] = data_lat.apply(lambda row: abs(row.Atel - row.Atelectasis), axis = 1)
data_lat['PlEf_diff'] = data_lat.apply(lambda row: abs(row.PlEf - row.Pleural_Effusion), axis = 1)
data_lat['Card_flag'] = data_lat.apply(lambda row: float(row.Card_diff > 0.5), axis = 1) # 1 means bad sample
data_lat['Edem_flag'] = data_lat.apply(lambda row: float(row.Edem_diff > 0.5), axis = 1) # 0 means good sample
data_lat['Cons_flag'] = data_lat.apply(lambda row: float(row.Cons_diff > 0.5), axis = 1)
data_lat['Atel_flag'] = data_lat.apply(lambda row: float(row.Atel_diff > 0.5), axis = 1)
data_lat['PlEf_flag'] = data_lat.apply(lambda row: float(row.PlEf_diff > 0.5), axis = 1)

train_lat = pd.read_csv('./CheXpert-v1.0-small/train_lat.csv')
train_lat['Cardiomegaly'] = data_lat['Card_flag']
train_lat['Edema'] = data_lat['Edem_flag']
train_lat['Consolidation'] = data_lat['Cons_flag']
train_lat['Atelectasis'] = data_lat['Atel_flag']
train_lat['Pleural Effusion'] = data_lat['PlEf_flag']

train_lat.to_csv('./CheXpert-v1.0{0}/train_lat_SD.csv'.format(img_type), index = False)



##################################
## SD sample weight calculation ##
##################################

img_type = '-small' ###
PATH = './results/' ###

with open('{}210806/testPROB_all.txt'.format(PATH), 'rb') as f:
    outPROB_base = pickle.load(f)

with open('{}210802_0.9/testPROB_all.txt'.format(PATH), 'rb') as f:
    outPROB_05 = pickle.load(f)

with open('{}210813/testPROB_all.txt'.format(PATH), 'rb') as f:
    outPROB_sd = pickle.load(f)

outPROB_base_05 = []
'''
for i in range(len(outPROB_base)):
    outPROB_base_05.append(list(map(min, zip(outPROB_base[i][0], outPROB_05[i][0])))) # min, max
'''
for i in range(len(outPROB_base)):
    outPROB_base_05.append(list(map(lambda x, y, z: (1-z)*x + z*y, outPROB_base[i][0], outPROB_05[i][0], outPROB_sd[i][0]))) # min, max


# Tranform data
imgtransResize = 320
transformList = []
transformList.append(transforms.Resize((imgtransResize, imgtransResize))) # 320
transformList.append(transforms.ToTensor())
transformSequence = transforms.Compose(transformList)

class_names = ["Card", "Edem", "Cons", "Atel", "PlEf"]
nnClassCount = 5
policy = "diff"
pathFileTest_agg = './CheXpert-v1.0{0}/test_agg.csv'.format(img_type)
datasetTest_agg = CheXpertDataSet(pathFileTest_agg, nnClassCount, policy, transformSequence)
dataLoaderTest_agg = DataLoader(dataset = datasetTest_agg, num_workers = 2, pin_memory = True)



# Draw ROC curves
EnsemTest = outPROB_base_05
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

#plt.savefig('{0}ROC_{1}.png'.format(PATH, nnClassCount), dpi = 100)
plt.close()



################################################
## SD sample weight calculation (frt and lat) ##
################################################

img_type = '-small' ###
PATH = './results/' ###

with open('{}210729/testPROB_frt.txt'.format(PATH), 'rb') as f:
    outPROB_frt_base = pickle.load(f)

with open('{}210729/testPROB_lat.txt'.format(PATH), 'rb') as f:
    outPROB_lat_base = pickle.load(f)

'''For overall failure model
with open('{}210802_0.9/testPROB_frt.txt'.format(PATH), 'rb') as f:
    outPROB_frt_05 = pickle.load(f)

with open('{}210802_0.9/testPROB_lat.txt'.format(PATH), 'rb') as f:
    outPROB_lat_05 = pickle.load(f)
'''
with open('{}210824_F0.5_Card/testPROB_frt.txt'.format(PATH), 'rb') as f:
    outPROB_frt_05_Card = pickle.load(f)

with open('{}210824_F0.5_Edem/testPROB_frt.txt'.format(PATH), 'rb') as f:
    outPROB_frt_05_Edem = pickle.load(f)

with open('{}210824_F0.5_Cons/testPROB_frt.txt'.format(PATH), 'rb') as f:
    outPROB_frt_05_Cons = pickle.load(f)

with open('{}210824_F0.5_Atel/testPROB_frt.txt'.format(PATH), 'rb') as f:
    outPROB_frt_05_Atel = pickle.load(f)

with open('{}210824_F0.5_PlEf/testPROB_frt.txt'.format(PATH), 'rb') as f:
    outPROB_frt_05_PlEf = pickle.load(f)

outPROB_frt_05 = [[[0, 0, 0, 0, 0]] for i in range(len(outPROB_frt_base))]
for i in range(outPROB_frt_base):
    outPROB_frt_05[i][0][0] = outPROB_frt_05_Card[i][0][0]
    outPROB_frt_05[i][0][1] = outPROB_frt_05_Edem[i][0][1]
    outPROB_frt_05[i][0][2] = outPROB_frt_05_Cons[i][0][2]
    outPROB_frt_05[i][0][3] = outPROB_frt_05_Atel[i][0][3]
    outPROB_frt_05[i][0][4] = outPROB_frt_05_PlEf[i][0][4]

with open('{}210824_F0.5_Card/testPROB_lat.txt'.format(PATH), 'rb') as f:
    outPROB_lat_05_Card = pickle.load(f)

with open('{}210824_F0.5_Edem/testPROB_lat.txt'.format(PATH), 'rb') as f:
    outPROB_lat_05_Edem = pickle.load(f)

with open('{}210824_F0.5_Cons/testPROB_lat.txt'.format(PATH), 'rb') as f:
    outPROB_lat_05_Cons = pickle.load(f)

with open('{}210824_F0.5_Atel/testPROB_lat.txt'.format(PATH), 'rb') as f:
    outPROB_lat_05_Atel = pickle.load(f)

with open('{}210824_F0.5_PlEf/testPROB_lat.txt'.format(PATH), 'rb') as f:
    outPROB_lat_05_PlEf = pickle.load(f)

outPROB_lat_05 = [[[0, 0, 0, 0, 0]] for i in range(len(outPROB_lat_base))]
for i in range(outPROB_lat_base):
    outPROB_lat_05[i][0][0] = outPROB_lat_05_Card[i][0][0]
    outPROB_lat_05[i][0][1] = outPROB_lat_05_Edem[i][0][1]
    outPROB_lat_05[i][0][2] = outPROB_lat_05_Cons[i][0][2]
    outPROB_lat_05[i][0][3] = outPROB_lat_05_Atel[i][0][3]
    outPROB_lat_05[i][0][4] = outPROB_lat_05_PlEf[i][0][4]

with open('{}210813/testPROB_frt.txt'.format(PATH), 'rb') as f:
    outPROB_frt_sd = pickle.load(f)

with open('{}210813/testPROB_lat.txt'.format(PATH), 'rb') as f:
    outPROB_lat_sd = pickle.load(f)


outPROB_base_05_frt = []
'''
for i in range(len(outPROB_base)):
    outPROB_base_05.append(list(map(min, zip(outPROB_base[i][0], outPROB_05[i][0])))) # min, max
'''
for i in range(len(outPROB_frt_base)):
    outPROB_base_05_frt.append(list(map(lambda x, y, z: (1-z)*x + z*y, outPROB_frt_base[i][0], outPROB_frt_05[i][0], outPROB_frt_sd[i][0]))) # min, max
# (1-z)*x + z*y

outPROB_base_05_lat = []
'''
for i in range(len(outPROB_base)):
    outPROB_base_05.append(list(map(min, zip(outPROB_base[i][0], outPROB_05[i][0])))) # min, max
'''
for i in range(len(outPROB_lat_base)):
    outPROB_base_05_lat.append(list(map(lambda x, y, z: (1-z)*x + z*y, outPROB_lat_base[i][0], outPROB_lat_05[i][0], outPROB_lat_sd[i][0]))) # min, max
# (1-z)*x + z*y

# Tranform data
imgtransResize = 320
transformList = []
transformList.append(transforms.Resize((imgtransResize, imgtransResize))) # 320
transformList.append(transforms.ToTensor())
transformSequence = transforms.Compose(transformList)

class_names = ["Card", "Edem", "Cons", "Atel", "PlEf"]
nnClassCount = 5
policy = "diff"
pathFileTest_frt = './CheXpert-v1.0{0}/test_frt.csv'.format(img_type)
pathFileTest_lat = './CheXpert-v1.0{0}/test_lat.csv'.format(img_type)
pathFileTest_agg = './CheXpert-v1.0{0}/test_agg.csv'.format(img_type)
datasetTest_agg = CheXpertDataSet(pathFileTest_agg, nnClassCount, policy, transformSequence)
dataLoaderTest_agg = DataLoader(dataset = datasetTest_agg, num_workers = 2, pin_memory = True)

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

for i in range(len(outPROB_base_05_frt)):
    for j in range(len(class_names)):
        df.iloc[i, j + 1] = outPROB_base_05_frt[i][j]
        
for i in range(len(outPROB_base_05_lat)):
    for j in range(len(class_names)):
        df.iloc[len(outPROB_base_05_frt) + i, j + 1] = outPROB_base_05_lat[i][j]

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

#plt.savefig('{0}ROC_{1}.png'.format(PATH, nnClassCount), dpi = 100)
plt.close()




###################################
## Train with only success modes ## ### for each
###################################
img_type = '-small' ###
PATH = './results/' ###

# frt 0.5
data_frt = pd.read_csv('./CheXpert-v1.0{0}/data_frt_boost1.csv'.format(img_type))

data_frt['Card_diff'] = data_frt.apply(lambda row: abs(row.Card - row.Cardiomegaly), axis = 1)
data_frt['Edem_diff'] = data_frt.apply(lambda row: abs(row.Edem - row.Edema), axis = 1)
data_frt['Cons_diff'] = data_frt.apply(lambda row: abs(row.Cons - row.Consolidation), axis = 1)
data_frt['Atel_diff'] = data_frt.apply(lambda row: abs(row.Atel - row.Atelectasis), axis = 1)
data_frt['PlEf_diff'] = data_frt.apply(lambda row: abs(row.PlEf - row.Pleural_Effusion), axis = 1)

data_frt['Card_flag'] = data_frt.apply(lambda row: row.Card_diff > 0.5, axis = 1)
data_frt['Edem_flag'] = data_frt.apply(lambda row: row.Edem_diff > 0.5, axis = 1)
data_frt['Cons_flag'] = data_frt.apply(lambda row: row.Cons_diff > 0.5, axis = 1)
data_frt['Atel_flag'] = data_frt.apply(lambda row: row.Atel_diff > 0.5, axis = 1)
data_frt['PlEf_flag'] = data_frt.apply(lambda row: row.PlEf_diff > 0.5, axis = 1)

data_frt_Card = data_frt[data_frt['Card_flag'] == False]
data_frt_Edem = data_frt[data_frt['Edem_flag'] == False]
data_frt_Cons = data_frt[data_frt['Cons_flag'] == False]
data_frt_Atel = data_frt[data_frt['Atel_flag'] == False]
data_frt_PlEf = data_frt[data_frt['PlEf_flag'] == False]

Traindata_frt = pd.read_csv('./CheXpert-v1.0{0}/train_frt.csv'.format(img_type)) ###
Traindata_frt = Traindata_frt.sort_values('Path').reset_index(drop=True)

data_frt_Card2 = Traindata_frt[Traindata_frt['Path'].isin(list(data_frt_Card['Path']))].reset_index(drop=True)
data_frt_Edem2 = Traindata_frt[Traindata_frt['Path'].isin(list(data_frt_Edem['Path']))].reset_index(drop=True)
data_frt_Cons2 = Traindata_frt[Traindata_frt['Path'].isin(list(data_frt_Cons['Path']))].reset_index(drop=True)
data_frt_Atel2 = Traindata_frt[Traindata_frt['Path'].isin(list(data_frt_Atel['Path']))].reset_index(drop=True)
data_frt_PlEf2 = Traindata_frt[Traindata_frt['Path'].isin(list(data_frt_PlEf['Path']))].reset_index(drop=True)

data_frt_Card2.to_csv('./CheXpert-v1.0{0}/train_frt0.5s_Card.csv'.format(img_type), index = False)
data_frt_Edem2.to_csv('./CheXpert-v1.0{0}/train_frt0.5s_Edem.csv'.format(img_type), index = False)
data_frt_Cons2.to_csv('./CheXpert-v1.0{0}/train_frt0.5s_Cons.csv'.format(img_type), index = False)
data_frt_Atel2.to_csv('./CheXpert-v1.0{0}/train_frt0.5s_Atel.csv'.format(img_type), index = False)
data_frt_PlEf2.to_csv('./CheXpert-v1.0{0}/train_frt0.5s_PlEf.csv'.format(img_type), index = False)


# lat 0.5
data_lat = pd.read_csv('./CheXpert-v1.0{0}/data_lat_boost1.csv'.format(img_type))

data_lat['Card_diff'] = data_lat.apply(lambda row: abs(row.Card - row.Cardiomegaly), axis = 1)
data_lat['Edem_diff'] = data_lat.apply(lambda row: abs(row.Edem - row.Edema), axis = 1)
data_lat['Cons_diff'] = data_lat.apply(lambda row: abs(row.Cons - row.Consolidation), axis = 1)
data_lat['Atel_diff'] = data_lat.apply(lambda row: abs(row.Atel - row.Atelectasis), axis = 1)
data_lat['PlEf_diff'] = data_lat.apply(lambda row: abs(row.PlEf - row.Pleural_Effusion), axis = 1)

data_lat['Card_flag'] = data_lat.apply(lambda row: row.Card_diff > 0.5, axis = 1)
data_lat['Edem_flag'] = data_lat.apply(lambda row: row.Edem_diff > 0.5, axis = 1)
data_lat['Cons_flag'] = data_lat.apply(lambda row: row.Cons_diff > 0.5, axis = 1)
data_lat['Atel_flag'] = data_lat.apply(lambda row: row.Atel_diff > 0.5, axis = 1)
data_lat['PlEf_flag'] = data_lat.apply(lambda row: row.PlEf_diff > 0.5, axis = 1)

data_lat_Card = data_lat[data_lat['Card_flag'] == False]
data_lat_Edem = data_lat[data_lat['Edem_flag'] == False]
data_lat_Cons = data_lat[data_lat['Cons_flag'] == False]
data_lat_Atel = data_lat[data_lat['Atel_flag'] == False]
data_lat_PlEf = data_lat[data_lat['PlEf_flag'] == False]

Traindata_lat = pd.read_csv('./CheXpert-v1.0{0}/train_lat.csv'.format(img_type)) ###
Traindata_lat = Traindata_lat.sort_values('Path').reset_index(drop=True)

data_lat_Card2 = Traindata_lat[Traindata_lat['Path'].isin(list(data_lat_Card['Path']))].reset_index(drop=True)
data_lat_Edem2 = Traindata_lat[Traindata_lat['Path'].isin(list(data_lat_Edem['Path']))].reset_index(drop=True)
data_lat_Cons2 = Traindata_lat[Traindata_lat['Path'].isin(list(data_lat_Cons['Path']))].reset_index(drop=True)
data_lat_Atel2 = Traindata_lat[Traindata_lat['Path'].isin(list(data_lat_Atel['Path']))].reset_index(drop=True)
data_lat_PlEf2 = Traindata_lat[Traindata_lat['Path'].isin(list(data_lat_PlEf['Path']))].reset_index(drop=True)

data_lat_Card2.to_csv('./CheXpert-v1.0{0}/train_lat0.5s_Card.csv'.format(img_type), index = False)
data_lat_Edem2.to_csv('./CheXpert-v1.0{0}/train_lat0.5s_Edem.csv'.format(img_type), index = False)
data_lat_Cons2.to_csv('./CheXpert-v1.0{0}/train_lat0.5s_Cons.csv'.format(img_type), index = False)
data_lat_Atel2.to_csv('./CheXpert-v1.0{0}/train_lat0.5s_Atel.csv'.format(img_type), index = False)
data_lat_PlEf2.to_csv('./CheXpert-v1.0{0}/train_lat0.5s_PlEf.csv'.format(img_type), index = False)




###################################
## Train with only failure modes ## ### for each
###################################
img_type = '-small' ###
PATH = './results/' ###

# frt 0.5
data_frt = pd.read_csv('./CheXpert-v1.0{0}/data_frt_boost1.csv'.format(img_type))

data_frt['Card_diff'] = data_frt.apply(lambda row: abs(row.Card - row.Cardiomegaly), axis = 1)
data_frt['Edem_diff'] = data_frt.apply(lambda row: abs(row.Edem - row.Edema), axis = 1)
data_frt['Cons_diff'] = data_frt.apply(lambda row: abs(row.Cons - row.Consolidation), axis = 1)
data_frt['Atel_diff'] = data_frt.apply(lambda row: abs(row.Atel - row.Atelectasis), axis = 1)
data_frt['PlEf_diff'] = data_frt.apply(lambda row: abs(row.PlEf - row.Pleural_Effusion), axis = 1)

data_frt['Card_flag'] = data_frt.apply(lambda row: row.Card_diff > 0.5, axis = 1)
data_frt['Edem_flag'] = data_frt.apply(lambda row: row.Edem_diff > 0.5, axis = 1)
data_frt['Cons_flag'] = data_frt.apply(lambda row: row.Cons_diff > 0.5, axis = 1)
data_frt['Atel_flag'] = data_frt.apply(lambda row: row.Atel_diff > 0.5, axis = 1)
data_frt['PlEf_flag'] = data_frt.apply(lambda row: row.PlEf_diff > 0.5, axis = 1)

data_frt_Card = data_frt[data_frt['Card_flag'] == True]
data_frt_Edem = data_frt[data_frt['Edem_flag'] == True]
data_frt_Cons = data_frt[data_frt['Cons_flag'] == True]
data_frt_Atel = data_frt[data_frt['Atel_flag'] == True]
data_frt_PlEf = data_frt[data_frt['PlEf_flag'] == True]

Traindata_frt = pd.read_csv('./CheXpert-v1.0{0}/train_frt.csv'.format(img_type)) ###
Traindata_frt = Traindata_frt.sort_values('Path').reset_index(drop=True)

data_frt_Card2 = Traindata_frt[Traindata_frt['Path'].isin(list(data_frt_Card['Path']))].reset_index(drop=True)
data_frt_Edem2 = Traindata_frt[Traindata_frt['Path'].isin(list(data_frt_Edem['Path']))].reset_index(drop=True)
data_frt_Cons2 = Traindata_frt[Traindata_frt['Path'].isin(list(data_frt_Cons['Path']))].reset_index(drop=True)
data_frt_Atel2 = Traindata_frt[Traindata_frt['Path'].isin(list(data_frt_Atel['Path']))].reset_index(drop=True)
data_frt_PlEf2 = Traindata_frt[Traindata_frt['Path'].isin(list(data_frt_PlEf['Path']))].reset_index(drop=True)

data_frt_Card2.to_csv('./CheXpert-v1.0{0}/train_frt0.5f_Card.csv'.format(img_type), index = False)
data_frt_Edem2.to_csv('./CheXpert-v1.0{0}/train_frt0.5f_Edem.csv'.format(img_type), index = False)
data_frt_Cons2.to_csv('./CheXpert-v1.0{0}/train_frt0.5f_Cons.csv'.format(img_type), index = False)
data_frt_Atel2.to_csv('./CheXpert-v1.0{0}/train_frt0.5f_Atel.csv'.format(img_type), index = False)
data_frt_PlEf2.to_csv('./CheXpert-v1.0{0}/train_frt0.5f_PlEf.csv'.format(img_type), index = False)


# lat 0.5
data_lat = pd.read_csv('./CheXpert-v1.0{0}/data_lat_boost1.csv'.format(img_type))

data_lat['Card_diff'] = data_lat.apply(lambda row: abs(row.Card - row.Cardiomegaly), axis = 1)
data_lat['Edem_diff'] = data_lat.apply(lambda row: abs(row.Edem - row.Edema), axis = 1)
data_lat['Cons_diff'] = data_lat.apply(lambda row: abs(row.Cons - row.Consolidation), axis = 1)
data_lat['Atel_diff'] = data_lat.apply(lambda row: abs(row.Atel - row.Atelectasis), axis = 1)
data_lat['PlEf_diff'] = data_lat.apply(lambda row: abs(row.PlEf - row.Pleural_Effusion), axis = 1)

data_lat['Card_flag'] = data_lat.apply(lambda row: row.Card_diff > 0.5, axis = 1)
data_lat['Edem_flag'] = data_lat.apply(lambda row: row.Edem_diff > 0.5, axis = 1)
data_lat['Cons_flag'] = data_lat.apply(lambda row: row.Cons_diff > 0.5, axis = 1)
data_lat['Atel_flag'] = data_lat.apply(lambda row: row.Atel_diff > 0.5, axis = 1)
data_lat['PlEf_flag'] = data_lat.apply(lambda row: row.PlEf_diff > 0.5, axis = 1)

data_lat_Card = data_lat[data_lat['Card_flag'] == True]
data_lat_Edem = data_lat[data_lat['Edem_flag'] == True]
data_lat_Cons = data_lat[data_lat['Cons_flag'] == True]
data_lat_Atel = data_lat[data_lat['Atel_flag'] == True]
data_lat_PlEf = data_lat[data_lat['PlEf_flag'] == True]

Traindata_lat = pd.read_csv('./CheXpert-v1.0{0}/train_lat.csv'.format(img_type)) ###
Traindata_lat = Traindata_lat.sort_values('Path').reset_index(drop=True)

data_lat_Card2 = Traindata_lat[Traindata_lat['Path'].isin(list(data_lat_Card['Path']))].reset_index(drop=True)
data_lat_Edem2 = Traindata_lat[Traindata_lat['Path'].isin(list(data_lat_Edem['Path']))].reset_index(drop=True)
data_lat_Cons2 = Traindata_lat[Traindata_lat['Path'].isin(list(data_lat_Cons['Path']))].reset_index(drop=True)
data_lat_Atel2 = Traindata_lat[Traindata_lat['Path'].isin(list(data_lat_Atel['Path']))].reset_index(drop=True)
data_lat_PlEf2 = Traindata_lat[Traindata_lat['Path'].isin(list(data_lat_PlEf['Path']))].reset_index(drop=True)

data_lat_Card2.to_csv('./CheXpert-v1.0{0}/train_lat0.5f_Card.csv'.format(img_type), index = False)
data_lat_Edem2.to_csv('./CheXpert-v1.0{0}/train_lat0.5f_Edem.csv'.format(img_type), index = False)
data_lat_Cons2.to_csv('./CheXpert-v1.0{0}/train_lat0.5f_Cons.csv'.format(img_type), index = False)
data_lat_Atel2.to_csv('./CheXpert-v1.0{0}/train_lat0.5f_Atel.csv'.format(img_type), index = False)
data_lat_PlEf2.to_csv('./CheXpert-v1.0{0}/train_lat0.5f_PlEf.csv'.format(img_type), index = False)