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
with open('{}210826_0.5f_Card/testPROB_frt.txt'.format(PATH), 'rb') as f:
    outPROB_frt_05_Card = pickle.load(f)

with open('{}210826_0.5f_Edem/testPROB_frt.txt'.format(PATH), 'rb') as f:
    outPROB_frt_05_Edem = pickle.load(f)

with open('{}210826_0.5f_Cons/testPROB_frt.txt'.format(PATH), 'rb') as f:
    outPROB_frt_05_Cons = pickle.load(f)

with open('{}210826_0.5f_Atel/testPROB_frt.txt'.format(PATH), 'rb') as f:
    outPROB_frt_05_Atel = pickle.load(f)

with open('{}210826_0.5f_PlEf/testPROB_frt.txt'.format(PATH), 'rb') as f:
    outPROB_frt_05_PlEf = pickle.load(f)

outPROB_frt_05 = [[[0, 0, 0, 0, 0]] for i in range(len(outPROB_frt_base))]
for i in range(len(outPROB_frt_base)):
    outPROB_frt_05[i][0][0] = outPROB_frt_05_Card[i][0][0]
    outPROB_frt_05[i][0][1] = outPROB_frt_05_Edem[i][0][1]
    outPROB_frt_05[i][0][2] = outPROB_frt_05_Cons[i][0][2]
    outPROB_frt_05[i][0][3] = outPROB_frt_05_Atel[i][0][3]
    outPROB_frt_05[i][0][4] = outPROB_frt_05_PlEf[i][0][4]

with open('{}210826_0.5f_Card/testPROB_lat.txt'.format(PATH), 'rb') as f:
    outPROB_lat_05_Card = pickle.load(f)

with open('{}210826_0.5f_Edem/testPROB_lat.txt'.format(PATH), 'rb') as f:
    outPROB_lat_05_Edem = pickle.load(f)

with open('{}210826_0.5f_Cons/testPROB_lat.txt'.format(PATH), 'rb') as f:
    outPROB_lat_05_Cons = pickle.load(f)

with open('{}210826_0.5f_Atel/testPROB_lat.txt'.format(PATH), 'rb') as f:
    outPROB_lat_05_Atel = pickle.load(f)

with open('{}210826_0.5f_PlEf/testPROB_lat.txt'.format(PATH), 'rb') as f:
    outPROB_lat_05_PlEf = pickle.load(f)

outPROB_lat_05 = [[[0, 0, 0, 0, 0]] for i in range(len(outPROB_lat_base))]
for i in range(len(outPROB_lat_base)):
    outPROB_lat_05[i][0][0] = outPROB_lat_05_Card[i][0][0]
    outPROB_lat_05[i][0][1] = outPROB_lat_05_Edem[i][0][1]
    outPROB_lat_05[i][0][2] = outPROB_lat_05_Cons[i][0][2]
    outPROB_lat_05[i][0][3] = outPROB_lat_05_Atel[i][0][3]
    outPROB_lat_05[i][0][4] = outPROB_lat_05_PlEf[i][0][4]

with open('{}210813/testPROB_frt.txt'.format(PATH), 'rb') as f:
    outPROB_frt_sd = pickle.load(f)

with open('{}210813/testPROB_lat.txt'.format(PATH), 'rb') as f:
    outPROB_lat_sd = pickle.load(f)


# Suppress Model Q (sd) to 0 or 1
for i in range(len(outPROB_frt_sd)):
    for j in range(5):
        if outPROB_frt_sd[i][0][j] < 0.5:
            outPROB_frt_sd[i][0][j] = 0
        elif outPROB_frt_sd[i][0][j] > 0.5:
            outPROB_frt_sd[i][0][j] = 1

for i in range(len(outPROB_lat_sd)):
    for j in range(5):
        if outPROB_lat_sd[i][0][j] < 0.5:
            outPROB_lat_sd[i][0][j] = 0
        elif outPROB_lat_sd[i][0][j] > 0.5:
            outPROB_lat_sd[i][0][j] = 1


outPROB_base_05_frt = []
'''
for i in range(len(outPROB_base)):
    outPROB_base_05.append(list(map(min, zip(outPROB_base[i][0], outPROB_05[i][0])))) # min, max
'''
for i in range(len(outPROB_frt_base)):
    outPROB_base_05_frt.append(list(map(lambda x, y, z: (1-z)*x + z*(1-y), outPROB_frt_base[i][0], outPROB_frt_05[i][0], outPROB_frt_sd[i][0]))) # min, max
# (1-z)*x + z*y

outPROB_base_05_lat = []
'''
for i in range(len(outPROB_base)):
    outPROB_base_05.append(list(map(min, zip(outPROB_base[i][0], outPROB_05[i][0])))) # min, max
'''
for i in range(len(outPROB_lat_base)):
    outPROB_base_05_lat.append(list(map(lambda x, y, z: (1-z)*x + z*(1-y), outPROB_lat_base[i][0], outPROB_lat_05[i][0], outPROB_lat_sd[i][0]))) # min, max
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







################################################
## SD sample weight calculation (frt and lat) ## ## Good / Bad classification
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
with open('{}210826_0.5f_Card/testPROB_frt.txt'.format(PATH), 'rb') as f:
    outPROB_frt_05_Card = pickle.load(f)

with open('{}210826_0.5f_Edem/testPROB_frt.txt'.format(PATH), 'rb') as f:
    outPROB_frt_05_Edem = pickle.load(f)

with open('{}210826_0.5f_Cons/testPROB_frt.txt'.format(PATH), 'rb') as f:
    outPROB_frt_05_Cons = pickle.load(f)

with open('{}210826_0.5f_Atel/testPROB_frt.txt'.format(PATH), 'rb') as f:
    outPROB_frt_05_Atel = pickle.load(f)

with open('{}210826_0.5f_PlEf/testPROB_frt.txt'.format(PATH), 'rb') as f:
    outPROB_frt_05_PlEf = pickle.load(f)

outPROB_frt_05 = [[[0, 0, 0, 0, 0]] for i in range(len(outPROB_frt_base))]
for i in range(len(outPROB_frt_base)):
    outPROB_frt_05[i][0][0] = outPROB_frt_05_Card[i][0][0]
    outPROB_frt_05[i][0][1] = outPROB_frt_05_Edem[i][0][1]
    outPROB_frt_05[i][0][2] = outPROB_frt_05_Cons[i][0][2]
    outPROB_frt_05[i][0][3] = outPROB_frt_05_Atel[i][0][3]
    outPROB_frt_05[i][0][4] = outPROB_frt_05_PlEf[i][0][4]

with open('{}210826_0.5f_Card/testPROB_lat.txt'.format(PATH), 'rb') as f:
    outPROB_lat_05_Card = pickle.load(f)

with open('{}210826_0.5f_Edem/testPROB_lat.txt'.format(PATH), 'rb') as f:
    outPROB_lat_05_Edem = pickle.load(f)

with open('{}210826_0.5f_Cons/testPROB_lat.txt'.format(PATH), 'rb') as f:
    outPROB_lat_05_Cons = pickle.load(f)

with open('{}210826_0.5f_Atel/testPROB_lat.txt'.format(PATH), 'rb') as f:
    outPROB_lat_05_Atel = pickle.load(f)

with open('{}210826_0.5f_PlEf/testPROB_lat.txt'.format(PATH), 'rb') as f:
    outPROB_lat_05_PlEf = pickle.load(f)

outPROB_lat_05 = [[[0, 0, 0, 0, 0]] for i in range(len(outPROB_lat_base))]
for i in range(len(outPROB_lat_base)):
    outPROB_lat_05[i][0][0] = outPROB_lat_05_Card[i][0][0]
    outPROB_lat_05[i][0][1] = outPROB_lat_05_Edem[i][0][1]
    outPROB_lat_05[i][0][2] = outPROB_lat_05_Cons[i][0][2]
    outPROB_lat_05[i][0][3] = outPROB_lat_05_Atel[i][0][3]
    outPROB_lat_05[i][0][4] = outPROB_lat_05_PlEf[i][0][4]

with open('{}210813/testPROB_frt.txt'.format(PATH), 'rb') as f:
    outPROB_frt_sd = pickle.load(f)

with open('{}210813/testPROB_lat.txt'.format(PATH), 'rb') as f:
    outPROB_lat_sd = pickle.load(f)

'''
# Suppress Model Q (sd) to 0 or 1
for i in range(len(outPROB_frt_sd)):
    for j in range(5):
        if outPROB_frt_sd[i][0][j] < 0.5:
            outPROB_frt_sd[i][0][j] = 0
        elif outPROB_frt_sd[i][0][j] > 0.5:
            outPROB_frt_sd[i][0][j] = 1

for i in range(len(outPROB_lat_sd)):
    for j in range(5):
        if outPROB_lat_sd[i][0][j] < 0.5:
            outPROB_lat_sd[i][0][j] = 0
        elif outPROB_lat_sd[i][0][j] > 0.5:
            outPROB_lat_sd[i][0][j] = 1
'''


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

column_names = ['Path'] + class_names + ['Card_flag', 'Edem_flag', 'Cons_flag', 'Atel_flag', 'PlEf_flag']
df = pd.DataFrame(0, index = np.arange(len(test_frt) + len(test_lat)), columns = column_names)
test_frt_list = list(test_frt['Path'].copy())
test_lat_list = list(test_lat['Path'].copy())

for i in range(len(test_frt_list)):
    df.iloc[i, 0] = test_frt_list[i].split('/')[2] + '/' + test_frt_list[i].split('/')[3]
    for j in range(5):
        df.iloc[i, 6+j] = outPROB_frt_sd[i][0][j]

for i in range(len(test_lat_list)):
    df.iloc[len(test_frt_list) + i, 0] = test_lat_list[i].split('/')[2] + '/' + test_frt_list[i].split('/')[3]
    for j in range(5):
        df.iloc[len(test_frt_list) + i, 6+j] = outPROB_lat_sd[i][0][j]

for i in range(len(outPROB_frt_base)):
    for j in range(len(class_names)):
        df.iloc[i, j + 1] = outPROB_frt_base[i][0][j]
        
for i in range(len(outPROB_lat_base)):
    for j in range(len(class_names)):
        df.iloc[len(outPROB_frt_base) + i, j + 1] = outPROB_lat_base[i][0][j]

len(df[df.Card_flag >= 0.1]) # 0.5: 0, 0.4:  0, 0.3:  25, 0.2:  45, 0.1:  80
len(df[df.Edem_flag >= 0.1]) # 0.5: 0, 0.4: 17, 0.3:  75, 0.2: 111, 0.1: 154
len(df[df.Cons_flag >= 0.1]) # 0.5: 0, 0.4:  0, 0.3:   0, 0.2:   0, 0.1:  16
len(df[df.Atel_flag >= 0.1]) # 0.5: 3, 0.4: 55, 0.3: 110, 0.2: 139, 0.1: 196
len(df[df.PlEf_flag >= 0.1]) # 0.5: 0, 0.4:  1, 0.3:  21, 0.2:  84, 0.1: 147

# 0.5 0.4 0.3
df2 = df.query('Card_flag <= 0.5 & Edem_flag <= 0.5 & Cons_flag <= 0.5 & Atel_flag <= 0.5 & PlEf_flag <= 0.5').drop(['Card_flag', 'Edem_flag', 'Cons_flag', 'Atel_flag', 'PlEf_flag'], axis = 1)
test_agg = pd.read_csv(pathFileTest_agg)
test_agg['Path_study'] = test_agg.Path.apply(lambda x: x.split('/')[2] + '/' + x.split('/')[3])
test_agg2 = test_agg[test_agg['Path_study'].isin(list(df2['Path']))].reset_index(drop=True)
test_agg2.to_csv('./CheXpert-v1.0{0}/test_agg_part.csv'.format(img_type), index = False)
datasetTest_agg = CheXpertDataSet('./CheXpert-v1.0{0}/test_agg_part.csv'.format(img_type), nnClassCount, policy, transformSequence)
dataLoaderTest_agg = DataLoader(dataset = datasetTest_agg, num_workers = 2, pin_memory = True)

df_agg = df2.groupby('Path').agg({'Card' : 'min',
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