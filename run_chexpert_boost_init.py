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
#pathFileTrain_frt = './CheXpert-v1.0{0}/train_frt.csv'.format(img_type)
pathFileTrain_frt1_valid = './CheXpert-v1.0{0}/train_frt1_valid.csv'.format(img_type)
pathFileTrain_frt1_train = './CheXpert-v1.0{0}/train_frt1_train.csv'.format(img_type)
pathFileTrain_frt2_valid = './CheXpert-v1.0{0}/train_frt2_valid.csv'.format(img_type)
pathFileTrain_frt2_train = './CheXpert-v1.0{0}/train_frt2_train.csv'.format(img_type)
pathFileTrain_frt3_valid = './CheXpert-v1.0{0}/train_frt3_valid.csv'.format(img_type)
pathFileTrain_frt3_train = './CheXpert-v1.0{0}/train_frt3_train.csv'.format(img_type)
pathFileTrain_frt4_valid = './CheXpert-v1.0{0}/train_frt4_valid.csv'.format(img_type)
pathFileTrain_frt4_train = './CheXpert-v1.0{0}/train_frt4_train.csv'.format(img_type)
pathFileTrain_frt5_valid = './CheXpert-v1.0{0}/train_frt5_valid.csv'.format(img_type)
pathFileTrain_frt5_train = './CheXpert-v1.0{0}/train_frt5_train.csv'.format(img_type)

#pathFileTrain_lat = './CheXpert-v1.0{0}/train_lat.csv'.format(img_type)
pathFileTrain_lat1_valid = './CheXpert-v1.0{0}/train_lat1_valid.csv'.format(img_type)
pathFileTrain_lat1_train = './CheXpert-v1.0{0}/train_lat1_train.csv'.format(img_type)
pathFileTrain_lat2_valid = './CheXpert-v1.0{0}/train_lat2_valid.csv'.format(img_type)
pathFileTrain_lat2_train = './CheXpert-v1.0{0}/train_lat2_train.csv'.format(img_type)
pathFileTrain_lat3_valid = './CheXpert-v1.0{0}/train_lat3_valid.csv'.format(img_type)
pathFileTrain_lat3_train = './CheXpert-v1.0{0}/train_lat3_train.csv'.format(img_type)
pathFileTrain_lat4_valid = './CheXpert-v1.0{0}/train_lat4_valid.csv'.format(img_type)
pathFileTrain_lat4_train = './CheXpert-v1.0{0}/train_lat4_train.csv'.format(img_type)
pathFileTrain_lat5_valid = './CheXpert-v1.0{0}/train_lat5_valid.csv'.format(img_type)
pathFileTrain_lat5_train = './CheXpert-v1.0{0}/train_lat5_train.csv'.format(img_type)

# Validation (actually no meaning)
pathFileValid_frt = './CheXpert-v1.0{0}/valid_frt.csv'.format(img_type)
pathFileValid_lat = './CheXpert-v1.0{0}/valid_lat.csv'.format(img_type)

'''
pathFileTest_frt = './CheXpert-v1.0{0}/test_frt.csv'.format(img_type)
pathFileTest_lat = './CheXpert-v1.0{0}/test_lat.csv'.format(img_type)
pathFileTest_agg = './CheXpert-v1.0{0}/test_agg.csv'.format(img_type)

pathFileTrain_valid_agg1 = './CheXpert-v1.0{0}/train_valid_agg1.csv'.format(img_type)
pathFileTrain_valid_agg2 = './CheXpert-v1.0{0}/train_valid_agg2.csv'.format(img_type)
pathFileTrain_valid_agg3 = './CheXpert-v1.0{0}/train_valid_agg3.csv'.format(img_type)
pathFileTrain_valid_agg4 = './CheXpert-v1.0{0}/train_valid_agg4.csv'.format(img_type)
pathFileTrain_valid_agg5 = './CheXpert-v1.0{0}/train_valid_agg5.csv'.format(img_type)
'''

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
#datasetTrain_frt = CheXpertDataSet(pathFileTrain_frt, nnClassCount, cfg.policy, transformSequence)
datasetTrain_frt1_valid = CheXpertDataSet(pathFileTrain_frt1_valid, nnClassCount, cfg.policy, transformSequence)
datasetTrain_frt1_train = CheXpertDataSet(pathFileTrain_frt1_train, nnClassCount, cfg.policy, transformSequence)
datasetTrain_frt2_valid = CheXpertDataSet(pathFileTrain_frt2_valid, nnClassCount, cfg.policy, transformSequence)
datasetTrain_frt2_train = CheXpertDataSet(pathFileTrain_frt2_train, nnClassCount, cfg.policy, transformSequence)
datasetTrain_frt3_valid = CheXpertDataSet(pathFileTrain_frt3_valid, nnClassCount, cfg.policy, transformSequence)
datasetTrain_frt3_train = CheXpertDataSet(pathFileTrain_frt3_train, nnClassCount, cfg.policy, transformSequence)
datasetTrain_frt4_valid = CheXpertDataSet(pathFileTrain_frt4_valid, nnClassCount, cfg.policy, transformSequence)
datasetTrain_frt4_train = CheXpertDataSet(pathFileTrain_frt4_train, nnClassCount, cfg.policy, transformSequence)
datasetTrain_frt5_valid = CheXpertDataSet(pathFileTrain_frt5_valid, nnClassCount, cfg.policy, transformSequence)
datasetTrain_frt5_train = CheXpertDataSet(pathFileTrain_frt5_train, nnClassCount, cfg.policy, transformSequence)

#datasetTrain_lat = CheXpertDataSet(pathFileTrain_lat, nnClassCount, cfg.policy, transformSequence)
datasetTrain_lat1_valid = CheXpertDataSet(pathFileTrain_lat1_valid, nnClassCount, cfg.policy, transformSequence)
datasetTrain_lat1_train = CheXpertDataSet(pathFileTrain_lat1_train, nnClassCount, cfg.policy, transformSequence)
datasetTrain_lat2_valid = CheXpertDataSet(pathFileTrain_lat2_valid, nnClassCount, cfg.policy, transformSequence)
datasetTrain_lat2_train = CheXpertDataSet(pathFileTrain_lat2_train, nnClassCount, cfg.policy, transformSequence)
datasetTrain_lat3_valid = CheXpertDataSet(pathFileTrain_lat3_valid, nnClassCount, cfg.policy, transformSequence)
datasetTrain_lat3_train = CheXpertDataSet(pathFileTrain_lat3_train, nnClassCount, cfg.policy, transformSequence)
datasetTrain_lat4_valid = CheXpertDataSet(pathFileTrain_lat4_valid, nnClassCount, cfg.policy, transformSequence)
datasetTrain_lat4_train = CheXpertDataSet(pathFileTrain_lat4_train, nnClassCount, cfg.policy, transformSequence)
datasetTrain_lat5_valid = CheXpertDataSet(pathFileTrain_lat5_valid, nnClassCount, cfg.policy, transformSequence)
datasetTrain_lat5_train = CheXpertDataSet(pathFileTrain_lat5_train, nnClassCount, cfg.policy, transformSequence)

datasetValid_frt = CheXpertDataSet(pathFileValid_frt, nnClassCount, cfg.policy, transformSequence)
datasetValid_lat = CheXpertDataSet(pathFileValid_lat, nnClassCount, cfg.policy, transformSequence)

'''
datasetTest_frt = CheXpertDataSet(pathFileTest_frt, nnClassCount, cfg.policy, transformSequence)
datasetTest_lat = CheXpertDataSet(pathFileTest_lat, nnClassCount, cfg.policy, transformSequence)
datasetTest_agg = CheXpertDataSet(pathFileTest_agg, nnClassCount, cfg.policy, transformSequence)

datasetTrain_valid_agg1 = CheXpertDataSet(pathFileTrain_valid_agg1, nnClassCount, cfg.policy, transformSequence)
datasetTrain_valid_agg2 = CheXpertDataSet(pathFileTrain_valid_agg2, nnClassCount, cfg.policy, transformSequence)
datasetTrain_valid_agg3 = CheXpertDataSet(pathFileTrain_valid_agg3, nnClassCount, cfg.policy, transformSequence)
datasetTrain_valid_agg4 = CheXpertDataSet(pathFileTrain_valid_agg4, nnClassCount, cfg.policy, transformSequence)
datasetTrain_valid_agg5 = CheXpertDataSet(pathFileTrain_valid_agg5, nnClassCount, cfg.policy, transformSequence)
'''
# Use subset of datasetTrain for training
'''train_num_frt = round(len(datasetTrain_frt) * cfg.train_ratio) # use subset of original training dataset
train_num_lat = round(len(datasetTrain_lat) * cfg.train_ratio) # use subset of original training dataset
datasetTrain_frt, _ = random_split(datasetTrain_frt, [train_num_frt, len(datasetTrain_frt) - train_num_frt])
datasetTrain_lat, _ = random_split(datasetTrain_lat, [train_num_lat, len(datasetTrain_lat) - train_num_lat])'''



########################
## Cross Validation 1 ##
########################
print('<<< Data Information (1) >>>')
print('Train data (frontal 1):', len(datasetTrain_frt1_train))
print('Train data (lateral 1):', len(datasetTrain_lat1_train))
print('Valid data (frontal 1):', len(datasetValid_frt))
print('Valid data (lateral 1):', len(datasetValid_lat))
print('Test data (frontal 1):', len(datasetTrain_frt1_valid))
print('Test data (lateral 1):', len(datasetTrain_lat1_valid))
#print('Test data (study):', len(datasetTest_agg), '\n')

# Create DataLoaders
dataLoaderTrain_frt1_valid = DataLoader(dataset = datasetTrain_frt1_valid, batch_size = trBatchSize, 
                                        shuffle = False, num_workers = 2, pin_memory = True)
dataLoaderTrain_frt1_train = DataLoader(dataset = datasetTrain_frt1_train, batch_size = trBatchSize, 
                                        shuffle = True, num_workers = 2, pin_memory = True)
dataLoaderVal_frt = DataLoader(dataset = datasetValid_frt, batch_size = trBatchSize, 
                               shuffle = False, num_workers = 2, pin_memory = True)

dataLoaderTrain_lat1_valid = DataLoader(dataset = datasetTrain_lat1_valid, batch_size = trBatchSize, 
                                        shuffle = False, num_workers = 2, pin_memory = True)
dataLoaderTrain_lat1_train = DataLoader(dataset = datasetTrain_lat1_train, batch_size = trBatchSize, 
                                        shuffle = True, num_workers = 2, pin_memory = True)
dataLoaderVal_lat = DataLoader(dataset = datasetValid_lat, batch_size = trBatchSize, 
                               shuffle = False, num_workers = 2, pin_memory = True)


# Initialize and load the model
PATH = args.output_path + 'boost1/'
if args.output_path[-1] != '/':
    PATH = PATH + '/'
else:
    PATH = PATH
    
if not os.path.exists(PATH): os.makedirs(PATH)

'''See 'materials.py' to check the class 'DenseNet121'.'''
model = DenseNet121(nnClassCount, nnIsTrained).cuda()
model = torch.nn.DataParallel(model).cuda()

# Train frontal model
'''See 'materials.py' to check the class 'CheXpertTrainer'.'''
model_num_frt, model_num_frt_each, train_time_frt = CheXpertTrainer.train(model, dataLoaderTrain_frt1_train, dataLoaderVal_frt, class_names,
                                                                          nnClassCount, trMaxEpoch, PATH, 'frt', checkpoint = None, cfg = cfg)

'''See 'materials.py' to check the class 'DenseNet121'.'''
model = DenseNet121(nnClassCount, nnIsTrained).cuda()
model = torch.nn.DataParallel(model).cuda()

# Train lateral model
'''See 'materials.py' to check the class 'CheXpertTrainer'.'''
model_num_lat, model_num_lat_each, train_time_lat = CheXpertTrainer.train(model, dataLoaderTrain_lat1_train, dataLoaderVal_lat, class_names,
                                                                          nnClassCount, trMaxEpoch, PATH, 'lat', checkpoint = None, cfg = cfg)

print('<<< Model 1 Trained >>>')
print('For frontal model 1,', 'm-epoch_{0}_frt.pth.tar'.format(model_num_frt), 'is the best overall.')
for i in range(5):
    print('For frontal {0},'.format(class_names[i]), 'm-epoch_{0}_frt.pth.tar'.format(model_num_frt_each[i]), 'is the best.')
print('')
print('For lateral model 1,', 'm-epoch_{0}_lat.pth.tar'.format(model_num_lat), 'is the best overall.')
for i in range(5):
    print('For lateral {0},'.format(class_names[i]), 'm-epoch_{0}_lat.pth.tar'.format(model_num_lat_each[i]), 'is the best.')
print('')


checkpoint_frt = PATH + 'm-epoch_{0}_frt.pth.tar'.format(trMaxEpoch) # Use the last model ('model_num_frt' if use valid set for model decision)
checkpoint_lat = PATH + 'm-epoch_{0}_lat.pth.tar'.format(trMaxEpoch) # Use the last model ('model_num_lat' if use valid set for model decision)
'''See 'materials.py' to check the class 'CheXpertTrainer'.'''
outGT_frt, outPRED_frt, outPROB_frt, aurocMean_frt, aurocIndividual_frt = CheXpertTrainer.test(model, dataLoaderTrain_frt1_valid, nnClassCount, checkpoint_frt, class_names, 'frt')
outGT_lat, outPRED_lat, outPROB_lat, aurocMean_lat, aurocIndividual_lat = CheXpertTrainer.test(model, dataLoaderTrain_lat1_valid, nnClassCount, checkpoint_lat, class_names, 'lat')

# Save the test outPROB_frt
with open('{}testPROB_frt1.txt'.format(PATH), 'wb') as fp:
    pickle.dump(outPROB_frt, fp)

# Save the test outPROB_lat
with open('{}testPROB_lat1.txt'.format(PATH), 'wb') as fp:
    pickle.dump(outPROB_lat, fp)



########################
## Cross Validation 2 ##
########################
print('<<< Data Information (2) >>>')
print('Train data (frontal 2):', len(datasetTrain_frt2_train))
print('Train data (lateral 2):', len(datasetTrain_lat2_train))
print('Valid data (frontal 2):', len(datasetValid_frt))
print('Valid data (lateral 2):', len(datasetValid_lat))
print('Test data (frontal 2):', len(datasetTrain_frt2_valid))
print('Test data (lateral 2):', len(datasetTrain_lat2_valid))
#print('Test data (study):', len(datasetTest_agg), '\n')

# Create DataLoaders
dataLoaderTrain_frt2_valid = DataLoader(dataset = datasetTrain_frt2_valid, batch_size = trBatchSize, 
                                        shuffle = False, num_workers = 2, pin_memory = True)
dataLoaderTrain_frt2_train = DataLoader(dataset = datasetTrain_frt2_train, batch_size = trBatchSize, 
                                        shuffle = True, num_workers = 2, pin_memory = True)
dataLoaderVal_frt = DataLoader(dataset = datasetValid_frt, batch_size = trBatchSize, 
                               shuffle = False, num_workers = 2, pin_memory = True)

dataLoaderTrain_lat2_valid = DataLoader(dataset = datasetTrain_lat2_valid, batch_size = trBatchSize, 
                                        shuffle = False, num_workers = 2, pin_memory = True)
dataLoaderTrain_lat2_train = DataLoader(dataset = datasetTrain_lat2_train, batch_size = trBatchSize, 
                                        shuffle = True, num_workers = 2, pin_memory = True)
dataLoaderVal_lat = DataLoader(dataset = datasetValid_lat, batch_size = trBatchSize, 
                               shuffle = False, num_workers = 2, pin_memory = True)


# Initialize and load the model
PATH = args.output_path + 'boost2/'
if args.output_path[-1] != '/':
    PATH = PATH + '/'
else:
    PATH = PATH

if not os.path.exists(PATH): os.makedirs(PATH)

'''See 'materials.py' to check the class 'DenseNet121'.'''
model = DenseNet121(nnClassCount, nnIsTrained).cuda()
model = torch.nn.DataParallel(model).cuda()

# Train frontal model
'''See 'materials.py' to check the class 'CheXpertTrainer'.'''
model_num_frt, model_num_frt_each, train_time_frt = CheXpertTrainer.train(model, dataLoaderTrain_frt2_train, dataLoaderVal_frt, class_names,
                                                                          nnClassCount, trMaxEpoch, PATH, 'frt', checkpoint = None, cfg = cfg)

'''See 'materials.py' to check the class 'DenseNet121'.'''
model = DenseNet121(nnClassCount, nnIsTrained).cuda()
model = torch.nn.DataParallel(model).cuda()

# Train lateral model
'''See 'materials.py' to check the class 'CheXpertTrainer'.'''
model_num_lat, model_num_lat_each, train_time_lat = CheXpertTrainer.train(model, dataLoaderTrain_lat2_train, dataLoaderVal_lat, class_names,
                                                                          nnClassCount, trMaxEpoch, PATH, 'lat', checkpoint = None, cfg = cfg)

print('<<< Model 2 Trained >>>')
print('For frontal model 2,', 'm-epoch_{0}_frt.pth.tar'.format(model_num_frt), 'is the best overall.')
for i in range(5):
    print('For frontal {0},'.format(class_names[i]), 'm-epoch_{0}_frt.pth.tar'.format(model_num_frt_each[i]), 'is the best.')
print('')
print('For lateral model 2,', 'm-epoch_{0}_lat.pth.tar'.format(model_num_lat), 'is the best overall.')
for i in range(5):
    print('For lateral {0},'.format(class_names[i]), 'm-epoch_{0}_lat.pth.tar'.format(model_num_lat_each[i]), 'is the best.')
print('')


checkpoint_frt = PATH + 'm-epoch_{0}_frt.pth.tar'.format(trMaxEpoch) # Use the last model ('model_num_frt' if use valid set for model decision)
checkpoint_lat = PATH + 'm-epoch_{0}_lat.pth.tar'.format(trMaxEpoch) # Use the last model ('model_num_lat' if use valid set for model decision)
'''See 'materials.py' to check the class 'CheXpertTrainer'.'''
outGT_frt, outPRED_frt, outPROB_frt, aurocMean_frt, aurocIndividual_frt = CheXpertTrainer.test(model, dataLoaderTrain_frt2_valid, nnClassCount, checkpoint_frt, class_names, 'frt')
outGT_lat, outPRED_lat, outPROB_lat, aurocMean_lat, aurocIndividual_lat = CheXpertTrainer.test(model, dataLoaderTrain_lat2_valid, nnClassCount, checkpoint_lat, class_names, 'lat')

# Save the test outPROB_frt
with open('{}testPROB_frt2.txt'.format(PATH), 'wb') as fp:
    pickle.dump(outPROB_frt, fp)

# Save the test outPROB_lat
with open('{}testPROB_lat2.txt'.format(PATH), 'wb') as fp:
    pickle.dump(outPROB_lat, fp)



########################
## Cross Validation 3 ##
########################
print('<<< Data Information (3) >>>')
print('Train data (frontal 3):', len(datasetTrain_frt3_train))
print('Train data (lateral 3):', len(datasetTrain_lat3_train))
print('Valid data (frontal 3):', len(datasetValid_frt))
print('Valid data (lateral 3):', len(datasetValid_lat))
print('Test data (frontal 3):', len(datasetTrain_frt3_valid))
print('Test data (lateral 3):', len(datasetTrain_lat3_valid))
#print('Test data (study):', len(datasetTest_agg), '\n')

# Create DataLoaders
dataLoaderTrain_frt3_valid = DataLoader(dataset = datasetTrain_frt3_valid, batch_size = trBatchSize, 
                                        shuffle = False, num_workers = 2, pin_memory = True)
dataLoaderTrain_frt3_train = DataLoader(dataset = datasetTrain_frt3_train, batch_size = trBatchSize, 
                                        shuffle = True, num_workers = 2, pin_memory = True)
dataLoaderVal_frt = DataLoader(dataset = datasetValid_frt, batch_size = trBatchSize, 
                               shuffle = False, num_workers = 2, pin_memory = True)

dataLoaderTrain_lat3_valid = DataLoader(dataset = datasetTrain_lat3_valid, batch_size = trBatchSize, 
                                        shuffle = False, num_workers = 2, pin_memory = True)
dataLoaderTrain_lat3_train = DataLoader(dataset = datasetTrain_lat3_train, batch_size = trBatchSize, 
                                        shuffle = True, num_workers = 2, pin_memory = True)
dataLoaderVal_lat = DataLoader(dataset = datasetValid_lat, batch_size = trBatchSize, 
                               shuffle = False, num_workers = 2, pin_memory = True)


# Initialize and load the model
PATH = args.output_path + 'boost3/'
if args.output_path[-1] != '/':
    PATH = PATH + '/'
else:
    PATH = PATH

if not os.path.exists(PATH): os.makedirs(PATH)

'''See 'materials.py' to check the class 'DenseNet121'.'''
model = DenseNet121(nnClassCount, nnIsTrained).cuda()
model = torch.nn.DataParallel(model).cuda()

# Train frontal model
'''See 'materials.py' to check the class 'CheXpertTrainer'.'''
model_num_frt, model_num_frt_each, train_time_frt = CheXpertTrainer.train(model, dataLoaderTrain_frt3_train, dataLoaderVal_frt, class_names,
                                                                          nnClassCount, trMaxEpoch, PATH, 'frt', checkpoint = None, cfg = cfg)

'''See 'materials.py' to check the class 'DenseNet121'.'''
model = DenseNet121(nnClassCount, nnIsTrained).cuda()
model = torch.nn.DataParallel(model).cuda()

# Train lateral model
'''See 'materials.py' to check the class 'CheXpertTrainer'.'''
model_num_lat, model_num_lat_each, train_time_lat = CheXpertTrainer.train(model, dataLoaderTrain_lat3_train, dataLoaderVal_lat, class_names,
                                                                          nnClassCount, trMaxEpoch, PATH, 'lat', checkpoint = None, cfg = cfg)

print('<<< Model 3 Trained >>>')
print('For frontal model 3,', 'm-epoch_{0}_frt.pth.tar'.format(model_num_frt), 'is the best overall.')
for i in range(5):
    print('For frontal {0},'.format(class_names[i]), 'm-epoch_{0}_frt.pth.tar'.format(model_num_frt_each[i]), 'is the best.')
print('')
print('For lateral model 3,', 'm-epoch_{0}_lat.pth.tar'.format(model_num_lat), 'is the best overall.')
for i in range(5):
    print('For lateral {0},'.format(class_names[i]), 'm-epoch_{0}_lat.pth.tar'.format(model_num_lat_each[i]), 'is the best.')
print('')


checkpoint_frt = PATH + 'm-epoch_{0}_frt.pth.tar'.format(trMaxEpoch) # Use the last model ('model_num_frt' if use valid set for model decision)
checkpoint_lat = PATH + 'm-epoch_{0}_lat.pth.tar'.format(trMaxEpoch) # Use the last model ('model_num_lat' if use valid set for model decision)
'''See 'materials.py' to check the class 'CheXpertTrainer'.'''
outGT_frt, outPRED_frt, outPROB_frt, aurocMean_frt, aurocIndividual_frt = CheXpertTrainer.test(model, dataLoaderTrain_frt3_valid, nnClassCount, checkpoint_frt, class_names, 'frt')
outGT_lat, outPRED_lat, outPROB_lat, aurocMean_lat, aurocIndividual_lat = CheXpertTrainer.test(model, dataLoaderTrain_lat3_valid, nnClassCount, checkpoint_lat, class_names, 'lat')

# Save the test outPROB_frt
with open('{}testPROB_frt3.txt'.format(PATH), 'wb') as fp:
    pickle.dump(outPROB_frt, fp)

# Save the test outPROB_lat
with open('{}testPROB_lat3.txt'.format(PATH), 'wb') as fp:
    pickle.dump(outPROB_lat, fp)



########################
## Cross Validation 4 ##
########################
print('<<< Data Information (4) >>>')
print('Train data (frontal 4):', len(datasetTrain_frt4_train))
print('Train data (lateral 4):', len(datasetTrain_lat4_train))
print('Valid data (frontal 4):', len(datasetValid_frt))
print('Valid data (lateral 4):', len(datasetValid_lat))
print('Test data (frontal 4):', len(datasetTrain_frt4_valid))
print('Test data (lateral 4):', len(datasetTrain_lat4_valid))
#print('Test data (study):', len(datasetTest_agg), '\n')

# Create DataLoaders
dataLoaderTrain_frt4_valid = DataLoader(dataset = datasetTrain_frt4_valid, batch_size = trBatchSize, 
                                        shuffle = False, num_workers = 2, pin_memory = True)
dataLoaderTrain_frt4_train = DataLoader(dataset = datasetTrain_frt4_train, batch_size = trBatchSize, 
                                        shuffle = True, num_workers = 2, pin_memory = True)
dataLoaderVal_frt = DataLoader(dataset = datasetValid_frt, batch_size = trBatchSize, 
                               shuffle = False, num_workers = 2, pin_memory = True)

dataLoaderTrain_lat4_valid = DataLoader(dataset = datasetTrain_lat4_valid, batch_size = trBatchSize, 
                                        shuffle = False, num_workers = 2, pin_memory = True)
dataLoaderTrain_lat4_train = DataLoader(dataset = datasetTrain_lat4_train, batch_size = trBatchSize, 
                                        shuffle = True, num_workers = 2, pin_memory = True)
dataLoaderVal_lat = DataLoader(dataset = datasetValid_lat, batch_size = trBatchSize, 
                               shuffle = False, num_workers = 2, pin_memory = True)


# Initialize and load the model
PATH = args.output_path + 'boost4/'
if args.output_path[-1] != '/':
    PATH = PATH + '/'
else:
    PATH = PATH

if not os.path.exists(PATH): os.makedirs(PATH)

'''See 'materials.py' to check the class 'DenseNet121'.'''
model = DenseNet121(nnClassCount, nnIsTrained).cuda()
model = torch.nn.DataParallel(model).cuda()

# Train frontal model
'''See 'materials.py' to check the class 'CheXpertTrainer'.'''
model_num_frt, model_num_frt_each, train_time_frt = CheXpertTrainer.train(model, dataLoaderTrain_frt4_train, dataLoaderVal_frt, class_names,
                                                                          nnClassCount, trMaxEpoch, PATH, 'frt', checkpoint = None, cfg = cfg)

'''See 'materials.py' to check the class 'DenseNet121'.'''
model = DenseNet121(nnClassCount, nnIsTrained).cuda()
model = torch.nn.DataParallel(model).cuda()

# Train lateral model
'''See 'materials.py' to check the class 'CheXpertTrainer'.'''
model_num_lat, model_num_lat_each, train_time_lat = CheXpertTrainer.train(model, dataLoaderTrain_lat4_train, dataLoaderVal_lat, class_names,
                                                                          nnClassCount, trMaxEpoch, PATH, 'lat', checkpoint = None, cfg = cfg)

print('<<< Model 4 Trained >>>')
print('For frontal model 4,', 'm-epoch_{0}_frt.pth.tar'.format(model_num_frt), 'is the best overall.')
for i in range(5):
    print('For frontal {0},'.format(class_names[i]), 'm-epoch_{0}_frt.pth.tar'.format(model_num_frt_each[i]), 'is the best.')
print('')
print('For lateral model 4,', 'm-epoch_{0}_lat.pth.tar'.format(model_num_lat), 'is the best overall.')
for i in range(5):
    print('For lateral {0},'.format(class_names[i]), 'm-epoch_{0}_lat.pth.tar'.format(model_num_lat_each[i]), 'is the best.')
print('')


checkpoint_frt = PATH + 'm-epoch_{0}_frt.pth.tar'.format(trMaxEpoch) # Use the last model ('model_num_frt' if use valid set for model decision)
checkpoint_lat = PATH + 'm-epoch_{0}_lat.pth.tar'.format(trMaxEpoch) # Use the last model ('model_num_lat' if use valid set for model decision)
'''See 'materials.py' to check the class 'CheXpertTrainer'.'''
outGT_frt, outPRED_frt, outPROB_frt, aurocMean_frt, aurocIndividual_frt = CheXpertTrainer.test(model, dataLoaderTrain_frt4_valid, nnClassCount, checkpoint_frt, class_names, 'frt')
outGT_lat, outPRED_lat, outPROB_lat, aurocMean_lat, aurocIndividual_lat = CheXpertTrainer.test(model, dataLoaderTrain_lat4_valid, nnClassCount, checkpoint_lat, class_names, 'lat')

# Save the test outPROB_frt
with open('{}testPROB_frt4.txt'.format(PATH), 'wb') as fp:
    pickle.dump(outPROB_frt, fp)

# Save the test outPROB_lat
with open('{}testPROB_lat4.txt'.format(PATH), 'wb') as fp:
    pickle.dump(outPROB_lat, fp)



########################
## Cross Validation 5 ##
########################
print('<<< Data Information (5) >>>')
print('Train data (frontal 5):', len(datasetTrain_frt5_train))
print('Train data (lateral 5):', len(datasetTrain_lat5_train))
print('Valid data (frontal 5):', len(datasetValid_frt))
print('Valid data (lateral 5):', len(datasetValid_lat))
print('Test data (frontal 5):', len(datasetTrain_frt5_valid))
print('Test data (lateral 5):', len(datasetTrain_lat5_valid))
#print('Test data (study):', len(datasetTest_agg), '\n')

# Create DataLoaders
dataLoaderTrain_frt5_valid = DataLoader(dataset = datasetTrain_frt5_valid, batch_size = trBatchSize, 
                                        shuffle = False, num_workers = 2, pin_memory = True)
dataLoaderTrain_frt5_train = DataLoader(dataset = datasetTrain_frt5_train, batch_size = trBatchSize, 
                                        shuffle = True, num_workers = 2, pin_memory = True)
dataLoaderVal_frt = DataLoader(dataset = datasetValid_frt, batch_size = trBatchSize, 
                               shuffle = False, num_workers = 2, pin_memory = True)

dataLoaderTrain_lat5_valid = DataLoader(dataset = datasetTrain_lat5_valid, batch_size = trBatchSize, 
                                        shuffle = False, num_workers = 2, pin_memory = True)
dataLoaderTrain_lat5_train = DataLoader(dataset = datasetTrain_lat5_train, batch_size = trBatchSize, 
                                        shuffle = True, num_workers = 2, pin_memory = True)
dataLoaderVal_lat = DataLoader(dataset = datasetValid_lat, batch_size = trBatchSize, 
                               shuffle = False, num_workers = 2, pin_memory = True)


# Initialize and load the model
PATH = args.output_path + 'boost5/'
if args.output_path[-1] != '/':
    PATH = PATH + '/'
else:
    PATH = PATH

if not os.path.exists(PATH): os.makedirs(PATH)

'''See 'materials.py' to check the class 'DenseNet121'.'''
model = DenseNet121(nnClassCount, nnIsTrained).cuda()
model = torch.nn.DataParallel(model).cuda()

# Train frontal model
'''See 'materials.py' to check the class 'CheXpertTrainer'.'''
model_num_frt, model_num_frt_each, train_time_frt = CheXpertTrainer.train(model, dataLoaderTrain_frt5_train, dataLoaderVal_frt, class_names,
                                                                          nnClassCount, trMaxEpoch, PATH, 'frt', checkpoint = None, cfg = cfg)

'''See 'materials.py' to check the class 'DenseNet121'.'''
model = DenseNet121(nnClassCount, nnIsTrained).cuda()
model = torch.nn.DataParallel(model).cuda()

# Train lateral model
'''See 'materials.py' to check the class 'CheXpertTrainer'.'''
model_num_lat, model_num_lat_each, train_time_lat = CheXpertTrainer.train(model, dataLoaderTrain_lat5_train, dataLoaderVal_lat, class_names,
                                                                          nnClassCount, trMaxEpoch, PATH, 'lat', checkpoint = None, cfg = cfg)

print('<<< Model 5 Trained >>>')
print('For frontal model 5,', 'm-epoch_{0}_frt.pth.tar'.format(model_num_frt), 'is the best overall.')
for i in range(5):
    print('For frontal {0},'.format(class_names[i]), 'm-epoch_{0}_frt.pth.tar'.format(model_num_frt_each[i]), 'is the best.')
print('')
print('For lateral model 5,', 'm-epoch_{0}_lat.pth.tar'.format(model_num_lat), 'is the best overall.')
for i in range(5):
    print('For lateral {0},'.format(class_names[i]), 'm-epoch_{0}_lat.pth.tar'.format(model_num_lat_each[i]), 'is the best.')
print('')


checkpoint_frt = PATH + 'm-epoch_{0}_frt.pth.tar'.format(trMaxEpoch) # Use the last model ('model_num_frt' if use valid set for model decision)
checkpoint_lat = PATH + 'm-epoch_{0}_lat.pth.tar'.format(trMaxEpoch) # Use the last model ('model_num_lat' if use valid set for model decision)
'''See 'materials.py' to check the class 'CheXpertTrainer'.'''
outGT_frt, outPRED_frt, outPROB_frt, aurocMean_frt, aurocIndividual_frt = CheXpertTrainer.test(model, dataLoaderTrain_frt5_valid, nnClassCount, checkpoint_frt, class_names, 'frt')
outGT_lat, outPRED_lat, outPROB_lat, aurocMean_lat, aurocIndividual_lat = CheXpertTrainer.test(model, dataLoaderTrain_lat5_valid, nnClassCount, checkpoint_lat, class_names, 'lat')

# Save the test outPROB_frt
with open('{}testPROB_frt5.txt'.format(PATH), 'wb') as fp:
    pickle.dump(outPROB_frt, fp)

# Save the test outPROB_lat
with open('{}testPROB_lat5.txt'.format(PATH), 'wb') as fp:
    pickle.dump(outPROB_lat, fp)