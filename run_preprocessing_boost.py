'''Data Preprocessing'''

###################
## Prerequisites ##
###################
import json
import argparse
import pandas as pd
from easydict import EasyDict as edict

parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('cfg_path', metavar='CFG_PATH', type = str, help = 'Path to the config file in yaml format.')
args = parser.parse_args()
with open(args.cfg_path) as f:
    cfg = edict(json.load(f))



###################
## Preprocessing ##
###################
# Each file contains pairs (path to image, output vector)
if cfg.image_type == 'small':
    img_type = '-small'
else:
    img_type = ''

Traindata = pd.read_csv('./CheXpert-v1.0{0}/train.csv'.format(img_type))

Traindata_frt = Traindata[Traindata['Path'].str.contains('frontal')].sort_values('Path').reset_index(drop=True).copy()
Traindata_frt1_valid = Traindata_frt[:38206].copy()
Traindata_frt1_train = Traindata_frt[38206:].copy()
Traindata_frt2_valid = Traindata_frt[38206:(38206*2)].copy()
Traindata_frt2_train = pd.concat([Traindata_frt[:38206].copy(), Traindata_frt[(38206*2):].copy()])
Traindata_frt3_valid = Traindata_frt[(38206*2):(38206*3-1)].copy()
Traindata_frt3_train = pd.concat([Traindata_frt[:(38206*2)].copy(), Traindata_frt[(38206*3-1):].copy()])
Traindata_frt4_valid = Traindata_frt[(38206*3-1):(38206*4-2)].copy()
Traindata_frt4_train = pd.concat([Traindata_frt[:(38206*3-1)].copy(), Traindata_frt[(38206*4-2):].copy()])
Traindata_frt5_valid = Traindata_frt[(38206*4-2):(38206*5-3)].copy()
Traindata_frt5_train = Traindata_frt[:(38206*4-2)].copy()

Traindata_lat = Traindata[Traindata['Path'].str.contains('lateral')].sort_values('Path').reset_index(drop=True).copy()
Traindata_lat1_valid = Traindata_lat[:6478].copy()
Traindata_lat1_train = Traindata_lat[6478:].copy()
Traindata_lat2_valid = Traindata_lat[6478:(6478*2)].copy()
Traindata_lat2_train = pd.concat([Traindata_lat[:6478].copy(), Traindata_lat[(6478*2):].copy()])
Traindata_lat3_valid = Traindata_lat[(6478*2):(6478*3-1)].copy()
Traindata_lat3_train = pd.concat([Traindata_lat[:(6478*2)].copy(), Traindata_lat[(6478*3-1):].copy()])
Traindata_lat4_valid = Traindata_lat[(6478*3-1):(6478*4-2)].copy()
Traindata_lat4_train = pd.concat([Traindata_lat[:(6478*3-1)].copy(), Traindata_lat[(6478*4-2):].copy()])
Traindata_lat5_valid = Traindata_lat[(6478*4-2):(6478*5-3)].copy()
Traindata_lat5_train = Traindata_lat[:(6478*4-2)].copy()

Traindata_frt.to_csv('./CheXpert-v1.0{0}/train_frt.csv'.format(img_type), index = False)
Traindata_lat.to_csv('./CheXpert-v1.0{0}/train_lat.csv'.format(img_type), index = False)

Traindata_frt1_valid.to_csv('./CheXpert-v1.0{0}/train_frt1_valid.csv'.format(img_type), index = False)
Traindata_frt1_train.to_csv('./CheXpert-v1.0{0}/train_frt1_train.csv'.format(img_type), index = False)
Traindata_frt2_valid.to_csv('./CheXpert-v1.0{0}/train_frt2_valid.csv'.format(img_type), index = False)
Traindata_frt2_train.to_csv('./CheXpert-v1.0{0}/train_frt2_train.csv'.format(img_type), index = False)
Traindata_frt3_valid.to_csv('./CheXpert-v1.0{0}/train_frt3_valid.csv'.format(img_type), index = False)
Traindata_frt3_train.to_csv('./CheXpert-v1.0{0}/train_frt3_train.csv'.format(img_type), index = False)
Traindata_frt4_valid.to_csv('./CheXpert-v1.0{0}/train_frt4_valid.csv'.format(img_type), index = False)
Traindata_frt4_train.to_csv('./CheXpert-v1.0{0}/train_frt4_train.csv'.format(img_type), index = False)
Traindata_frt5_valid.to_csv('./CheXpert-v1.0{0}/train_frt5_valid.csv'.format(img_type), index = False)
Traindata_frt5_train.to_csv('./CheXpert-v1.0{0}/train_frt5_train.csv'.format(img_type), index = False)

Traindata_lat1_valid.to_csv('./CheXpert-v1.0{0}/train_lat1_valid.csv'.format(img_type), index = False)
Traindata_lat1_train.to_csv('./CheXpert-v1.0{0}/train_lat1_train.csv'.format(img_type), index = False)
Traindata_lat2_valid.to_csv('./CheXpert-v1.0{0}/train_lat2_valid.csv'.format(img_type), index = False)
Traindata_lat2_train.to_csv('./CheXpert-v1.0{0}/train_lat2_train.csv'.format(img_type), index = False)
Traindata_lat3_valid.to_csv('./CheXpert-v1.0{0}/train_lat3_valid.csv'.format(img_type), index = False)
Traindata_lat3_train.to_csv('./CheXpert-v1.0{0}/train_lat3_train.csv'.format(img_type), index = False)
Traindata_lat4_valid.to_csv('./CheXpert-v1.0{0}/train_lat4_valid.csv'.format(img_type), index = False)
Traindata_lat4_train.to_csv('./CheXpert-v1.0{0}/train_lat4_train.csv'.format(img_type), index = False)
Traindata_lat5_valid.to_csv('./CheXpert-v1.0{0}/train_lat5_valid.csv'.format(img_type), index = False)
Traindata_lat5_train.to_csv('./CheXpert-v1.0{0}/train_lat5_train.csv'.format(img_type), index = False)

print('Train data length(frontal):', len(Traindata_frt))

print('Valid data from Train data length(frontal1):', len(Traindata_frt1_valid))
print('Valid data from Train data length(frontal2):', len(Traindata_frt2_valid))
print('Valid data from Train data length(frontal3):', len(Traindata_frt3_valid))
print('Valid data from Train data length(frontal4):', len(Traindata_frt4_valid))
print('Valid data from Train data length(frontal5):', len(Traindata_frt5_valid))

print('Train data length(lateral):', len(Traindata_lat))

print('Valid data from Train data length(lateral1):', len(Traindata_lat1_valid))
print('Valid data from Train data length(lateral2):', len(Traindata_lat2_valid))
print('Valid data from Train data length(lateral3):', len(Traindata_lat3_valid))
print('Valid data from Train data length(lateral4):', len(Traindata_lat4_valid))
print('Valid data from Train data length(lateral5):', len(Traindata_lat5_valid))

print('Train data length(total):', len(Traindata_frt) + len(Traindata_lat))


Traindata_frt1_valid.loc[:, 'Study'] = Traindata_frt1_valid.Path.str.split('/').str[2] + '/' + Traindata_frt1_valid.Path.str.split('/').str[3]
Traindata_frt1_valid_agg = Traindata_frt1_valid.groupby('Study').agg('first').reset_index()
Traindata_frt1_valid_agg = Traindata_frt1_valid_agg.sort_values('Path')
Traindata_frt1_valid_agg = Traindata_frt1_valid_agg.drop('Study', axis = 1)
Traindata_frt1_valid_agg.to_csv('./CheXpert-v1.0{0}/train_valid_agg1.csv'.format(img_type), index = False)
print('Valid 1 data length(study):', len(Traindata_frt1_valid_agg))

Traindata_frt2_valid.loc[:, 'Study'] = Traindata_frt2_valid.Path.str.split('/').str[2] + '/' + Traindata_frt2_valid.Path.str.split('/').str[3]
Traindata_frt2_valid_agg = Traindata_frt2_valid.groupby('Study').agg('first').reset_index()
Traindata_frt2_valid_agg = Traindata_frt2_valid_agg.sort_values('Path')
Traindata_frt2_valid_agg = Traindata_frt2_valid_agg.drop('Study', axis = 1)
Traindata_frt2_valid_agg.to_csv('./CheXpert-v1.0{0}/train_valid_agg2.csv'.format(img_type), index = False)
print('Valid 2 data length(study):', len(Traindata_frt2_valid_agg))

Traindata_frt3_valid.loc[:, 'Study'] = Traindata_frt3_valid.Path.str.split('/').str[2] + '/' + Traindata_frt3_valid.Path.str.split('/').str[3]
Traindata_frt3_valid_agg = Traindata_frt3_valid.groupby('Study').agg('first').reset_index()
Traindata_frt3_valid_agg = Traindata_frt3_valid_agg.sort_values('Path')
Traindata_frt3_valid_agg = Traindata_frt3_valid_agg.drop('Study', axis = 1)
Traindata_frt3_valid_agg.to_csv('./CheXpert-v1.0{0}/train_valid_agg3.csv'.format(img_type), index = False)
print('Valid 3 data length(study):', len(Traindata_frt3_valid_agg))

Traindata_frt4_valid.loc[:, 'Study'] = Traindata_frt4_valid.Path.str.split('/').str[2] + '/' + Traindata_frt4_valid.Path.str.split('/').str[3]
Traindata_frt4_valid_agg = Traindata_frt4_valid.groupby('Study').agg('first').reset_index()
Traindata_frt4_valid_agg = Traindata_frt4_valid_agg.sort_values('Path')
Traindata_frt4_valid_agg = Traindata_frt4_valid_agg.drop('Study', axis = 1)
Traindata_frt4_valid_agg.to_csv('./CheXpert-v1.0{0}/train_valid_agg4.csv'.format(img_type), index = False)
print('Valid 4 data length(study):', len(Traindata_frt4_valid_agg))

Traindata_frt5_valid.loc[:, 'Study'] = Traindata_frt5_valid.Path.str.split('/').str[2] + '/' + Traindata_frt5_valid.Path.str.split('/').str[3]
Traindata_frt5_valid_agg = Traindata_frt5_valid.groupby('Study').agg('first').reset_index()
Traindata_frt5_valid_agg = Traindata_frt5_valid_agg.sort_values('Path')
Traindata_frt5_valid_agg = Traindata_frt5_valid_agg.drop('Study', axis = 1)
Traindata_frt5_valid_agg.to_csv('./CheXpert-v1.0{0}/train_valid_agg5.csv'.format(img_type), index = False)
print('Valid 5 data length(study):', len(Traindata_frt5_valid_agg))



Validdata = pd.read_csv('./CheXpert-v1.0{0}/valid.csv'.format(img_type))
Validdata_frt = Validdata[Validdata['Path'].str.contains('frontal')].copy()
Validdata_lat = Validdata[Validdata['Path'].str.contains('lateral')].copy()
Validdata_frt.to_csv('./CheXpert-v1.0{0}/valid_frt.csv'.format(img_type), index = False)
Validdata_lat.to_csv('./CheXpert-v1.0{0}/valid_lat.csv'.format(img_type), index = False)
print('Valid data length(frontal):', len(Validdata_frt))
print('Valid data length(lateral):', len(Validdata_lat))
print('Valid data length(total):', len(Validdata_frt) + len(Validdata_lat))

Testdata = pd.read_csv('./CheXpert-v1.0{0}/valid.csv'.format(img_type))
Testdata_frt = Testdata[Testdata['Path'].str.contains('frontal')].copy()
Testdata_lat = Testdata[Testdata['Path'].str.contains('lateral')].copy()
Testdata_frt.to_csv('./CheXpert-v1.0{0}/test_frt.csv'.format(img_type), index = False)
Testdata_lat.to_csv('./CheXpert-v1.0{0}/test_lat.csv'.format(img_type), index = False)
print('Test data length(frontal):', len(Testdata_frt))
print('Test data length(lateral):', len(Testdata_lat))
print('Test data length(total):', len(Testdata_frt) + len(Testdata_lat))

# Make testset for 200 studies (use given valid set as test set)
Testdata_frt.loc[:, 'Study'] = Testdata_frt.Path.str.split('/').str[2] + '/' + Testdata_frt.Path.str.split('/').str[3]
Testdata_frt_agg = Testdata_frt.groupby('Study').agg('first').reset_index()
Testdata_frt_agg = Testdata_frt_agg.sort_values('Path')
Testdata_frt_agg = Testdata_frt_agg.drop('Study', axis = 1)
Testdata_frt_agg.to_csv('./CheXpert-v1.0{0}/test_agg.csv'.format(img_type), index = False)
print('Test data length(study):', len(Testdata_frt_agg))