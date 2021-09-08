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
Traindata_frt = Traindata[Traindata['Path'].str.contains('frontal')].copy()
Traindata_lat = Traindata[Traindata['Path'].str.contains('lateral')].copy()
Traindata_frt.to_csv('./CheXpert-v1.0{0}/train_frt.csv'.format(img_type), index = False)
Traindata_lat.to_csv('./CheXpert-v1.0{0}/train_lat.csv'.format(img_type), index = False)
print('Train data length(frontal):', len(Traindata_frt))
print('Train data length(lateral):', len(Traindata_lat))
print('Train data length(total):', len(Traindata_frt) + len(Traindata_lat))

Validdata = pd.read_csv('./CheXpert-v1.0{0}/valid.csv'.format(img_type))
Validdata_frt = Validdata[Validdata['Path'].str.contains('frontal')].copy()
Validdata_lat = Validdata[Validdata['Path'].str.contains('lateral')].copy()
Validdata_frt.to_csv('./CheXpert-v1.0{0}/valid_frt.csv'.format(img_type), index = False)
Validdata_lat.to_csv('./CheXpert-v1.0{0}/valid_lat.csv'.format(img_type), index = False)
print('Valid data length(frontal):', len(Validdata_frt))
print('Valid data length(lateral):', len(Validdata_lat))
print('Valid data length(total):', len(Validdata_frt) + len(Validdata_lat))

Testdata = pd.read_csv('./CheXpert-v1.0{0}/valid.csv'.format(img_type))
Testdata_frt = Testdata[Testdata['Path'].str.contains('frontal')].copy() # to avoid SettingWithCopyWarning
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