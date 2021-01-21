'''Data Preprocessing'''

###################
## Prerequisites ##
###################
import pandas as pd



###################
## Preprocessing ##
###################
# Each file contains pairs (path to image, output vector)
Traindata_raw = pd.read_csv('./CheXpert-v1.0-small/train.csv')
paths = list(Traindata_raw['Path'])

for i in range(len(paths)):
    paths[i] = paths[i][26:45]

path_unique = list(dict.fromkeys(paths))
border = path_unique[500]
border_idx = len(paths) - 1 - paths[::-1].index(border)

Traindata = Traindata_raw[border_idx:]
Traindata_frt = Traindata[Traindata['Path'].str.contains('frontal')]
Traindata_lat = Traindata[Traindata['Path'].str.contains('lateral')]
Traindata_frt.to_csv('./CheXpert-v1.0-small/train_frt.csv', index = False)
Traindata_lat.to_csv('./CheXpert-v1.0-small/train_lat.csv', index = False)
print('Train data length(frontal):', len(Traindata_frt))
print('Train data length(lateral):', len(Traindata_lat))
print('Train data length(total):', len(Traindata_frt) + len(Traindata_lat))

Validdata = pd.read_csv('./CheXpert-v1.0-small/valid.csv')
Validdata_frt = Validdata[Validdata['Path'].str.contains('frontal')]
Validdata_lat = Validdata[Validdata['Path'].str.contains('lateral')]
Validdata_frt.to_csv('./CheXpert-v1.0-small/valid_frt.csv', index = False)
Validdata_lat.to_csv('./CheXpert-v1.0-small/valid_lat.csv', index = False)
print('Valid data length(frontal):', len(Validdata_frt))
print('Valid data length(lateral):', len(Validdata_lat))
print('Valid data length(total):', len(Validdata_frt) + len(Validdata_lat))

Testdata = Traindata_raw[:border_idx] # use first 500 studies from training set as test set (observation ratio is almost same!)
Testdata_frt = Testdata[Testdata['Path'].str.contains('frontal')]
Testdata_lat = Testdata[Testdata['Path'].str.contains('lateral')]
Testdata_frt.to_csv('./CheXpert-v1.0-small/test_frt.csv', index = False)
Testdata_lat.to_csv('./CheXpert-v1.0-small/test_lat.csv', index = False)
print('Test data length(frontal):', len(Testdata_frt))
print('Test data length(lateral):', len(Testdata_lat))
print('Test data length(total):', len(Testdata_frt) + len(Testdata_lat))