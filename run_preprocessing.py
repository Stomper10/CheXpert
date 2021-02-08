'''Data Preprocessing'''

###################
## Prerequisites ##
###################
import pandas as pd
pd.set_option('mode.chained_assignment',  None)



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
Traindata.to_csv('./CheXpert-v1.0-small/train_all.csv', index = False)
print('Train data length:', len(Traindata))

Validdata = pd.read_csv('./CheXpert-v1.0-small/valid.csv')
Validdata.to_csv('./CheXpert-v1.0-small/valid_all.csv', index = False)
print('Valid data length:', len(Validdata))

Testdata = Traindata_raw[:border_idx] # use first 500 studies from training set as test set (observation ratio is almost same!)
Testdata.to_csv('./CheXpert-v1.0-small/test_all.csv', index = False)
print('Test data length:', len(Testdata))