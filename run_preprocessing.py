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
studies = 1
test_images = []
for i in range(len(paths) - 1):
    if paths[i][26:45] != paths[i+1][26:45]:
        studies += 1
    if studies <= 500:
        test_images.append(paths[i])

Traindata = Traindata_raw[len(test_images):]
Traindata.to_csv('./CheXpert-v1.0-small/train_mod.csv', index = False)
print('Train data length:', len(Traindata))

Validdata = pd.read_csv('./CheXpert-v1.0-small/valid.csv')
Validdata.to_csv('./CheXpert-v1.0-small/valid_mod.csv', index = False)
print('Valid data length:', len(Validdata))

Testdata = Traindata_raw[:len(test_images)] # use first 500 studies from training set as test set (observation ratio is almost same!)
Testdata.to_csv('./CheXpert-v1.0-small/test_mod.csv', index = False)
print('Test data length:', len(Testdata))