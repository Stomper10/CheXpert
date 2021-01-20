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
    if studies <= 501:
        test_images.append(paths[i])

Traindata = Traindata_raw[len(test_images):]
Traindata_frt = Traindata[Traindata['Path'].str.contains('frontal')]
Traindata_lat = Traindata[Traindata['Path'].str.contains('lateral')]
Traindata_frt.to_csv('./CheXpert-v1.0-small/train_frt.csv', index = False)
Traindata_lat.to_csv('./CheXpert-v1.0-small/train_lat.csv', index = False)
print('Train data length(frontal):', len(Traindata_frt))
print('Train data length(lateral):', len(Traindata_lat))

Validdata = pd.read_csv('./CheXpert-v1.0-small/valid.csv')
Validdata_frt = Validdata[Validdata['Path'].str.contains('frontal')]
Validdata_lat = Validdata[Validdata['Path'].str.contains('lateral')]
Validdata_frt.to_csv('./CheXpert-v1.0-small/valid_frt.csv', index = False)
Validdata_lat.to_csv('./CheXpert-v1.0-small/valid_lat.csv', index = False)
print('Valid data length(frontal):', len(Validdata_frt))
print('Valid data length(lateral):', len(Validdata_lat))

Testdata = Traindata_raw[:len(test_images)] # use first 500 studies from training set as test set (observation ratio is almost same!)
Testdata_frt = Testdata[Testdata['Path'].str.contains('frontal')]
Testdata_lat = Testdata[Testdata['Path'].str.contains('lateral')]
Testdata_frt.to_csv('./CheXpert-v1.0-small/test_frt.csv', index = False)
Testdata_lat.to_csv('./CheXpert-v1.0-small/test_lat.csv', index = False)
print('Test data length(frontal):', len(Testdata_frt))
print('Test data length(lateral):', len(Testdata_lat))