'''Data Preprocessing'''

###################
## Prerequisites ##
###################
import pandas as pd



###################
## Preprocessing ##
###################
# Each file contains pairs (path to image, output vector)
Traindata = pd.read_csv('./CheXpert-v1.0-small/train.csv')
Traindata = Traindata[500:]
Traindata.to_csv('./CheXpert-v1.0-small/train_mod.csv', index = False)
print('Train data length:', len(Traindata))

Validdata = pd.read_csv('./CheXpert-v1.0-small/valid.csv')
Validdata.to_csv('./CheXpert-v1.0-small/valid_mod.csv', index = False)
print('Valid data length:', len(Validdata))

Testdata = Traindata.head(500) # use first 500 training data as test data (observation ratio is almost same as training set!)
Testdata.to_csv('./CheXpert-v1.0-small/test_mod.csv', index = False)
print('Test data length:', len(Testdata))