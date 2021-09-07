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



######################
## Arguments to Set ##
######################
'''parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('cfg_path', metavar='CFG_PATH', type = str, help = 'Path to the config file in yaml format.')
parser.add_argument('output_path', type = str, help = 'Directory name where results located.')
args = parser.parse_args()
with open(args.cfg_path) as f:
    cfg = edict(json.load(f))
    
PATH = args.output_path
if args.output_path[-1] != '/':
    PATH = PATH + '/'
else:
    PATH = PATH
    
# Paths to the files with training, validation, and test sets.
if cfg.image_type == 'small':
    img_type = '-small'
else:
    img_type = ''
'''

img_type = '-small' ###
PATH = './results/210713_b/' ###

#############################
## Pile Prediction Results ##
#############################
with open('{}boost1/testPROB_frt1.txt'.format(PATH), 'rb') as f:
    outPROB_frt1 = pickle.load(f)

with open('{}boost1/testPROB_lat1.txt'.format(PATH), 'rb') as f:
    outPROB_lat1 = pickle.load(f)

with open('{}boost2/testPROB_frt2.txt'.format(PATH), 'rb') as f:
    outPROB_frt2 = pickle.load(f)

with open('{}boost2/testPROB_lat2.txt'.format(PATH), 'rb') as f:
    outPROB_lat2 = pickle.load(f)

with open('{}boost3/testPROB_frt3.txt'.format(PATH), 'rb') as f:
    outPROB_frt3 = pickle.load(f)

with open('{}boost3/testPROB_lat3.txt'.format(PATH), 'rb') as f:
    outPROB_lat3 = pickle.load(f)

with open('{}boost4/testPROB_frt4.txt'.format(PATH), 'rb') as f:
    outPROB_frt4 = pickle.load(f)

with open('{}boost4/testPROB_lat4.txt'.format(PATH), 'rb') as f:
    outPROB_lat4 = pickle.load(f)

with open('{}boost5/testPROB_frt5.txt'.format(PATH), 'rb') as f:
    outPROB_frt5 = pickle.load(f)

with open('{}boost5/testPROB_lat5.txt'.format(PATH), 'rb') as f:
    outPROB_lat5 = pickle.load(f)

test_frt1 = pd.read_csv('./CheXpert-v1.0{0}/train_frt1_valid.csv'.format(img_type))
test_lat1 = pd.read_csv('./CheXpert-v1.0{0}/train_lat1_valid.csv'.format(img_type))

test_frt2 = pd.read_csv('./CheXpert-v1.0{0}/train_frt2_valid.csv'.format(img_type))
test_lat2 = pd.read_csv('./CheXpert-v1.0{0}/train_lat2_valid.csv'.format(img_type))

test_frt3 = pd.read_csv('./CheXpert-v1.0{0}/train_frt3_valid.csv'.format(img_type))
test_lat3 = pd.read_csv('./CheXpert-v1.0{0}/train_lat3_valid.csv'.format(img_type))

test_frt4 = pd.read_csv('./CheXpert-v1.0{0}/train_frt4_valid.csv'.format(img_type))
test_lat4 = pd.read_csv('./CheXpert-v1.0{0}/train_lat4_valid.csv'.format(img_type))

test_frt5 = pd.read_csv('./CheXpert-v1.0{0}/train_frt5_valid.csv'.format(img_type))
test_lat5 = pd.read_csv('./CheXpert-v1.0{0}/train_lat5_valid.csv'.format(img_type))

class_names = ["Card", "Edem", "Cons", "Atel", "PlEf"]
column_names = ['Path'] + class_names

# Boost 1
df1 = pd.DataFrame(0, index = np.arange(len(test_frt1) + len(test_lat1)), columns = column_names)
test_frt_list1 = list(test_frt1['Path'].copy())
test_lat_list1 = list(test_lat1['Path'].copy())

for i in range(len(test_frt_list1)):
    df1.iloc[i, 0] = test_frt_list1[i]

for i in range(len(test_lat_list1)):
    df1.iloc[len(test_frt_list1) + i, 0] = test_lat_list1[i]

for i in range(len(outPROB_frt1)):
    for j in range(len(outPROB_frt1[i])):
        for k in range(len(class_names)):
            df1.iloc[i*16 + j, k + 1] = outPROB_frt1[i][j][k]

for i in range(len(outPROB_lat1)):
    for j in range(len(outPROB_lat1[i])):
        for k in range(len(class_names)):
            df1.iloc[len(test_frt1) + i*16 + j, k + 1] = outPROB_lat1[i][j][k]
print("Boost df1 done.")

# Boost 2
df2 = pd.DataFrame(0, index = np.arange(len(test_frt2) + len(test_lat2)), columns = column_names)
test_frt_list2 = list(test_frt2['Path'].copy())
test_lat_list2 = list(test_lat2['Path'].copy())

for i in range(len(test_frt_list2)):
    df2.iloc[i, 0] = test_frt_list2[i]

for i in range(len(test_lat_list2)):
    df2.iloc[len(test_frt_list2) + i, 0] = test_lat_list2[i]

for i in range(len(outPROB_frt2)):
    for j in range(len(outPROB_frt2[i])):
        for k in range(len(class_names)):
            df2.iloc[i*16 + j, k + 1] = outPROB_frt2[i][j][k]

for i in range(len(outPROB_lat2)):
    for j in range(len(outPROB_lat2[i])):
        for k in range(len(class_names)):
            df2.iloc[len(test_frt2) + i*16 + j, k + 1] = outPROB_lat2[i][j][k]
print("Boost df2 done.")

# Boost 3
df3 = pd.DataFrame(0, index = np.arange(len(test_frt3) + len(test_lat3)), columns = column_names)
test_frt_list3 = list(test_frt3['Path'].copy())
test_lat_list3 = list(test_lat3['Path'].copy())

for i in range(len(test_frt_list3)):
    df3.iloc[i, 0] = test_frt_list3[i]

for i in range(len(test_lat_list3)):
    df3.iloc[len(test_frt_list3) + i, 0] = test_lat_list3[i]

for i in range(len(outPROB_frt3)):
    for j in range(len(outPROB_frt3[i])):
        for k in range(len(class_names)):
            df3.iloc[i*16 + j, k + 1] = outPROB_frt3[i][j][k]

for i in range(len(outPROB_lat3)):
    for j in range(len(outPROB_lat3[i])):
        for k in range(len(class_names)):
            df3.iloc[len(test_frt3) + i*16 + j, k + 1] = outPROB_lat3[i][j][k]
print("Boost df3 done.")

# Boost 4
df4 = pd.DataFrame(0, index = np.arange(len(test_frt4) + len(test_lat4)), columns = column_names)
test_frt_list4 = list(test_frt4['Path'].copy())
test_lat_list4 = list(test_lat4['Path'].copy())

for i in range(len(test_frt_list4)):
    df4.iloc[i, 0] = test_frt_list4[i]

for i in range(len(test_lat_list4)):
    df4.iloc[len(test_frt_list4) + i, 0] = test_lat_list4[i]

for i in range(len(outPROB_frt4)):
    for j in range(len(outPROB_frt4[i])):
        for k in range(len(class_names)):
            df4.iloc[i*16 + j, k + 1] = outPROB_frt4[i][j][k]

for i in range(len(outPROB_lat4)):
    for j in range(len(outPROB_lat4[i])):
        for k in range(len(class_names)):
            df4.iloc[len(test_frt4) + i*16 + j, k + 1] = outPROB_lat4[i][j][k]
print("Boost df4 done.")

# Boost 5
df5 = pd.DataFrame(0, index = np.arange(len(test_frt5) + len(test_lat5)), columns = column_names)
test_frt_list5 = list(test_frt5['Path'].copy())
test_lat_list5 = list(test_lat5['Path'].copy())

for i in range(len(test_frt_list5)):
    df5.iloc[i, 0] = test_frt_list5[i]

for i in range(len(test_lat_list5)):
    df5.iloc[len(test_frt_list5) + i, 0] = test_lat_list5[i]

for i in range(len(outPROB_frt5)):
    for j in range(len(outPROB_frt5[i])):
        for k in range(len(class_names)):
            df5.iloc[i*16 + j, k + 1] = outPROB_frt5[i][j][k]

for i in range(len(outPROB_lat5)):
    for j in range(len(outPROB_lat5[i])):
        for k in range(len(class_names)):
            df5.iloc[len(test_frt5) + i*16 + j, k + 1] = outPROB_lat5[i][j][k]
print("Boost df5 done.")


df_pile = pd.concat([df1, df2, df3, df4, df5], axis = 0).reset_index(drop=True)
print("Boost df piled.")


#################
## Pile Labels ##
#################
label_frt1 = test_frt1[['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']].copy()
label_frt2 = test_frt2[['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']].copy()
label_frt3 = test_frt3[['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']].copy()
label_frt4 = test_frt4[['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']].copy()
label_frt5 = test_frt5[['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']].copy()

label_lat1 = test_lat1[['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']].copy()
label_lat2 = test_lat2[['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']].copy()
label_lat3 = test_lat3[['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']].copy()
label_lat4 = test_lat4[['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']].copy()
label_lat5 = test_lat5[['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']].copy()

label_pile = pd.concat([label_frt1, label_lat1, 
                        label_frt2, label_lat2, 
                        label_frt3, label_lat3, 
                        label_frt4, label_lat4, 
                        label_frt5, label_lat5]).reset_index(drop=True)

label_pile['Cardiomegaly'] = label_pile['Cardiomegaly'].replace([-1.0], 0)
label_pile['Consolidation'] = label_pile['Consolidation'].replace([-1.0], 0)
label_pile['Atelectasis'] = label_pile['Atelectasis'].replace([-1.0], 1)
label_pile['Edema'] = label_pile['Edema'].replace([-1.0], 1)
label_pile['Pleural Effusion'] = label_pile['Pleural Effusion'].replace([-1.0], 1)
label_pile = label_pile.fillna(0)
print("Boost label piled.")



###############
## Pile EHRs ##
###############
ehr_frt1 = test_frt1[['Sex', 'Age', 'AP/PA']].copy()
ehr_frt2 = test_frt2[['Sex', 'Age', 'AP/PA']].copy()
ehr_frt3 = test_frt3[['Sex', 'Age', 'AP/PA']].copy()
ehr_frt4 = test_frt4[['Sex', 'Age', 'AP/PA']].copy()
ehr_frt5 = test_frt5[['Sex', 'Age', 'AP/PA']].copy()

ehr_lat1 = test_lat1[['Sex', 'Age', 'AP/PA']].copy()
ehr_lat2 = test_lat2[['Sex', 'Age', 'AP/PA']].copy()
ehr_lat3 = test_lat3[['Sex', 'Age', 'AP/PA']].copy()
ehr_lat4 = test_lat4[['Sex', 'Age', 'AP/PA']].copy()
ehr_lat5 = test_lat5[['Sex', 'Age', 'AP/PA']].copy()

ehr_pile = pd.concat([ehr_frt1, ehr_lat1, 
                      ehr_frt2, ehr_lat2, 
                      ehr_frt3, ehr_lat3, 
                      ehr_frt4, ehr_lat4, 
                      ehr_frt5, ehr_lat5]).reset_index(drop=True)
print("Boost EHR piled.")



##################################
## Agg Probs & Labels & Studies ##
##################################
pl_pile = pd.concat([df_pile, label_pile, ehr_pile], axis = 1)
pl_pile = pl_pile.rename(columns={'Pleural Effusion': 'Pleural_Effusion'})
'''pl_pile_agg = pl_pile.groupby('Path').agg({'Card' : 'min',
                                           'Edem' : 'min',
                                           'Cons' : 'min',
                                           'Atel' : 'min',
                                           'PlEf' : 'min',
                                           'Cardiomegaly' : 'min',
                                           'Edema' : 'min',
                                           'Consolidation' : 'min',
                                           'Atelectasis' : 'min',
                                           'Pleural_Effusion' : 'min',
                                           'Sex' : lambda d: " ".join(set(d)),
                                           'Age' : 'min',
                                           'AP/PA' : lambda d: set(d)}).reset_index() # max -> mean -> min'''


pl_pile_agg = pl_pile.sort_values('Path').reset_index(drop=True)

pl_pile_agg['Card_diff'] = pl_pile_agg.apply(lambda row: abs(row.Card - row.Cardiomegaly), axis = 1)
pl_pile_agg['Edem_diff'] = pl_pile_agg.apply(lambda row: abs(row.Edem - row.Edema), axis = 1)
pl_pile_agg['Cons_diff'] = pl_pile_agg.apply(lambda row: abs(row.Cons - row.Consolidation), axis = 1)
pl_pile_agg['Atel_diff'] = pl_pile_agg.apply(lambda row: abs(row.Atel - row.Atelectasis), axis = 1)
pl_pile_agg['PlEf_diff'] = pl_pile_agg.apply(lambda row: abs(row.PlEf - row.Pleural_Effusion), axis = 1)

pl_pile_agg['Card_flag'] = pl_pile_agg.apply(lambda row: row.Card_diff > 0.5, axis = 1)
pl_pile_agg['Edem_flag'] = pl_pile_agg.apply(lambda row: row.Edem_diff > 0.5, axis = 1)
pl_pile_agg['Cons_flag'] = pl_pile_agg.apply(lambda row: row.Cons_diff > 0.5, axis = 1)
pl_pile_agg['Atel_flag'] = pl_pile_agg.apply(lambda row: row.Atel_diff > 0.5, axis = 1)
pl_pile_agg['PlEf_flag'] = pl_pile_agg.apply(lambda row: row.PlEf_diff > 0.5, axis = 1)
print("Boost diff calculated.")



######################
## Confusion Matrix ##
######################
print(pl_pile_agg.Card_flag.value_counts())
print(pl_pile_agg.Edem_flag.value_counts())
print(pl_pile_agg.Cons_flag.value_counts())
print(pl_pile_agg.Atel_flag.value_counts())
print(pl_pile_agg.PlEf_flag.value_counts())

cm_Card_sex = pd.crosstab(pl_pile_agg['Card_flag'], pl_pile_agg['Sex'], rownames=['Card'], colnames=['Sex'])
print(cm_Card_sex)



#########################
## Update Data Weights ##
#########################
data_frt = pl_pile_agg[pl_pile_agg['Path'].str.contains('frontal')].reset_index(drop=True).copy()
data_lat = pl_pile_agg[pl_pile_agg['Path'].str.contains('lateral')].reset_index(drop=True).copy()

data_frt = data_frt.drop(['Card_diff', 'Edem_diff', 'Cons_diff', 'Atel_diff', 'PlEf_diff', 
                          'Card_flag', 'Edem_flag', 'Cons_flag', 'Atel_flag', 'PlEf_flag',
                          'Sex', 'Age', 'AP/PA'], axis = 1)
data_lat = data_lat.drop(['Card_diff', 'Edem_diff', 'Cons_diff', 'Atel_diff', 'PlEf_diff', 
                          'Card_flag', 'Edem_flag', 'Cons_flag', 'Atel_flag', 'PlEf_flag',
                          'Sex', 'Age', 'AP/PA'], axis = 1)

data_frt['weight_0'] = 1 / len(data_frt)
data_lat['weight_0'] = 1 / len(data_lat)

# data_frt['weight_1'] = data_frt['weight_0'] * np.exp(-cfg.lr * (cfg.nnClassCount - 1) / cfg.nnClassCount * (np.log(data_frt['Card']) * data_frt['Cardiomegaly'] + np.log(data_frt['Edem']) * data_frt['Edema'] + np.log(data_frt['Cons']) * data_frt['Consolidation'] + np.log(data_frt['Atel']) * data_frt['Atelectasis'] + np.log(data_frt['PlEf']) * data_frt['Pleural_Effusion']))
data_frt['weight_1'] = data_frt['weight_0'] * np.exp(-0.0001 * (5 - 1) / 5 * (np.log(data_frt['Card']) * data_frt['Cardiomegaly'] + np.log(data_frt['Edem']) * data_frt['Edema'] + np.log(data_frt['Cons']) * data_frt['Consolidation'] + np.log(data_frt['Atel']) * data_frt['Atelectasis'] + np.log(data_frt['PlEf']) * data_frt['Pleural_Effusion'])) ###
data_frt['weight_1'] = data_frt['weight_1'] / np.sum(data_frt['weight_1']) * len(data_frt) # normalize weights

data_lat['weight_1'] = data_lat['weight_0'] * np.exp(-0.0001 * (5 - 1) / 5 * (np.log(data_lat['Card']) * data_lat['Cardiomegaly'] + np.log(data_lat['Edem']) * data_lat['Edema'] + np.log(data_lat['Cons']) * data_lat['Consolidation'] + np.log(data_lat['Atel']) * data_lat['Atelectasis'] + np.log(data_lat['PlEf']) * data_lat['Pleural_Effusion'])) ###
data_lat['weight_1'] = data_lat['weight_1'] / np.sum(data_lat['weight_1']) * len(data_lat) # normalize weights

data_frt.to_csv('./CheXpert-v1.0{0}/data_frt_boost1.csv'.format(img_type), index = False)
data_lat.to_csv('./CheXpert-v1.0{0}/data_lat_boost1.csv'.format(img_type), index = False)



# Make batches
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

weight_1_frt = list(chunks(data_frt['weight_1'].tolist(), 16))
weight_1_lat = list(chunks(data_lat['weight_1'].tolist(), 16))

# Save the sample weights
with open('{}weight_1_frt.txt'.format(PATH), 'wb') as fp:
    pickle.dump(weight_1_frt, fp)

with open('{}weight_1_lat.txt'.format(PATH), 'wb') as fp:
    pickle.dump(weight_1_lat, fp)