import pandas as pd

train = pd.read_csv('./config/train.csv')
train['Path'] = '/home/jwy4888/leelab/' + train['Path'].astype(str)
train

dev = pd.read_csv('./config/dev.csv')
dev['Path'] = '/home/jwy4888/leelab/' + dev['Path'].astype(str)
dev

train.to_csv('./config/train.csv', index=False)
dev.to_csv('./config/dev.csv', index=False)