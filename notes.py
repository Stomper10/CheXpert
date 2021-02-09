import pandas as pd

train = pd.read_csv('./config/train.csv')

train['Path'] = train['Path'].str.replace('//home/leelab', '')

dev = pd.read_csv('./config/dev.csv')
dev['Path'] = dev['Path'].str.replace('//home/leelab', '')
dev

train.to_csv('./config/train.csv', index=False)
dev.to_csv('./config/dev.csv', index=False)