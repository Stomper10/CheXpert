device_ids = "0,1,2,3"


list(map(int, device_ids.split(',')))

import numpy as np
np.zeros(5)

import json
from easydict import EasyDict as edict
import os

with open('./config/example.json') as f:
    cfg = edict(json.load(f))
    print(cfg)
    if True:
        print(json.dumps(cfg, indent=4))
cfg

with open(os.path.join('./logdir', 'cfg.json'), 'w') as f:
    json.dump(cfg, f, indent=1)

pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

help="If get"   "parameters from pretrained model"


dst_folder = os.path.join('./logdir', 'classification')
print(dst_folder)

import subprocess
rc, size = subprocess.getstatusoutput('ls')
print(size)

os.path.dirname(os.path.abspath(__file__)) + '/../'


my_dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},    # uncertain policy -> zeroes
           {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]

my_dict[1].get('1.0')

with open('../../../Desktop/Univ/LeeLab/CheXpert_store/CheXpert-v1.0-small/valid.csv') as f:
    header = f.readline().strip('\n').split(',')
    label_header = [
        header[7],  # Cardiomagaly
        header[10], # Edema
        header[11], # Consolidation
        header[13], # Atelectasis
        header[15]] # Pleural Effusion
    for line in f:
        labels = []
        fields = line.strip('\n').split(',')
        image_path = fields[0] # e.g. CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg
        flg_enhance = False
        for index, value in enumerate(fields[5:]):
            if index == 5 or index == 8:                    # if Edema of Atelectasis
                labels.append(my_dict[1].get(value))      # apply 'ones' policy (append value of dict[1])
            elif index == 2 or index == 6 or index == 10:   # if Cardiomegaly or Consolidation or Pleural Effusion
                labels.append(my_dict[0].get(value))      # apply 'zeroes' policy
                if my_dict[0].get(
                        value) == '1' and \
                        [2,6].count(index) > 0:
                    flg_enhance = True
        # labels = ([self.dict.get(n, n) for n in fields[5:]])
        labels = list(map(int, labels))
        print(flg_enhance)
        _image_paths.append(image_path)
        assert os.path.exists(image_path), image_path # assert COND, MESS -> if COND is False
        self._labels.append(labels) # append list of single image (5 obs)
        if flg_enhance and self._mode == 'train':
            for i in range(self.cfg.enhance_times):
                self._image_paths.append(image_path)
                self._labels.append(labels)

[2, 6].count(0)



import numpy as np
np.power(0.1, 0)
0.0001 * np.power(0.1, 2)

summary_train, best_dict = train_epoch( # 210205
            summary_train, summary_dev, cfg, args, model,
            dataloader_train, dataloader_dev, optimizer,
            summary_writer, best_dict, dev_header)

# target output size of 5x7
import torch
from torch import nn
m = nn.AdaptiveAvgPool2d((1,1))
n = nn.AdaptiveMaxPool2d((1,1))
input = torch.randn(1, 8, 8)
output1 = m(input)
output2 = n(input)
torch.cat((1, 1),1)

for index, num_class in enumerate([1,1,1,1,1]):
    print(index)
    print(num_class)