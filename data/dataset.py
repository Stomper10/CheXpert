import numpy as np
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
from data.imgaug import GetTransforms
from data.utils import transform
np.random.seed(0)


class ImageDataset(Dataset):
    def __init__(self, label_path, cfg, mode='train'):
        self.cfg = cfg
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},    # uncertain policy -> zeroes
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]  # uncertain policy -> ones
        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = [
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
                for index, value in enumerate(fields[5:]): # index: 0 to 13, value: label
                    if index == 5 or index == 8:                    # if Edema or Atelectasis
                        labels.append(self.dict[1].get(value))      # apply 'ones' policy (append value to 'labels' list: float to int)
                        if self.dict[1].get(                             # if original value was 1.0 or -1.0
                                value) == '1' and \
                                self.cfg.enhance_index.count(index) > 0: # always False since index == 5 or index == 8
                            flg_enhance = True
                    elif index == 2 or index == 6 or index == 10:   # if Cardiomegaly or Consolidation or Pleural Effusion
                        labels.append(self.dict[0].get(value))      # apply 'zeroes' policy (append value to 'labels' list: float to int)
                        if self.dict[0].get(                             # if original value was 1.0 or -1.0
                                value) == '1' and \
                                self.cfg.enhance_index.count(index) > 0: # True when index == 2 or index == 6
                            flg_enhance = True
                # labels = ([self.dict.get(n, n) for n in fields[5:]])
                labels = list(map(int, labels)) # e.g. ['0', '0', '0', '1', '1'] -> [0, 0, 0, 1, 1]
                self._image_paths.append(image_path)
                assert os.path.exists(image_path), image_path # assert COND, MESS -> if COND is False
                self._labels.append(labels) # append list of single image (5 obs)
                if flg_enhance and self._mode == 'train':       # enhance obs가 존재하는 image에 대해 -> image training에 cfg.enhance_times번 더 추가
                    for i in range(self.cfg.enhance_times):     # cfg.enhance_times == 1 (default)
                        self._image_paths.append(image_path)
                        self._labels.append(labels)
        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = cv2.imread(self._image_paths[idx], 0) # read image as grayscale
        image = Image.fromarray(image) # from numpy array to image
        if self._mode == 'train':           # /data/imgaug.py
            image = GetTransforms(image, type=self.cfg.use_transforms_type) # tfs.RandomAffine as default
        image = np.array(image)
        image = transform(image, self.cfg)  # /data/utils.py: image 3 x 512 x 512
        labels = np.array(self._labels[idx]).astype(np.float32) # list to ndarray with np.float32 type

        path = self._image_paths[idx]

        if self._mode == 'train' or self._mode == 'dev':
            return (image, labels)
        elif self._mode == 'test':
            return (image, path)
        elif self._mode == 'heatmap':
            return (image, path, labels)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))
