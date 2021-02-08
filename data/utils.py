import numpy as np
import cv2


def border_pad(image, cfg):
    h, w, c = image.shape

    if cfg.border_pad == 'zero':
        image = np.pad(image, ((0, cfg.long_side - h),
                               (0, cfg.long_side - w), (0, 0)),
                       mode='constant',
                       constant_values=0.0)                     # zero padding
    elif cfg.border_pad == 'pixel_mean':
        image = np.pad(image, ((0, cfg.long_side - h),          # 512 x 512 맞춰주기
                               (0, cfg.long_side - w), (0, 0)), # 오른쪽 32개열 if w=480
                       mode='constant',
                       constant_values=cfg.pixel_mean)          # default = 128
    else:
        image = np.pad(image, ((0, cfg.long_side - h),
                               (0, cfg.long_side - w), (0, 0)),
                       mode=cfg.border_pad)

    return image


def fix_ratio(image, cfg):
    h, w, c = image.shape

    if h >= w:
        ratio = h * 1.0 / w
        h_ = cfg.long_side      # 512 -> 긴 쪽을 512로 고정
        w_ = round(h_ / ratio)  # 원래 이미지 비율 유지 (h=512에 맞춰서)
    else:
        ratio = w * 1.0 / h
        w_ = cfg.long_side      # 512 -> 긴 쪽을 512로 고정
        h_ = round(w_ / ratio)  # 원래 이미지 비율 유지 (w=512에 맞춰서)

    image = cv2.resize(image, dsize=(w_, h_), interpolation=cv2.INTER_LINEAR) # cv2.INTER_LINEAR: 사이즈가 변할때 pixel 사이의 값을 결정하는 방법
    image = border_pad(image, cfg)

    return image


def transform(image, cfg): # equalizeHist, gaussian_blur, COLOR_GRAY2RGB, 512 x 512 with padding then to torch str
    assert image.ndim == 2, "image must be gray image"
    if cfg.use_equalizeHist:
        image = cv2.equalizeHist(image)

    if cfg.gaussian_blur > 0:
        image = cv2.GaussianBlur(
            image,
            (cfg.gaussian_blur, cfg.gaussian_blur), 0) # argument 0 makes the kernel size (3,3) with proper std

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # convert gray to RGB color

    image = fix_ratio(image, cfg) # image: 512 x 512 x 3
    # augmentation for train or co_train

    # normalization
    image = image.astype(np.float32) - cfg.pixel_mean
    # vgg and resnet do not use pixel_std, densenet and inception use.
    if cfg.pixel_std:
        image /= cfg.pixel_std
    # normal image tensor :  H x W x C (0 x 1 x 2)
    # torch image tensor :   C X H X W (2 x 0 x 1)
    image = image.transpose((2, 0, 1))

    return image
