# CheXpert: A Large Chest X-Ray Dataset and Competitions
> CheXpert is a large dataset of chest X-rays and competition for automated chest X-ray interpretation, which features uncertainty labels and radiologist-labeled reference standard evaluation sets. *- from the CheXpert [webpage](https://stanfordmlgroup.github.io/competitions/chexpert/) description*

This repository aims to reproduce CheXpert [paper](https://arxiv.org/pdf/1901.07031.pdf)'s results using the PyTorch library.
So I tried to use the same model(DenseNet121) and the parameters as the paper.

This repository especially referenced [here](https://github.com/gaetandi/cheXpert) for the coding part.  
You can easily run the code with the following instructions.

# 0. Prerequisites
- Python3
- Git Bash

Open Git Bash and use `git clone` command to download this repository.
```bash
git clone https://github.com/Stomper10/CheXpert.git
```

# 1. Download Data
At the bottom of the CheXpert [webpage](https://stanfordmlgroup.github.io/competitions/chexpert/), write a registration form to download the CheXpert dataset.
You will receive a link to the download over email. Right-click your mouse on the download link(439GB or 11GB) and click 'Copy link address'.

Then, open Git Bash to run the following command. Paste your link address inside the double quotation marks(RUN WITH QUOTATION MARKS). It will take a while even for the downsampled(11GB) dataset. In this experiment, I used the downsampled dataset.
```bash
wget "link_address_you_copied"
```

After you downloaded the dataset, run the following command to unzip the `.zip` file.
```bash
unzip CheXpert-v1.0-small.zip
```

Now the dataset is ready. You have to place `CheXpert-v1.0-small` directory and `CheXpert_DenseNet121.ipynb` file(or `CheXpert_DenseNet121.py`) at the same location unless you modify the path in the source code.

# 2. Run the Code
If you want to use `.ipynb` file, run `CheXpert_DenseNet121.ipynb` file. If you want `.py` file instead, `CheXpert_DenseNet121.py` is also ready for you. Same contents with the different file extensions. Maybe you need to install `PyTorch` and `Bar` packages before you run the code.  

For those who want to compare, my running environment was as below(used GPU server).
* Intel Xeon Silver 4216 CPU
* 512GB memory
* Nvidia Titan RTX GPU

# 3. Results
You may get training & validation losses and ROC curves for results. You can also check the computational costs. Saved model and ROC curve `.png` files are saved in the `Results` directory(I manually moved saved model and `.png` files after creating `Results` directory). Let me just show you the ROC curves here.


![](https://github.com/Stomper10/CheXpert/blob/master/Results/ROCfor10%25.png)


This table below is a comparison with original paper results(used 10% of the training dataset).


Observation | Experiment AUC | Paper AUC | Difference
:-: | :-: | :-: | :-:
Atelectasis | 0.64 | 0.85 | -0.21
Cardiomegaly | 0.83 | 0.90 | -0.07
Consolidation | 0.61 | 0.90 | -0.29
Edema | 0.81 | 0.92 | -0.11
Pleural Effusion | 0.82 | 0.97 | -0.15


# 4. Task Lists
- [x] Use subset of training dataset(10%) to check computational costs and performances.
- [ ] Use whole training dataset and compare performances.
- [ ] Apply Grad-CAM method for localization.
- [ ] Use original dataset for training(~439GB).

# 5. References
- CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison, Irvin, Jeremy, et al., 2019 [[Arxiv:1901.07031]](https://arxiv.org/pdf/1901.07031.pdf)
- Densely Connected Convolutional Networks, Huang et al., 2018 [[Arxiv:1608.06993]](https://arxiv.org/pdf/1608.06993.pdf)
- [Github repository](https://github.com/gaetandi/cheXpert)
