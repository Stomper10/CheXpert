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

# 1. Download the Data
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

Now the dataset is ready. You have to place `CheXpert-v1.0-small` directory and `CheXpert_DenseNet121.ipynb` file(or `_ALL` & `_FL` files) at the same location unless you modify the path in the source code.

# 2. Run the Code
Maybe you need to install `PyTorch` and `barbar` packages before you run the code.
* Train with 10% of dataset: Use `CheXpert_DenseNet121_10%.ipynb` file.

* Train with 100% of dataset: Use `CheXpert_DenseNet121_ALL.ipynb` file. 

* Train with 100% of dataset(using federated learning): Use `CheXpert_DenseNet121_FL.ipynb` file. You can modify federated learning hyperparameters.

# 3. Results
You may get training & validation losses and ROC curves for results. You can also check the computational costs. Saved model and ROC curve `.png` files are saved in the `Results` directory(I manually moved saved model and `.png` files after creating `Results` directory). Let me just show you the ROC curves here.

![](https://github.com/Stomper10/CheXpert/blob/master/Results/ROCfor100%25.png)

This table shows a comparison with original paper results(used 100% of the training dataset).

Observation | Experiment AUC | Paper AUC | Difference
:-: | :-: | :-: | :-:
Atelectasis | 0.75 | 0.85 | -0.10
Cardiomegaly | 0.87 | 0.90 | -0.03
Consolidation | 0.77 | 0.90 | -0.13
Edema | 0.81 | 0.84 | -0.08
Pleural Effusion | 0.89 | 0.97 | -0.08

For those who want to compare the running environment, mine was as below(used GPU server).
* Intel Xeon Silver 4216 CPU
* 512GB memory
* Nvidia Titan RTX GPU

# 4. Task Lists
- [x] Use subset of training dataset(10%) to check computational costs and performances.
- [x] Adjust the number of training data per each class.
- [x] Use whole training dataset and compare performances.
- [x] Try federated learning technique.
- [ ] Apply Grad-CAM method for localization.
- [ ] Use original dataset for training(~439GB).

# 5. References
- CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison, Irvin et al., 2019 [[arXiv:1901.07031]](https://arxiv.org/pdf/1901.07031.pdf)
- Densely Connected Convolutional Networks, Huang et al., 2018 [[arXiv:1608.06993v5]](https://arxiv.org/pdf/1608.06993.pdf)
- Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, Selvaraju et al., 2019 [[arXiv:1610.02391v4]](https://arxiv.org/pdf/1610.02391.pdf)
- Communication-Efficient Learning of Deep Networks from Decentralized Data, McMahan et al., 2017 [[arXiv:1602.05629v3]](https://arxiv.org/pdf/1602.05629.pdf)
- [Github repository](https://github.com/gaetandi/cheXpert)
