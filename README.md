# CheXpert: A Large Chest X-Ray Dataset and Competitions
> CheXpert is a large dataset of chest X-rays and competition for automated chest X-ray interpretation, which features uncertainty labels and radiologist-labeled reference standard evaluation sets. *- from the CheXpert [webpage](https://stanfordmlgroup.github.io/competitions/chexpert/) description*

This repository aims to reproduce CheXpert [paper](https://arxiv.org/pdf/1901.07031.pdf)'s results using the PyTorch library.
So I tried to use the same model(DenseNet121) and the parameters as the paper.

This repository especially referenced [here](https://github.com/gaetandi/cheXpert) for the coding part.  
You can easily run the code with the following instructions.
This code is written for the GPU environment.

# 0. Prerequisites
- Python3
- Git Bash (whatever can handle git)

Open Git Bash and use `git clone` command to download this repository.

```bash
git clone https://github.com/Stomper10/CheXpert.git
```

# 1. Download the Data
At the bottom of the CheXpert [webpage](https://stanfordmlgroup.github.io/competitions/chexpert/), write a registration form to download the CheXpert dataset.
You will receive an email with the download link. Right-click your mouse on the download link (439GB or 11GB) and click 'Copy link address'.

Then, open Git Bash to run the following command. Paste your link address inside the double quotation marks (PLEASE RUN WITH QUOTATION MARKS). It will take a while even for the downsampled (11GB) dataset. In this experiment, I used the downsampled dataset.

```bash
wget "link_address_you_copied"
```

After you downloaded the dataset, run the following command to unzip the `.zip` file.

```bash
unzip CheXpert-v1.0-small.zip
```

Now the dataset is ready. As you see this repository structure, you have to place the `CheXpert-v1.0-small` directory and the rest `.py` files at the same location unless you modify the path in the source code.

# 2. Run the Code
You may need to install the `PyTorch` library before you run the code.
1. Data Preprocessing
In the current version, I set the model to use only frontal images. So, you MUST run the following code before training the model.
```bash
python3 run_preprocessing.py
```

2. Run the Model
You can give several options to run the model.
Options | Shortcut | Description
:-: | :-: | :-:
--policy | -p | Uncertain label policy.
--ratio | -r | Training data ratio.
--output_path| -o | Path to save models and ROC curve plot.
--random_seed | -s | Random seed for reproduction.

If you want to use 1% of training set to train the model with `policy = ones`, run like below.
```bash
python3 run_chexpert.py \
  --policy = ones \
  --ratio = 0.01 \
  --output_path = results \
  --random_seed = 1
```

I recommend to use `nohup` command if you run this code on server.
```bash
nohup python3 run_chexpert.py > result.txt &
```

* Train using the federated learning: Use `CheXpert_DenseNet121_FL.ipynb` file. You can modify federated learning hyperparameters.
* You can also try the Grad-CAM method on test dataset with `Grad-CAM.ipynb` file after you get the trained model.

# 3. Results
You may get training and validation losses, as well as the test accuracy like ROC curves. You can also check the computational costs. Saved model and ROC curve `.png` files are saved in the `Results` directory(I manually moved saved model and `.png` files after creating `Results` directory). Let me just show you the ROC curves here.

![](https://github.com/Stomper10/CheXpert/blob/master/Results/ROCfor100%25.png)

This table shows a comparison with original paper results(used 100% of the training dataset).

Observation | Experiment AUC | Paper AUC | Difference
:-: | :-: | :-: | :-:
Atelectasis | 0.74 | 0.85 | -0.11
Cardiomegaly | 0.86 | 0.90 | -0.04
Consolidation | 0.76 | 0.90 | -0.14
Edema | 0.84 | 0.84 | 0.00
Pleural Effusion | 0.89 | 0.97 | -0.08

For those who want to compare the running environment, mine was as below(used GPU server).
* Intel Xeon Silver 4216 CPU
* 512GB memory
* Four Nvidia Titan RTX GPUs

# 4. Task Lists
- [x] Use subset of training dataset(10%) to check computational costs and performances.
- [x] Adjust the number of training data per each class.
- [x] Use whole training dataset and compare performances.
- [x] Try federated learning technique.
- [x] Apply Grad-CAM method for localization.
- [ ] Use original dataset for training(~439GB).

# 5. References
- CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison, Irvin et al., 2019 [[arXiv:1901.07031]](https://arxiv.org/pdf/1901.07031.pdf)
- Densely Connected Convolutional Networks, Huang et al., 2018 [[arXiv:1608.06993v5]](https://arxiv.org/pdf/1608.06993.pdf)
- Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, Selvaraju et al., 2019 [[arXiv:1610.02391v4]](https://arxiv.org/pdf/1610.02391.pdf)
- Communication-Efficient Learning of Deep Networks from Decentralized Data, McMahan et al., 2017 [[arXiv:1602.05629v3]](https://arxiv.org/pdf/1602.05629.pdf)
- CheXpert model: [Github repository](https://github.com/gaetandi/cheXpert)
- Grad-CAM implementaion: [Github repository](https://github.com/ooodmt/MLMIP.git)
