# CheXpert: A Large Chest X-Ray Dataset and Competitions
> CheXpert is a large dataset of chest X-rays and competition for automated chest X-ray interpretation, which features uncertainty labels and radiologist-labeled reference standard evaluation sets. *- from the CheXpert [webpage](https://stanfordmlgroup.github.io/competitions/chexpert/) description*

This repository aims to reproduce and improve CheXpert [paper](https://arxiv.org/pdf/1901.07031.pdf)'s results using the PyTorch library.
For the baseline, I tried to use the same model(DenseNet121) and the same parameters as the paper.

Since I know my results are still far short of the paper's, I will do many experiments from now on. If you have any ideas or suggestions to improve the model, please let me know!

This repository especially referenced [here](https://github.com/gaetandi/cheXpert) for the coding part. You can easily run the code with the following instructions.

This code is written for the GPU environment. It could be hard to run in CPU environment.



# 0. Prerequisites
- Python 3.6+
- PyTorch (1.7.1 in my case)
- Git Bash (whatever can handle git)

Open Git Bash and use `git clone` command to download this repository.

```bash
git clone https://github.com/Stomper10/CheXpert.git
```



# 1. Download the Data
At the bottom of the CheXpert [webpage](https://stanfordmlgroup.github.io/competitions/chexpert/), write a registration form to download the CheXpert dataset.
You will receive an email with the download link. Right-click your mouse on the download link(439GB or 11GB) and click 'Copy link address'.

Then, open Git Bash to run the following command. Paste your link address inside the double quotation marks(PLEASE RUN WITH QUOTATION MARKS). It will take a while even for the downsampled(11GB) dataset. In this experiment, I used the downsampled dataset.

```bash
wget "link_address_you_copied"
```

After you downloaded the dataset, run the following command to unzip the `.zip` file.

```bash
unzip CheXpert-v1.0-small.zip
```

Now the dataset is ready. As you see this repository structure, you have to place the `CheXpert-v1.0-small` directory and the rest files(`.py` and `.ipynb`) at the same location unless you modify the path in the source code.



# 2. Run the CheXpert
## Data Preprocessing
You MUST run the following command before running the model. In the current version, I set the model to use only frontal images.
```bash
python3 run_preprocessing.py
```

## Run the Model
Now you can run the model like below. **But**, I recommend you to use several options to run the model more efficiently.
```bash
python3 run_chexpert.py
```

Options | Shortcut | Description
:-: | :-: | :-:
--policy | -p | Uncertain label policy.
--ratio | -r | Training data ratio.
--output_path| -o | Path to save results.
--random_seed | -s | Random seed for reproduction.

If you want to use 1% of training set to train the model with `policy = ones`, run like below.
```bash
python3 run_chexpert.py \
  --policy=ones \
  --ratio=0.01 \
  --output_path=results/ \
  --random_seed=1
```

I also recommend you to use the `nohup` command if you run this code on server since it takes several hours.
```bash
nohup python3 run_chexpert.py > result.txt &
```

### This part is optional
* You can try deep ensembles with `run_ensembles.py` file.
Under the `ensembles` directory, place experiment output directories you want to aggregate.
When running the `run_chexpert.py`, set `--output_path` under the ensembles directory.
`run_ensembles.py` have `--policy` and `--output_path` options just like `run_chexpert.py`. The ensemble results will be saved in `--output_path` you set.

* You can try federated learning using the `CheXpert_DenseNet121_FL.ipynb` file. You can modify federated learning hyperparameters. (arranged version will be provided)
* You can try the Grad-CAM method on test set with `Grad-CAM.ipynb` file after you get the trained model. (arranged version will be provided)



# 3. Results
You may get training and validation losses, as well as the test accuracy and ROC curves. You can also check the computational costs. Models(`*.pth.tar`), test set probabilities(`testPROB.txt`), and ROC curve(`ROC.png`) files will be saved in the `results` directory. If you run the code with `nohup` command, printed outputs will also be saved. Let me just show you the ROC curves here(100 simple ensembles).

![ROC_ensem_mean](https://user-images.githubusercontent.com/43818471/103856596-408c9a80-50f8-11eb-9be5-41b38847998f.png)

The following table shows a comparison with original paper results.

* Stanford Baseline(ensemble) AUC = 0.907
* My Baseline(ensemble) AUC = 0.790 (Note that test set is different!)

Observation | Experiment AUC | Paper AUC | Difference
:-: | :-: | :-: | :-:
Atelectasis | 0.74 | 0.85 | -0.11
Cardiomegaly | 0.87 | 0.90 | -0.03
Consolidation | 0.76 | 0.90 | -0.14
Edema | 0.84 | 0.92 | -0.08
Pleural Effusion | 0.90 | 0.97 | -0.07

For those who want to compare the running environment, mine was as below(used GPU server).
* Intel Xeon Silver 4216 CPU
* 512GB memory
* Four Nvidia Titan RTX GPUs

# 4. Task Lists
- [x] Use subset of training dataset(10%) to check computational costs and performances.
- [x] Adjust the number of training data per each class.
- [x] Use whole training dataset(frontal images).
- [x] Try federated learning technique.
- [x] Apply Grad-CAM method for localization.
- [x] Try simple ensembles.
- [ ] Use also lateral images for training.
- [ ] Try various models for ensembles.
- [ ] Use original dataset for training(~439GB).

# 5. References
- CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison, Irvin et al., 2019 [[arXiv:1901.07031]](https://arxiv.org/pdf/1901.07031.pdf)
- Densely Connected Convolutional Networks, Huang et al., 2018 [[arXiv:1608.06993v5]](https://arxiv.org/pdf/1608.06993.pdf)
- Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, Selvaraju et al., 2019 [[arXiv:1610.02391v4]](https://arxiv.org/pdf/1610.02391.pdf)
- Communication-Efficient Learning of Deep Networks from Decentralized Data, McMahan et al., 2017 [[arXiv:1602.05629v3]](https://arxiv.org/pdf/1602.05629.pdf)
- CheXpert implementaion: [Github repository](https://github.com/gaetandi/cheXpert)
- Grad-CAM implementaion: [Github repository](https://github.com/ooodmt/MLMIP.git)
