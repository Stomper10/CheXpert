# CheXpert: A Large Chest X-Ray Dataset and Competitions
> CheXpert is a large dataset of chest X-rays and competition for automated chest X-ray interpretation, which features uncertainty labels and radiologist-labeled reference standard evaluation sets. *- from CheXpert <a href="https://stanfordmlgroup.github.io/competitions/chexpert/" target="_blank">webpage</a> description*

This repository aims to reproduce CheXpert <a href="https://arxiv.org/pdf/1901.07031.pdf" target="_blank">paper</a>'s results using PyTorch library.
So I tried to use same model(DenseNet121) and the parameters as the paper.

This repository especially referenced <a href="https://github.com/gaetandi/cheXpert" target="_blank">here</a> for the coding part.  
You can easily run the code with the following instructions.

# 0. Prerequisites
- Python3
- Git Bash

Open Git Bash ans use `git clone` command to download this repository.
```bash
git clone https://github.com/Stomper10/CheXpert.git
```

# 1. Download Data
At the bottom of CheXpert <a href="https://stanfordmlgroup.github.io/competitions/chexpert/" target="_blank">webpage</a>, write a registration form to download the CheXpert dataset.
You will receive a link to the download over email. Right click your mouse on the download link(439GB or 11GB) and click 'Copy link address'.

Then, open Git Bash to run the following command. Paste your link address inside the double quotation marks(RUN WITH QUOTATION MARKS). It will take a while even for downsampled(11GB) dataset. In this experiment, I used downsampled dataset.
```bash
wget "link_address_you_copied"
```

After you downloaded the dataset, run the following command to unzip the `.zip` file.
```bash
unzip CheXpert-v1.0-small.zip
```

Now, dataset is ready. You have to place `CheXpert-v1.0-small` directory and `CheXpert_DenseNet121.ipynb` file(or `CheXpert_DenseNet121.py`) at the same location unless you modify the path in the source code.

# 2. Run the Code
If you want to use `.ipynb` file, run `CheXpert_DenseNet121.ipynb` file. If you want `.py` file instead, `CheXpert_DenseNet121.py` is also ready for you. Same contents with different file extension. Maybe you need to install `PyTorch` and `Bar` packages before you run the code.  

For those who want to compare, my running environment was as below(used GPU server).
* Intel Xeon Silver 4216 CPU
* 512GB memory
* Nvidia Titan RTX GPU

# 3. Results
You may get training & validation loss and ROC curve for results. You can also check the computational cost. Results are saved in `Result` folder in this repository. Let me just show you ROC curve here.
![Alt text]()

# 4. References
- CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison, Irvin, Jeremy, et al., 2019 <a href="https://arxiv.org/pdf/1901.07031.pdf" target="_blank">[Arxiv:1901.07031]</a>
- Densely Connected Convolutional Networks, Huang et al., 2018 <a href="https://arxiv.org/pdf/1608.06993.pdf" target="_blank">[Arxiv:1608.06993]</a>
- <a href="https://github.com/gaetandi/cheXpert" target="_blank">Github repository</a>
