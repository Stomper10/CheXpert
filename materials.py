'''Material classes for 'run_chexpert.py'''

###################
## Prerequisites ##
###################
import time
import pickle
import random
import csv
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score

use_gpu = torch.cuda.is_available()



######################
## Create a Dataset ##
######################
class CheXpertDataSet(Dataset):
    def __init__(self, data_PATH, nnClassCount, transform = None):
        """
        data_PATH: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels.
        """
        image_names = []
        labels = []

        with open(data_PATH, 'r') as f:
            csvReader = csv.reader(f)
            next(csvReader, None) # skip the header
            for line in csvReader:
                image_name = line[0]
                if nnClassCount == 5:
                    npline = np.array(line)
                    idx = [7, 10, 11, 13, 15]
                    label = list(npline[idx])
                    for i in range(nnClassCount):
                        if label[i]:
                            a = float(label[i])
                            if a == 1:
                                label[i] = 1
                            elif a == -1:
                                if i == 1 or i == 3 or i == 4:  # Atelectasis, Edema, Pleural Effusion
                                    label[i] = 1                    # U-Ones
                                elif i == 0 or i == 2:          # Cardiomegaly, Consolidation
                                    label[i] = 0                    # U-Zeroes
                            else:
                                label[i] = 0
                        else:
                            label[i] = 0
                else:
                    label = label[5:]
                    for i in range(nnClassCount):
                        if label[i]:
                            a = float(label[i])
                            if a == 1:
                                label[i] = 1
                            elif a == -1:
                                if i == 5 or i == 8 or i == 10:  # Atelectasis, Edema, Pleural Effusion
                                    label[i] = 1                    # U-Ones
                                elif i == 2 or i == 6:          # Cardiomegaly, Consolidation
                                    label[i] = 0                    # U-Zeroes
                            else:
                                label[i] = 0
                        else:
                            label[i] = 0
                        
                image_names.append('./' + image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        '''Take the index of item and returns the image and its labels'''
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)


############################
## Create CheXpertTrainer ##
############################
class CheXpertTrainer():
    def train(model, dataLoaderTrain, dataLoaderVal, nnClassCount, trMaxEpoch, PATH, f_or_l, checkpoint, cfg):
        optimizer = optim.Adam(model.parameters(), lr = cfg.lr, # setting optimizer & scheduler
                               betas = tuple(cfg.betas), eps = cfg.eps, weight_decay = cfg.weight_decay) 
        loss = torch.nn.BCELoss() # setting loss function
        
        if checkpoint != None and use_gpu: # loading checkpoint
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
            
        # Train the network
        lossMIN, lossMIN_Card, lossMIN_Edem, lossMIN_Cons, lossMIN_Atel, lossMIN_PlEf = 100000, 100000, 100000, 100000, 100000, 100000
        Card_traj, Edem_traj, Cons_traj, Atel_traj, PlEf_traj = [], [], [], [], []
        train_start = []
        train_end = []
        print('<<< Training & Evaluating ({}) >>>'.format(f_or_l))
        for epochID in range(0, trMaxEpoch):
            train_start.append(time.time()) # training starts
            losst = CheXpertTrainer.epochTrain(model, dataLoaderTrain, optimizer, trMaxEpoch, nnClassCount, loss)
            train_end.append(time.time())   # training ends
            lossv, lossv_Card, lossv_Edem, lossv_Cons, lossv_Atel, lossv_PlEf = CheXpertTrainer.epochVal(model, dataLoaderVal, optimizer, trMaxEpoch, nnClassCount, loss)

            Card_traj.append(lossv_Card.float())
            Edem_traj.append(lossv_Edem.float())
            Cons_traj.append(lossv_Cons.float())
            Atel_traj.append(lossv_Atel.float())
            PlEf_traj.append(lossv_PlEf.float())

            print("Training loss: {:.3f},".format(losst), "Valid loss: {:.3f}".format(lossv))
            
            if lossv < lossMIN:
                lossMIN = lossv
                model_num = epochID + 1
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 
                            'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 
                            '{0}m-epoch_{1}_{2}.pth.tar'.format(PATH, epochID + 1, f_or_l))
                print('Epoch ' + str(epochID + 1) + ' [save] loss = ' + str(lossv))
            else:
                print('Epoch ' + str(epochID + 1) + ' [----] loss = ' + str(lossv))

            if lossv_Card < lossMIN_Card:
                lossMIN_Card = lossv_Card
                model_num_Card = epochID + 1
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 
                            'best_loss': lossMIN_Card, 'optimizer' : optimizer.state_dict()}, 
                            '{0}m-epoch_{1}_{2}_Card.pth.tar'.format(PATH, epochID + 1, f_or_l))
                print('Epoch ' + str(epochID + 1) + ' [save] loss Card = ' + str(lossv_Card))
            else:
                print('Epoch ' + str(epochID + 1) + ' [----] loss Card = ' + str(lossv_Card))

            if lossv_Edem < lossMIN_Edem:
                lossMIN_Edem = lossv_Edem
                model_num_Edem = epochID + 1
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 
                            'best_loss': lossMIN_Edem, 'optimizer' : optimizer.state_dict()}, 
                            '{0}m-epoch_{1}_{2}_Edem.pth.tar'.format(PATH, epochID + 1, f_or_l))
                print('Epoch ' + str(epochID + 1) + ' [save] loss Edem = ' + str(lossv_Edem))
            else:
                print('Epoch ' + str(epochID + 1) + ' [----] loss Edem = ' + str(lossv_Edem))

            if lossv_Cons < lossMIN_Cons:
                lossMIN_Cons = lossv_Cons
                model_num_Cons = epochID + 1
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 
                            'best_loss': lossMIN_Cons, 'optimizer' : optimizer.state_dict()}, 
                            '{0}m-epoch_{1}_{2}_Cons.pth.tar'.format(PATH, epochID + 1, f_or_l))
                print('Epoch ' + str(epochID + 1) + ' [save] loss Cons = ' + str(lossv_Cons))
            else:
                print('Epoch ' + str(epochID + 1) + ' [----] loss Cons = ' + str(lossv_Cons))

            if lossv_Atel < lossMIN_Atel:
                lossMIN_Atel = lossv_Atel
                model_num_Atel = epochID + 1
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 
                            'best_loss': lossMIN_Atel, 'optimizer' : optimizer.state_dict()}, 
                            '{0}m-epoch_{1}_{2}_Atel.pth.tar'.format(PATH, epochID + 1, f_or_l))
                print('Epoch ' + str(epochID + 1) + ' [save] loss Atel = ' + str(lossv_Atel))
            else:
                print('Epoch ' + str(epochID + 1) + ' [----] loss Atel = ' + str(lossv_Atel))

            if lossv_PlEf < lossMIN_PlEf:
                lossMIN_PlEf = lossv_PlEf
                model_num_PlEf = epochID + 1
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 
                            'best_loss': lossMIN_PlEf, 'optimizer' : optimizer.state_dict()}, 
                            '{0}m-epoch_{1}_{2}_PlEf.pth.tar'.format(PATH, epochID + 1, f_or_l))
                print('Epoch ' + str(epochID + 1) + ' [save] loss PlEf = ' + str(lossv_PlEf))
            else:
                print('Epoch ' + str(epochID + 1) + ' [----] loss PlEf = ' + str(lossv_PlEf))

        train_time = np.array(train_end) - np.array(train_start)

        traj_all = [Card_traj, Edem_traj, Cons_traj, Atel_traj, PlEf_traj]
        names = ['Card', 'Edem', 'Cons', 'Atel', 'PlEf']
        xlab = list(range(1, trMaxEpoch + 1))
        
        if nnClassCount == 5:
            fig, ax = plt.subplots(nrows = 1, ncols = 5)
            fig.set_size_inches((50, 10))
            for i in range(nnClassCount):
                ax[i].plot(xlab, traj_all[i])
                ax[i].set_title('Valid loss trajectory: ' + names[i])
                ax[i].set_xlim([0, trMaxEpoch + 1])
                ax[i].set_xticks(np.arange(1, trMaxEpoch + 1, step = 1))
                ax[i].set_ylim([0, 1])
                ax[i].set_ylabel('Valid loss')
                ax[i].set_xlabel('Epoch Number')

            plt.savefig('{0}{1}_traj_all.png'.format(PATH, f_or_l), dpi = 100)
            plt.close()
        print('')

        return model_num, model_num_Card, model_num_Edem, model_num_Cons, model_num_Atel, model_num_PlEf, train_time
       
        
    def epochTrain(model, dataLoaderTrain, optimizer, epochMax, classCount, loss):
        model.train()
        losstrain = 0

        for batchID, (varInput, target) in enumerate(dataLoaderTrain):
            optimizer.zero_grad()
            
            varTarget = target.cuda(non_blocking = True)
            varOutput = model(varInput)
            lossvalue = loss(varOutput, varTarget)
                       
            lossvalue.backward()
            optimizer.step()
            
            losstrain += lossvalue.item()
            if batchID % 1000 == 999:
                print('[Batch: %5d] loss: %.3f'%(batchID + 1, losstrain / 2000))
            
        return losstrain / len(dataLoaderTrain)
    
    
    def epochVal(model, dataLoaderVal, optimizer, epochMax, classCount, loss):
        model.eval()
        lossVal = 0
        
        lossVal_Card, lossVal_Edem, lossVal_Cons, lossVal_Atel, lossVal_PlEf = 0, 0, 0, 0, 0

        with torch.no_grad():
            for i, (varInput, target) in enumerate(dataLoaderVal):
                
                target = target.cuda(non_blocking = True)
                varOutput = model(varInput)

                varOutput_Card = torch.tensor([i[0] for i in varOutput.tolist()])
                target_Card = torch.tensor([i[0] for i in target.tolist()])
                varOutput_Edem = torch.tensor([i[1] for i in varOutput.tolist()])
                target_Edem = torch.tensor([i[1] for i in target.tolist()])
                varOutput_Cons = torch.tensor([i[2] for i in varOutput.tolist()])
                target_Cons = torch.tensor([i[2] for i in target.tolist()])
                varOutput_Atel = torch.tensor([i[3] for i in varOutput.tolist()])
                target_Atel = torch.tensor([i[3] for i in target.tolist()])
                varOutput_PlEf = torch.tensor([i[4] for i in varOutput.tolist()])
                target_PlEf = torch.tensor([i[4] for i in target.tolist()])

                lossVal += loss(varOutput, target)
                lossv = lossVal / len(dataLoaderVal)

                lossVal_Card += loss(varOutput_Card, target_Card)
                lossv_Card = lossVal_Card / len(dataLoaderVal)
                lossVal_Edem += loss(varOutput_Edem, target_Edem)
                lossv_Edem = lossVal_Edem / len(dataLoaderVal)
                lossVal_Cons += loss(varOutput_Cons, target_Cons)
                lossv_Cons = lossVal_Cons / len(dataLoaderVal)
                lossVal_Atel += loss(varOutput_Atel, target_Atel)
                lossv_Atel = lossVal_Atel / len(dataLoaderVal)
                lossVal_PlEf += loss(varOutput_PlEf, target_PlEf)
                lossv_PlEf = lossVal_PlEf / len(dataLoaderVal)
                                
        return lossv, lossv_Card, lossv_Edem, lossv_Cons, lossv_Atel, lossv_PlEf

    
    def computeAUROC(dataGT, dataPRED, classCount):
        # Computes area under ROC curve 
        # dataGT: ground truth data
        # dataPRED: predicted data
        outAUROC = []
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            try:
                outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            except ValueError:
                pass
        return outAUROC
    
    
    def test(model, dataLoaderTest, nnClassCount, checkpoint, class_names, f_or_l):
        cudnn.benchmark = True
        
        if checkpoint != None and use_gpu:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])

        if use_gpu:
            outGT = torch.FloatTensor().cuda()
            outPRED = torch.FloatTensor().cuda()
        else:
            outGT = torch.FloatTensor()
            outPRED = torch.FloatTensor()
       
        model.eval()
        
        outPROB = []
        with torch.no_grad():
            for i, (input, target) in enumerate(dataLoaderTest):

                target = target.cuda()
                outGT = torch.cat((outGT, target), 0).cuda()
                outProb = model(input) # probability
                outProb = outProb.tolist()
                outPROB.append(outProb)

                bs, c, h, w = input.size()
                varInput = input.view(-1, c, h, w)
                
                out = model(varInput)
                outPRED = torch.cat((outPRED, out), 0)
        aurocIndividual = CheXpertTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        print('<<< Model Test Results ({}) >>>'.format(f_or_l))
        print('AUROC mean ', aurocMean)
        
        for i in range (0, len(aurocIndividual)):
            print(class_names[i], ' ', aurocIndividual[i])
        print('')
        return outGT, outPRED, outPROB, aurocMean, aurocIndividual



##################
## Define Model ##
##################
class DenseNet121(nn.Module):
    '''Model modified.
    The architecture of this model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    '''
    def __init__(self, out_size, nnIsTrained):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained = nnIsTrained)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x



###############################
## Define Ensembles Function ##
###############################
def EnsemAgg(EnsemResult, dataLoader, nnClassCount, class_names):
    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()
    with torch.no_grad():
        for i, (input, target) in enumerate(dataLoader):
            target = target.cuda()
            outGT = torch.cat((outGT, target), 0).cuda()
            
            bs, c, h, w = input.size()
            varInput = input.view(-1, c, h, w)

            # out = model(varInput)
            out = torch.tensor([EnsemResult[i]]).cuda()
            outPRED = torch.cat((outPRED, out), 0)
    aurocIndividual = CheXpertTrainer.computeAUROC(outGT, outPRED, nnClassCount)
    aurocMean = np.array(aurocIndividual).mean()
    print('<<< Ensembles Test Results >>>')
    print('AUROC mean ', aurocMean)

    for i in range (0, len(aurocIndividual)):
        print(class_names[i], ' ', aurocIndividual[i])
    print('')

    return outGT, outPRED, aurocMean, aurocIndividual