'''
Code written by: Xiaoqing Liu
If you use significant portions of this code or the ideas from our paper, please cite it :)
'''

import os
import random
import time
import numpy as np
import torch
import math
import torch.nn as nn
from PIL import Image, ImageOps
from argparse import ArgumentParser

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage

from dataset import train,test
from transform import Relabel, ToLabel, Colorize

import importlib
from iouEval import iouEval, getColorEntry

from shutil import copyfile
from edanet import EDANet

NUM_CHANNELS = 3
NUM_CLASSES = 2 #pascal=22, cityscapes=20

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()


image_transform = ToPILImage()
input_transform = Compose([
    Resize((512,512)),
    #CenterCrop(256),
    ToTensor(),
    #Normalize([112.65,112.65,112.65],[32.43,32.43,32.43])
    #Normalize([.485, .456, .406], [.229, .224, .225]),
])
target_transform = Compose([
    Resize((512,512)),
    #CenterCrop(324),
    ToLabel(),
    #Relabel(255, 1),
])

class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d,self).__init__()

        self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)


def train( model):
    best_acc = 0
    num_epochs=60
     
    loader = DataLoader(train(input_transform, target_transform),num_workers=1, batch_size=4, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    savedir = './save'

    automated_log_path = savedir + "/log.txt"
    modeltxtpath = savedir + "/model.txt"

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))

    optimizer = Adam(model.parameters(), 1e-4, (0.9, 0.999),  eps=1e-08, weight_decay=2e-4)     ## scheduler 1
    start_epoch = 1
    lr_updater = lr_scheduler.StepLR(optimizer, 100,
                                     0.1)                             ## scheduler 2
        #Note: this only loads initialized weights. If you want to resume a training use "--resume" option!!
    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict keys are there
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            own_state[name].copy_(param)
        return model

    for epoch in range(start_epoch, num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        lr_updater.step()

        epoch_loss = []
        time_train = []

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        for step, (images, labels) in enumerate(loader):

            start_time = time.time()
            
            images = images.cuda()
            labels = labels.cuda()

            inputs = Variable(images)
            targets = Variable(labels)
            #print inputs.size(),targets.size()
            outputs = model(inputs)
 
            #print outputs.size()
            optimizer.zero_grad()
            loss = criterion(outputs, targets[:,0])
            loss.backward()
            optimizer.step()
            #print loss.item()
            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)
     
            if step % 100 == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print('epoch:%f'%epoch,'step:%f'%step,'loss:%f'%average)

            with open(automated_log_path, "a") as myfile:
                myfile.write("\n%d\t\t%d\t\t%.4f" % (epoch, step,average ))
        if epoch % 1 == 0 and epoch != 0:

            filename = 'main-'+'eda'+'-step-'+str(step)+'-epoch-'+str(epoch)+'.pth'
            torch.save(model.state_dict(), './save/model/'+filename)
    return(model) 

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)

def main():
    savedir = './save'

    Net = EDANet(NUM_CLASSES)
    Net = Net.cuda()
    train(Net)

if __name__ == '__main__':
    main()
