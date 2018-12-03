#!/usr/bin/env python

"""
Project for IFT6269.
"""

__authors__ = "Jimmy Leroux, Nicolas Laliberte, Frederic Boileau"
__version__ = "1.0"
__maintainer__ = "Jimmy Leroux, Nicolas Laliberte, Frederic Boileau"
__email__ = "jim.leroux1@gmail.com, n.laliberte01@gmail.com, "
__studentid__ = "1024610, 1005803, "


import textGenerator
import numpy as np
import pandas as pd
import unidecode
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
import bcolz
import pickle
import torchvision

def main(*args,**kwargs):


    plt.style.use('ggplot')
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('axes', labelsize=15)

    # To use GPU if available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed(10)

    # load quotes approx 800 000 words
    #dataset = pd.read_csv('data/shortjokes.csv')
    #dataset = ' '.join(dataset.values[:,1].tolist()).lower().split()

    # load harry potter 600 000 words
    with open('hp.txt','r') as  file:
        dataset = file.read()
    dataset = dataset.lower().split()
    #dataset = [dataset[i].translate(
    #    str.maketrans("","",string.punctuation)) for i in range(len(dataset))]
    #dataset = list(filter(('').__ne__,dataset))

    # create the network.
    target_vocab = list(set(dataset))
    t_vocab = {k:v for v,k in enumerate(target_vocab)}

    rnn = textGenerator.RNN(device, len(target_vocab),256).to(device)

    loss_train, loss_test = textGenerator.train(rnn, device,dataset,t_vocab, num_epoch=20,
                                                sequence_size=100, batch_size=64, lr=0.01)

    plt.figure()
    plt.plot(loss_train, 'sk-',label='Trainset')
    plt.plot(loss_test, 'sr-', label='Testset')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
  main()
