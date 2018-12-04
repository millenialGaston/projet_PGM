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
import nltk
from nltk.corpus import brown

import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

import unidecode
import string
import argparse
import random
from copy import deepcopy
from collections import namedtuple

Numerical_Parameters = namedtuple('Numerical_Parameters',
                                  'num_epoch sequence_size batch_size lr')
RNN_Parameters = namedtuple('RNN_Parameters',
                              'input_size hidden_size output_size')

# To use GPU if available.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def fetchData(name : str,extension : str, filtering=False):
  dataPath = 'data/'
  dataset = None
  setNeedsColumnParsing = {"QUOTE","shortjokes"}
  setRegularParsing = {"hp","shakes","returnoftheking"}
  fullPath = dataPath + name + '.' + extension

  if name in setNeedsColumnParsing:
    dataset = pd.read_csv(fullPath)
    dataset = ' '.join(
      dataset.values[:,1].tolist()).lower().split()

  elif name in setRegularParsing:
    with open(fullPath,'r') as  file:
      dataset = file.read()
    dataset = dataset.lower().split()

  # filtering
  if filtering == True:
    for datum in dataset:
      datum.translate(str.maketrans("","",string.punctuation))
    dataset = list(filter(('').__ne__,dataset))

  return dataset

def hardCode():
  dataset2 = fetchData("returnoftheking","txt", filtering=False)
  dataset = fetchData("hp","txt", filtering=False)

  numericalParams = Numerical_Parameters(
      num_epoch = 20,
      sequence_size = 100,
      batch_size = 32,
      lr = 0.005)

  return dataset, dataset2, numericalParams

def plotting(loss_train,loss_test,loss_cross):
  plt.style.use('ggplot')
  plt.rc('xtick', labelsize=15)
  plt.rc('ytick', labelsize=15)
  plt.rc('axes', labelsize=15)
  plt.figure()
  plt.plot(loss_train, 'sk-',label='Trainset')
  plt.plot(loss_test, 'sr-', label='Testset')
  plt.plot(loss_cross, 'sb-', label='crossset')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()

def cliParsing():
  # For later
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_epoch')
  parser.add_argument('--batch_size')
  parser.add_argument('--learning_rate')
  args = parser.parse_args()

def main(*args,**kwargs):

  torch.cuda.manual_seed(10)
  dataset, dataset2, numericalParams = hardCode()
  cross_dataset = dataset2
  target_vocab = list(set(dataset+dataset2))
  t_vocab = {k:v for v,k in enumerate(target_vocab)}
  dataset2 = fetchData("returnoftheking","txt", filtering=False)
  dataset = fetchData("hp","txt", filtering=False)
  t,l=textGenerator.create_class_data([dataset,dataset2],t_vocab,100,200000)
  print(t.shape,l.shape)
  rnnParams = RNN_Parameters(input_size=len(target_vocab),
                             hidden_size=256,
                             output_size=len(target_vocab))

  rnn = textGenerator.RNN(device, *rnnParams).to(device)

  modelParams = [rnn,device,dataset,
                    t_vocab,target_vocab,cross_dataset]
  loss_train, loss_test, loss_cross = \
    textGenerator.train(*modelParams, *numericalParams)

  plotting(loss_train, loss_test, loss_cross)

if __name__ == '__main__':
  main()
