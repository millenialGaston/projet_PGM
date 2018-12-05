#!/usr/bin/env python

"""
Project for IFT6269.
"""

__authors__ = "Jimmy Leroux, Nicolas Laliberte, Frederic Boileau"
__version__ = "1.0"
__maintainer__ = "Jimmy Leroux, Nicolas Laliberte, Frederic Boileau"
__email__ = "jim.leroux1@gmail.com, n.laliberte01@gmail.com, "
__studentid__ = "1024610, 1005803, "

import textGenerator as tg
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
    dataset = [dataset[i].translate(
        str.maketrans("","",string.punctuation)) for i in range(len(dataset))]
    dataset = list(filter(('').__ne__,dataset))

  return dataset

def hardCode():
  dataset = fetchData("hp","txt", filtering=False)
  dataset2 = fetchData("returnoftheking","txt", filtering=False)
  dataset3 = fetchData("QUOTE","csv", filtering=False)
  dataset4 = fetchData("shakes","txt", filtering=False)

  return dataset, dataset2, dataset3, dataset4

def plotting(loss_train,loss_test,loss_cross):
  plt.style.use('ggplot')
  plt.rc('xtick', labelsize=15)
  plt.rc('ytick', labelsize=15)
  plt.rc('axes', labelsize=15)
  plt.figure()
  plt.plot(loss_train, 'sk-',label='Trainset')
  plt.plot(loss_test, 'sr-', label='Testset')
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
  # CREATE THE DICTIONARIES ----------------------------------------------
  torch.cuda.manual_seed(10)
  dataset,dataset2,dataset3,dataset4 = hardCode()
  target_vocab = list(set(dataset+dataset2+dataset3+dataset4))
  t_vocab = {k:v for v,k in enumerate(target_vocab)}
  print("datasetlen:",len(dataset),len(dataset2),len(dataset3),len(dataset4))
  print("Total Vocab Len:",len(t_vocab))
  # ----------------------------------------------------------------------
  # TRAIN CLASSIFIER -----------------------------------------------------
  d,l=tg.create_class_data([dataset,dataset2,dataset3,dataset4],
    t_vocab,100,100000)
  rnnParams = RNN_Parameters(input_size=len(target_vocab),
                             hidden_size=256,
                             output_size=4)
  classifier = tg.sequence_classifier(device, *rnnParams).to(device)
  modelParams = [classifier,device,(d,l),
  	t_vocab,target_vocab]
  numericalParams = Numerical_Parameters(
    num_epoch = 20,
    sequence_size = 100,
    batch_size = 32,
    lr = 0.0001)

  loss_train, loss_test = \
    tg.train(*modelParams, *numericalParams, mode="classification")

  #plotting(loss_train, loss_test)
  # -----------------------------------------------------------------------
  # TRAIN THE MODELS ------------------------------------------------------
  # -------- HP -----------------------------------------------------------
  rnnParams = RNN_Parameters(input_size=len(target_vocab),
    						hidden_size=256,
    						output_size=len(target_vocab))
  hpmodel = tg.RNN(device, *rnnParams).to(device)
  modelParams = [hpmodel,device,dataset,
    t_vocab,target_vocab]
  numericalParams = Numerical_Parameters(
    num_epoch = 5,
    sequence_size = 100,
    batch_size = 16,
    lr = 0.01)
 
  loss_train, loss_test = \
    tg.train(*modelParams, *numericalParams, mode="textgen")
  # -------- lotr ----------------------------------------------------------
  rnnParams = RNN_Parameters(input_size=len(target_vocab),
    						hidden_size=256,
    						output_size=len(target_vocab))
  lotrmodel = tg.RNN(device, *rnnParams).to(device)
  modelParams = [lotrmodel,device,dataset2,
    t_vocab,target_vocab]
  numericalParams = Numerical_Parameters(
    num_epoch = 5,
    sequence_size = 100,
    batch_size = 16,
    lr = 0.01)
 
  loss_train, loss_test = \
    tg.train(*modelParams, *numericalParams, mode="textgen")
 # --------- quote ---------------------------------------------------------
  rnnParams = RNN_Parameters(input_size=len(target_vocab),
    						hidden_size=256,
    						output_size=len(target_vocab))
  quotemodel = tg.RNN(device, *rnnParams).to(device)
  modelParams = [quotemodel,device,dataset3,
    t_vocab,target_vocab]
  numericalParams = Numerical_Parameters(
    num_epoch = 5,
    sequence_size = 100,
    batch_size = 16,
    lr = 0.01)
 
  loss_train, loss_test = \
    tg.train(*modelParams, *numericalParams, mode="textgen")
  # -------- shakes --------------------------------------------------------
  rnnParams = RNN_Parameters(input_size=len(target_vocab),
    						hidden_size=256,
    						output_size=len(target_vocab))
  shakesmodel = tg.RNN(device, *rnnParams).to(device)
  modelParams = [shakesmodel,device,dataset4,
    t_vocab,target_vocab]
  numericalParams = Numerical_Parameters(
    num_epoch = 5,
    sequence_size = 100,
    batch_size = 16,
    lr = 0.01)
 
  loss_train, loss_test = \
    tg.train(*modelParams, *numericalParams, mode="textgen")
  # -------------------------------------------------------------------------
  models = [hpmodel, lotrmodel, quotemodel, shakesmodel]
  d,l = tg.create_texgen_data(models, device, target_vocab, t_vocab,100,400)
  tg.evaluate_texgen(classifier, device, (d,l),100, 16)
  plt.show()
if __name__ == '__main__':
  main()
