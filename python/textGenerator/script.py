#!/usr/bin/env python

"""
Project for IFT6269.
"""

__authors__ = "Jimmy Leroux, Nicolas Laliberte, Frederic Boileau"
__version__ = "1.0"
__maintainer__ = "Jimmy Leroux, Nicolas Laliberte, Frederic Boileau"
__email__ = "jim.leroux1@gmail.com, n.laliberte01@gmail.com, "
__studentid__ = "1024610, 1005803, "

from pathlib import Path
import unidecode
import string
import argparse
import random
from copy import deepcopy

from collections import namedtuple
import dataclasses
from dataclasses import dataclass
from typing import List, Tuple

import operator
from functools import reduce
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from nltk.corpus import gutenberg as gut
from IPython.core import debugger
import re
Idebug = debugger.Pdb().set_trace

import textGenerator as tg

# To use GPU if available.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Numerical_Parameters = namedtuple('Numerical_Parameters',
                                  'num_epoch sequence_size batch_size lr')
RNN_Parameters = namedtuple('RNN_Parameters',
                              'input_size hidden_size output_size')
@dataclass
class Text_Fetch_Parameters:
  name : str
  ext  : str
  filtering : bool = False
  #Enables unpacking
  def __iter__(self):
        yield from dataclasses.astuple(self)


def main(*args,**kwargs):
  torch.cuda.manual_seed(10)
  save = False
  #beauty gut text files
  data, target_vocab, t_vocab = fetchGutData()
  models, losses = beautyTrainGenerator(data,target_vocab,t_vocab)

  #data : List[List[str]] = None
  #data,target_vocab,t_vocab = fetchUglyData()
  #print("Dictinary size: {}".format(len(t_vocab)))
  #for i,j in enumerate(data):
  #  print("Length dataset {}:{}".format(i,len(j)))
  #classifier, loss_train, loss_test = uglyTrainClassifier(data,target_vocab,t_vocab)
  #models, losses = uglyTrainGenerators(data,target_vocab,t_vocab)

  d,l = tg.create_texgen_data(models, device, target_vocab, t_vocab,100,1000)
  tg.evaluate_texgen(classifier, device, (d,l),100, 16)

  #Save
  if(save):
    saveModels(models,classifier)

  return models, target_vocab,t_vocab, losses

def fetchData(name : str, extension : str, filtering=False, charLevel=False) -> str:
  dataPath = 'data/'
  dataset = None
  setNeedsColumnParsing = {"QUOTE","shortjokes"}
  setRegularParsing = {"hpnew2","shakes","returnoftheking2"}
  fullPath = dataPath + name + '.' + extension

  if name in setNeedsColumnParsing:
    dataset = pd.read_csv(fullPath)
    dataset = ' '.join(
      dataset.values[:,1].tolist()).lower()

  elif name in setRegularParsing:
    with open(fullPath,'r') as  file:
      dataset = file.read()
    dataset = dataset.lower()

  # filtering
  if filtering and not charLevel:
    return nltk.word_tokenize(dataset)
  if filtering and charLevel:
    return [char for char in dataset.lower()]

  dataset = dataset.split()
  return dataset

def fetchGutData():
  names = []
  gut_names = gut.fileids()
  availableTexts = gut_names + ['hp','shakes','lotr','quotes']
  while not names:
    print("================================================\n")
    print("List of available text to train textGenerator:\n")
    print(availableTexts)
    print("\n\n" + "Enter filenames seperated by whitespaces : ")
    user_input = [str(x) for x in input().split()]
    for user_in in user_input:
      if user_in not in availableTexts:
        print("\n Error not found : " + user_in + "\n")

    names = list(set(user_input) and set(availableTexts))
    gut_choice = list(set(user_input) and set(gut_names))
    if not names:
      print("Error no text selected ===> Try again Please \n\n")

  print("==============================")
  print("OK thanks training started\n\n")

  char_data: List[List[str]] = [[w.lower() for w in gut.raw(name)]
                                           for name in gut_choice]
  uglyData, _, _ = fetchUglyData(charLevel=True)
  char_data += uglyData

  data = list(zip(names,char_data))
  target_vocab = list(set(reduce(operator.concat,char_data)))
  t_vocab = {k:v for v,k in enumerate(target_vocab)}

  return data, target_vocab, t_vocab

def fetchUglyData(charLevel=False):

  dat1 = fetchData("hpnew2","txt",True,charLevel)
  dat2 = fetchData("returnoftheking2","txt",True,charLevel)
  dat3 = fetchData("QUOTE","csv",True,charLevel)
  dat4 = fetchData("shakes","txt",True,charLevel)
  data = dat1+dat2+dat3+dat4
  target_vocab = list(set(data))
  t_vocab = {k:v for v,k in enumerate(target_vocab)}
  return [dat1,dat2,dat3,dat4], target_vocab, t_vocab

def beautyTrainClassifier(data,target_vocab,t_vocab):
  pass

def uglyTrainClassifier(data,target_vocab,t_vocab):
  rnnParams = RNN_Parameters(len(target_vocab), 256, 4)
  dataTensor, labelsTensor = tg.create_class_data(data,t_vocab,50,100000)

  classifier = tg.sequence_classifier(device, *rnnParams).to(device)
  mp = [classifier,device, (dataTensor,labelsTensor), t_vocab, target_vocab]
  numParam = Numerical_Parameters(5,50,64,0.0001)
  loss_train, loss_test = tg.train(*mp, *numParam, mode="classification")
  return classifier, loss_train, loss_test

def beautyTrainGenerator(data,target_vocab,t_vocab):
  rnnParams = RNN_Parameters(len(target_vocab), 512, len(target_vocab))
  numParam = Numerical_Parameters(1,50,64,0.1)
  models = list()
  losses = list()
  for d in data :
    models.append(tg.RNN(device, *rnnParams).to(device))
    fileCheck = Path('models/' + d[0])
    cached = fileCheck.exists()
    if cached:
      model.load_state_dict(torch.load('models/' + d[0]))
    else:
      modelParam = [models[-1],device, d[1] , t_vocab,target_vocab]
      l_train, l_test = tg.train(*modelParam, *numParam, mode="textgen")
      torch.save(model.state_dict(),'models/' + d[0] + '.model')
      losses.append((l_train,l_test))

  return models, losses

def uglyTrainGenerators(data,target_vocab,t_vocab):
  rnnParams = RNN_Parameters(len(target_vocab), 512, len(target_vocab))
  numParam = Numerical_Parameters(5,50,64,0.005)
  models = list()
  losses = list()
  for d in data:
    models.append(tg.RNN(device, *rnnParams).to(device))
    modelParam = [models[-1],device, d, t_vocab,target_vocab]
    l_train,l_test = tg.train(*modelParam, *numParam, mode="textgen")
    losses.append((l_train,l_test))
  return models, losses

def saveModels(models, classifier, isJim=False):
  if isJim:
    torch.save(models[0].state_dict(),'C:/Users/Jimmy/Desktop/' + 'hpmodel')
    torch.save(models[1].state_dict(),'C:/Users/Jimmy/Desktop/' + 'lotrmodel')
    torch.save(models[2].state_dict(),'C:/Users/Jimmy/Desktop/' + 'quotemodel')
    torch.save(models[3].state_dict(),'C:/Users/Jimmy/Desktop/' + 'shakesmodel')
    torch.save(classifier.state_dict(),'C:/Users/Jimmy/Desktop/' + 'classifier')
  else:
    torch.save(models[0].state_dict(),'models/' + 'hpmodel')
    torch.save(models[1].state_dict(),'models/' + 'lotrmodel')
    torch.save(models[2].state_dict(),'models/' + 'quotemodel')
    torch.save(models[3].state_dict(),'models/' + 'shakesmodel')
    torch.save(classifier.state_dict(),'models/' + 'classifier')

def localDataFetchDriver(toFetch: Text_Fetch_Parameters = None) -> List[Tuple[str,str]]:
  if toFetch is None:
    toFetch = [Text_Fetch_Parameters("shakes","txt",False)]
  data = [(p.name,fetchData(*p)) for p in toFetch]
  return data

def plotting(loss_train,loss_test):
  plt.style.use('ggplot')
  plt.rc('xtick', labelsize=25)
  plt.rc('ytick', labelsize=25)
  plt.rc('axes', labelsize=25)
  plt.figure()
  plt.plot(loss_train, 'sk-',label='Trainset')
  plt.plot(loss_test, 'sr-', label='Testset')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(fontsize=25)
  plt.show()

def cliParsing():
  # For later
  pass
  return args

if __name__ == '__main__':
  model,target_vocab,t_vocab, losses = main()
