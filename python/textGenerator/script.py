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
Idebug = debugger.Pdb().set_trace

import textGenerator as tg

# To use GPU if available.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Containers
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

def fetchData(name : str, extension : str, filtering=False) -> str:
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

def localDataFetchDriver(toFetch: Text_Fetch_Parameters = None) -> List[str]:
  if toFetch is None:
    toFetch = [Text_Fetch_Parameters("shakes","txt",False)]
  data = [(p.name,fetchData(*p)) for p in toFetch]
  return data

def main(*args,**kwargs):

  torch.cuda.manual_seed(10)

  #data is list of tuples, the first entry is the
  #name of the text file wihout #extension and the
  #second is the raw text data.
  #Idealement meme quand on change de facon ou de sorte
  # de data set on garderait la structure List[Tuple(name, rawtext)]

  data : List[Tuple(str,str)] = localDataFetchDriver()
  Idebug()
  listOfRawData = [d[1] for d in data]
  target_vocab = list(set(reduce(operator.concat, listOfRawData)))
  t_vocab = {k:v for v,k in enumerate(target_vocab)}

  # Train GENERATORS-----------------------------------------------------
  for d in data :
    rnnParams = RNN_Parameters(len(target_vocab),256,len(target_vocab))
    model = tg.RNN(device, *rnnParams).to(device)
    fileCheck = Path('models/' + d[0])
    cached = fileCheck.exists()
    if cached:
      model.load_state_dict(torch.load('models/' + d[0]))
    else:
      modelParam = [model ,device, d[1] , t_vocab,target_vocab]
      numParam = Numerical_Parameters(5,100,16,0.01)
      loss_train, loss_test = tg.train(*modelParam, *numParam, mode="textgen")
      torch.save(model.state_dict(),'models/' + d[0])

  # TRAIN CLASSIFIER -----------------------------------------------------
  #rnnParams = RNN_Parameters(len(target_vocab), 256, 4)
  #dataTensor, labelsTensor = tg.create_class_data(data , t_vocab,100,100000)

  #classifier = tg.sequence_classifier(device, *rnnParams).to(device)
  #mp = [classifier,device, (dataTensor,labelsTensor), t_vocab, target_vocab]
  #numParam = Numerical_Parameters(10,100,32,0.0001)
  #loss_train, loss_test = tg.train(*mp, *numParam, mode="classification")

  ##Classify syntethic data
  ## -------------------------------------------------------------------------
  #models = [hpmodel, lotrmodel, quotemodel, shakesmodel]
  #d,l = tg.create_texgen_data(models, device, target_vocab, t_vocab,100,1600)
  #tg.evaluate_texgen(classifier, device, (d,l),100, 16)
  #plt.show()


def plotting(loss_train,loss_test,loss_cross):
  plt.style.use('ggplot')
  plt.rc('xtick', labelsize=25)
  plt.rc('ytick', labelsize=25)
  plt.rc('axes', labelsize=25)
  plt.figure()
  plt.plot(loss_train, 'sk-',label='Trainset')
  plt.plot(loss_test, 'sr-', label='Testset')
  plt.tight_layout()
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(fontsize=25)
  plt.show()

def cliParsing():
  # For later
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_epoch')
  parser.add_argument('--batch_size')
  parser.add_argument('--learning_rate')
  args = parser.parse_args()

if __name__ == '__main__':
  main()
