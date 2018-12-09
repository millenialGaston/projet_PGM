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
  setRegularParsing = {"newhp","shakes","returnoftheking"}
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
  if filtering == True:
    #punc = [p for p in string.punctuation]
    #dataset = [dataset[i].translate(
    #    str.maketrans("","",string.punctuation)) for i in range(len(dataset))]
    #dataset = list(filter(('').__ne__,dataset)) + punc
    return nltk.word_tokenize(dataset)
  dataset = dataset.split()
  return dataset

def localDataFetchDriver(toFetch: Text_Fetch_Parameters = None) -> List[str]:
  if toFetch is None:
    toFetch = [Text_Fetch_Parameters("shakes","txt",False)]
  data = [(p.name,fetchData(*p)) for p in toFetch]
  return data

def main(*args,**kwargs):

  torch.cuda.manual_seed(10)

# UNCOMMENT FOR GENERALIZZZZZEDDDDDDD  STUFFFF -------------------------------
  #data : List[Tuple(str,str)] = localDataFetchDriver()


  # Data selection ---------------------------------------------------
  # names = []
  # gut_names = gut.fileids()
  # while not names:
  #   print("================================================\n")
  #   print("List of available text to train textGenerator:\n")
  #   print(gut_names)
  #   print("\n\n" + "Enter filenames seperated by whitespaces : ")
  #   user_input = [str(x) for x in input().split()]
  #   for user_in in user_input:
  #     if user_in not in gut_names:
  #       print("\n Error not found : " + user_in + "\n")

  #   names = list(set(user_input) & set(gut_names))
  #   if not names:
  #     print("Error no text selected ===> Try again Please \n\n")

  # print("==============================")
  # print("OK thanks training started\n\n")
  # word_data: List[List[str]] = [[w.lower() for w in gut.words(name)]
  #                                          for name in names]
  # data = list(zip(names,word_data))
  #target_vocab = list(set(reduce(operator.concat,word_data)))
  #t_vocab = {k:v for v,k in enumerate(target_vocab)}


  # Train GENERATORS-----------------------------------------------------
  # for d in data :
  #   Idebug()
  #   rnnParams = RNN_Parameters(len(target_vocab),256,len(target_vocab))
  #   model = tg.RNN(device, *rnnParams).to(device)
  #   fileCheck = Path('models/' + d[0])
  #   cached = fileCheck.exists()
  #   if cached:
  #     model.load_state_dict(torch.load('models/' + d[0]))
  #   else:
  #     modelParam = [model ,device, d[1] , t_vocab,target_vocab]
  #     numParam = Numerical_Parameters(1,20,16,0.01)
  #     loss_train, loss_test = tg.train(*modelParam, *numParam, mode="textgen")
  #     torch.save(model.state_dict(),'models/' + d[0])

  #   print(tg.evaluate(model,device,target_vocab, t_vocab,'i', 40))
  # -------------------------------------------------------------------------------
  
  # UGLY STUFF BUT WORKING FOR NOW -----------------------------------------------
  dat1 = fetchData("newhp","txt",True)
  dat2 = fetchData("returnoftheking","txt",True)
  dat3 = fetchData("QUOTE","csv",True)
  dat4 = fetchData("shakes","txt",True)
  data = dat1+dat2+dat3+dat4
  print(len(dat1),len(dat2),len(dat3),len(dat4))
  target_vocab = list(set(dat1))
  t_vocab = {k:v for v,k in enumerate(target_vocab)}

  # TRAIN CLASSIFIER -----------------------------------------------------
  rnnParams = RNN_Parameters(len(target_vocab), 256, 4)

  #dataTensor, labelsTensor = tg.create_class_data([dat1,dat2,dat3,dat4], t_vocab,100,100000)

  #classifier = tg.sequence_classifier(device, *rnnParams).to(device)
  #mp = [classifier,device, (dataTensor,labelsTensor), t_vocab, target_vocab]
  #numParam = Numerical_Parameters(1,100,32,0.0001)
  #loss_train, loss_test = tg.train(*mp, *numParam, mode="classification")

  # TRAIN MODELS
  rnnParams = RNN_Parameters(len(target_vocab), 256, len(target_vocab))
  numParam = Numerical_Parameters(10,50,64,0.005)
  hpmodel = tg.RNN(device, *rnnParams).to(device)
  lotrmodel = tg.RNN(device, *rnnParams).to(device)
  quotemodel = tg.RNN(device, *rnnParams).to(device)
  shakesmodel = tg.RNN(device, *rnnParams).to(device)
  modelParam = [hpmodel ,device, dat1 , t_vocab,target_vocab]
  _,_ = tg.train(*modelParam, *numParam, mode="textgen")
  modelParam = [lotrmodel ,device, dat2 , t_vocab,target_vocab]
  _,_ = tg.train(*modelParam, *numParam, mode="textgen")
  modelParam = [quotemodel ,device, dat3 , t_vocab,target_vocab]
  _,_ = tg.train(*modelParam, *numParam, mode="textgen")
  modelParam = [shakesmodel ,device, dat4 , t_vocab,target_vocab]
  _,_ = tg.train(*modelParam, *numParam, mode="textgen")
  ##Classify syntethic data
  ## -------------------------------------------------------------------------
  models = [hpmodel, lotrmodel, quotemodel, shakesmodel]
  d,l = tg.create_texgen_data(models, device, target_vocab, t_vocab,100,10)
  tg.evaluate_texgen(classifier, device, (d,l),100, 16)
  plt.show()


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
