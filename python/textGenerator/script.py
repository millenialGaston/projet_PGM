#!/usr/bin/env python

"""
Project for IFT6269.
"""

__authors__ = "Jimmy Leroux, Nicolas Laliberte, Frederic Boileau"
__version__ = "1.0"
__maintainer__ = "Jimmy Leroux, Nicolas Laliberte, Frederic Boileau"
__email__ = "jim.leroux1@gmail.com, n.laliberte01@gmail.com, "
__studentid__ = "1024610, 1005803, "

import unidecode
import string
import re

from collections import namedtuple
from typing import List, Tuple, Dict
import operator
import functools
from functools import reduce
import itertools

import nltk
import nltk.tokenize as tokenize

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from os import listdir
from os.path import isfile, join
from pathlib import Path

import textGenerator as tg

# To use GPU if available.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Numerical_Parameters = namedtuple('Numerical_Parameters',
                                  'num_epoch sequence_size batch_size lr')
RNN_Parameters = namedtuple('RNN_Parameters',
                              'input_size hidden_size output_size')

def main(*args,**kwargs):
  torch.cuda.manual_seed(10)
  save = False

  data = fetchTextData()
  tokensDict, target_vocab, t_vocab = preProcessData(data)
  #models, losses = trainGenerator(data,target_vocab,t_vocab)
  classifier, loss_train, loss_test = \
    trainClassifier(list(tokensDict.values()),target_vocab,t_vocab)

  #d,l = tg.create_texgen_data(list(models.values()),
  #                            device, target_vocab, t_vocab,100,1000)
  #tg.evaluate_texgen(classifier, device, (d,l),100, 16)

  #Save
  if(save):
    saveModels(models,classifier)

  return models, target_vocab,t_vocab, losses

def preProcessData(data):
  tokensDict = {k : tokenize.word_tokenize(d) for (k,d) in data.items()}
  target_vocab = list(set(itertools.chain(*tokensDict.values())))
  t_vocab = {k:v for v,k in enumerate(target_vocab)}
  #human_names = {k : get_human_names(t) for (k,t) in tokensDict.items()}
  return tokensDict, target_vocab, t_vocab

def get_human_names(tokens):
    pos = nltk.pos_tag(tokens)
    sentt = nltk.ne_chunk(pos, binary = False)
    person_list = []
    person = []
    name = ""
    for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON'):
        for leaf in subtree.leaves():
            person.append(leaf[0])
        if len(person) > 1: #avoid grabbing lone surnames
            for part in person:
                name += part + ' '
            if name[:-1] not in person_list:
                person_list.append(name[:-1])
            name = ''
        person = []

    return person_list

def fetchTextData() -> Dict[str,str]:
  names = None
  path = "./data/"
  availableTexts = [f for f in listdir(path) if isfile(join(path, f))]
  while not names:
    print("\n\n" + str(availableTexts) + \
          "\n\nEnter filenames seperated by whitespaces : ")
    user_input = [str(x) for x in input().split()]
    notFound = [f for f in user_input if f not in availableTexts]
    if notFound:
      print("\n Error not found : " + str(notFound) + "\n")
    names = set(user_input) & set(availableTexts)
    if not names:
      print("Error no text selected ===> Try again Please \n\n")
  print("OK")

  data = {}
  for name in list(names):
    with open(path+name,'r',encoding='utf-8-sig') as file:
      data[name] = file.read().lower()

  return data

def trainClassifier(data,target_vocab,t_vocab):
  rnnParams = RNN_Parameters(len(target_vocab), 256, 4)
  dataTensor, labelsTensor = tg.create_class_data(data,t_vocab,50,100000)
  classifier = tg.sequence_classifier(device, *rnnParams).to(device)
  mp = [classifier,device, (dataTensor,labelsTensor), t_vocab, target_vocab]
  numParam = Numerical_Parameters(5,50,64,0.0001)
  loss_train, loss_test = tg.train(*mp, *numParam, mode="classification")
  return classifier, loss_train, loss_test

def trainGenerator(data,target_vocab,t_vocab):
  rnnParams = RNN_Parameters(len(target_vocab), 512, len(target_vocab))
  numParam = Numerical_Parameters(1,50,64,0.1)
  models = {}
  losses = list()
  for d in data :
    m = tg.RNN(device, *rnnParams).to(device)
    models[d[0]] = m
    fileCheck = Path('models/' + d[0])
    cached = fileCheck.exists()
    if cached:
      m.load_state_dict(torch.load('models/' + d[0]))
    else:
      modelParam = [m,device, d[1] , t_vocab,target_vocab]
      l_train, l_test = m.train(*modelParam, *numParam, mode="textgen")
      createFolder("models/")
      torch.save(model.state_dict(),'models/' + d[0] + '.model')
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
    for m in models:
      torch.save(m[1].state_dict(),'models/' + m[0])
    torch.save(classifier.state_dict(),'models/' + 'classifier')

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

def createFolder(directory):
  try:
      if not os.path.exists(directory):
          os.makedirs(directory)
      return directory
  except OSError:
      print ('Error: Creating directory. ' +  directory)


if __name__ == '__main__':
  model,target_vocab,t_vocab, losses = main()
