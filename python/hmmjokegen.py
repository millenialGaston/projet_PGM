#!/usr/bin/env python

"""
Project for IFT6269.
"""

__authors__ = "Jimmy Leroux, Nicolas Laliberte, Frederic Boileau"
__version__ = "1.0"
__maintainer__ = "Jimmy Leroux, Nicolas Laliberte, Frederic Boileau"
__email__ = "jim.leroux1@gmail.com, n.laliberte01@gmail.com, "
__studentid__ = "1024610, 1005803, "

import numpy
import pandas as pd
import unidecode
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import re
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, 256)
        self.linear = nn.Linear(256, output_size)
    
    def forward(self, inputs, hidden):
        inputs = self.encoder(inputs.view(len(inputs), -1))
        output, hidden = self.gru(inputs.view(len(inputs), 1, -1), hidden)
        output = self.decoder(output.view(len(inputs), -1))
        output = self.linear(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size)

def random_minibatch(data, sequence_size):
    mini = torch.zeros(sequence_size).long()
    start_index = random.randint(0, len(data) - sequence_size)
    end_index = start_index + sequence_size + 1
    tomini = data[start_index: end_index]
    
    for c in range(sequence_size):
        mini[c] = all_characters.index(tomini[c])    
    
    dat, target = mini[:-1], mini[1:]
    return dat, target

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])   
    return tensor

def evaluate(model, init_str='W', predict_len=100, temperature=0.8):
    model.eval()
    with torch.no_grad():
        hidden = model.init_hidden()
        inputs = char_tensor(init_str)
        #print(prime_input)
        predicted = init_str

        _, hidden = model(inputs.to(device), hidden.to(device))
        inp = inputs[-1].reshape(1)
        
        for p in range(predict_len):
            output, hidden = model(inp.to(device), hidden.to(device))
            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]
            
            # Add predicted character to string and use as next input
            predicted_char = all_characters[top_i]
            predicted += predicted_char
            inp = char_tensor(predicted_char)
    return predicted

def train(data, model, num_epoch, mini_batch_size=200, lr=0.005):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    loss_avg = 0
    for epoch in range(1, num_epoch):
        model.train()
        loss_avg = 0
        model.zero_grad()
        hidden = model.init_hidden()
        inputs, targets = random_minibatch(data, mini_batch_size)
        output, hidden = model(inputs.to(device), hidden.to(device))
        loss = criterion(output.to(device), targets.to(device))
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(evaluate(model,'W', 100), '\n')
            print(epoch)
            print(loss.item())
        if epoch % 5 == 0:
            losses.append(loss.item())    

    return losses

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed(10)

    all_characters = string.printable
    n_characters = len(all_characters)
    
    dataset = pd.read_csv('shortjokes.csv')
    dataset = ' '.join(dataset.values[:,1].tolist())

    rnn = RNN(input_size=n_characters, hidden_size=512, output_size=n_characters,
        n_layers=1).to(device)
    
    losses = train(dataset, rnn, 50, 200)
    
    plt.figure()
    plt.plot(losses)
    plt.show()
