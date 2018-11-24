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
import torch.nn.functional as F
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt

class RNN(nn.Module):
    '''
    Define the model structure.
    '''

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size    # Size of the character list.
        self.hidden_size = hidden_size  # Size of the hidden layer.
        self.output_size = output_size  # Size of output, here same as input.
        self.n_layers = n_layers
        self.encoder = nn.Embedding(input_size, hidden_size) # Encode inputs.
        self.gru = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, 256)
        self.linear = nn.Linear(256, output_size)

    def forward(self, inputs, hidden):
        inputs = self.encoder(inputs.view(len(inputs), -1))
        output, hidden = self.gru(inputs.view(len(inputs),1, -1), hidden)
        output = F.relu(self.decoder(output.view(len(inputs), -1)))
        output = self.linear(output)
        return output, hidden

    def init_hidden(self):
        '''
        Initialize the hidden layer to 0s.
        '''

        return (torch.zeros(self.n_layers, 1, self.hidden_size),
            torch.zeros(self.n_layers, 1, self.hidden_size))

def random_minibatch(data, sequence_size):
    '''
    Create a random 'minibatch' from the data. Sample randomly a seqeuence of
    length sequence_size from the total dataset and create a training exemple.

    Parameters:
    -----------
    data: String containing all the jokes back to back.
    sequence_size: size of the extracted substring.

    Returns:
    --------
    dat: The substring used as input in the model.
    target: The target value for each character in dat. If we have 'abcd', we
            define dat = 'abc' and target = 'bcd'.
    '''

    mini = torch.zeros(sequence_size).long()
    while True:
        try:
            start_index = random.randint(0, len(data) - sequence_size)
            end_index = start_index + sequence_size + 1
            tomini = data[start_index: end_index]
        
            for c in range(sequence_size):
                mini[c] = all_characters.index(tomini[c])    
            dat, target = mini[:-1], mini[1:]
        except:
            print('resample')
        else:
            break
    return dat, target

def char_tensor(string):
    '''
    Function used to convert a string to a tensor of 'index'.

    Parameters:
    -----------
    string: The string to convert.

    Returns:
    --------
    tensor: Tensor containing the indices (in our dictionary) of each character
            of string.
    '''

    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])   
    return tensor

def evaluate(model, init_str='W', predict_len=100, temperature=0.8):
    '''
    Function generating a sequence of length predict_len from our model. We
    start by propagating an init_str in the model to build an initial hidden
    state.

    Parameters:
    -----------
    model: The model we want to evaluate. Here it's a RNN instance.
    init_str: String used to build up / initialize the hidden state.
    predict_len: Length of the sequence we wish to generate.
    temperature: Parameter used to control the randomness of the chosen best
                    best sequence.

    Returns:
    --------
    predicted: The predicted / generated sequence.
    '''

    model.eval()
    with torch.no_grad():
        hidden = model.init_hidden()
        init = char_tensor(init_str)
        predicted = init_str

        # Build up the hidden state with inputs.
        _, hidden = model(init.to(device), (hidden[0].to(device),hidden[1].to(
            device)))
        
        # Take the last element of init as input.
        inp = init[-1].reshape(1)
        
        for p in range(predict_len):
            output, hidden = model(
                inp.to(device), (hidden[0].to(device),hidden[1].to(device)))
            
            # Sample from the network as a multinomial distribution.
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]
            
            # Add predicted character to string and use as next input.
            predicted_char = all_characters[top_i]
            predicted += predicted_char
            inp = char_tensor(predicted_char)
    return predicted

def train(data, model, num_epoch, mini_batch_size=200, lr=0.005):
    '''
    Function used to train the model on the joke dataset.

    Parameters:
    -----------
    data: The dataset we are training on i.e the jokes.
    model: The model we are training.
    num_epoch: The number of epoch we want to do. In fact it's not really an
                epoch because we are not going through the whole dataset.
    mini_batch_size: The size of the sequence we are passing to the network.
    lr: The learning rate.

    Returns:
    --------
    losses: list of the losses at each 5 step.
    '''

    # Define the optimizer and pass the model's parameters to it.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # We use Cross entropy loss. This combine negative loss likelihood with a
    # softmax function for the prediction.
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    loss_avg = 0
    for epoch in range(1, num_epoch):
        model.train()
        if epoch > 300:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        loss_avg = 0
        # zero the gradient after each step.
        model.zero_grad()
        # init the hidden state.
        hidden = model.init_hidden()
        # get a training exemple.
        inputs, targets = random_minibatch(data, mini_batch_size)
        # pass it through the network.
        output, hidden = model(
            inputs.to(device), (hidden[0].to(device),hidden[1].to(device)))
        # calculate the loss.
        loss = criterion(output.to(device), targets.to(device))
        # populate the gradients.
        loss.backward()
        # make a step.
        optimizer.step()

        if epoch % 5 == 0:
            # Print an exemple of generated sequence.
            print(evaluate(model,'Christ is ', 100), '\n')
            print(epoch)
            print(loss.item())
        if epoch % 5 == 0:
            losses.append(loss.item())    
    return losses

if __name__ == '__main__':
    plt.style.use('ggplot')     
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('axes', labelsize=15)

    # To use GPU if available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed(10)
    # Our dictionary / alphabet.
    all_characters = string.printable
    n_characters = len(all_characters)
    # load the dataset.
    dataset = pd.read_csv('shortjokes.csv')
    dataset = ' '.join(dataset.values[:,1].tolist())
    # create the network.
    rnn = RNN(input_size=n_characters, hidden_size=512, output_size=n_characters,
        n_layers=1).to(device)
    
    losses = train(dataset, rnn, num_epoch=2000, mini_batch_size=4000, lr=0.0001)
    
    plt.figure()
    plt.plot(losses, 'sk-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
