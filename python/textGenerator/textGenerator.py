#!/usr/bin/env python

"""
Project for IFT6269.
"""

__authors__ = "Jimmy Leroux, Nicolas Laliberte, Frederic Boileau"
__version__ = "1.0"
__maintainer__ = "Jimmy Leroux, Nicolas Laliberte, Frederic Boileau"
__email__ = "jim.leroux1@gmail.com, n.laliberte01@gmail.com, "
__studentid__ = "1024610, 1005803, "

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

class RNN(nn.Module):
    '''
    Define the model structure.
    '''

    def __init__(self, device, input_size, hidden_size, output_size,
            n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size    # Size of the character list.
        self.hidden_size = hidden_size  # Size of the hidden layer.
        self.output_size = output_size  # Size of output, here same as input.
        self.n_layers = n_layers
        self.embedding_dim = 128
        self.encoder= nn.Embedding(input_size, self.embedding_dim) # Encode inputs.
        self.lstm = nn.LSTM(self.embedding_dim, hidden_size, n_layers,
            batch_first=True)
        self.linear1 = nn.Linear(self.hidden_size, output_size)
        self.device = device

    def forward(self, inputs, hidden, sequence_len, batch_size):
        inputs = inputs.view(batch_size, sequence_len)
        inputs = self.encoder(inputs)
        inputs, hidden = self.lstm(inputs, hidden)
        inputs = inputs.contiguous()
        output = self.linear1(inputs.view(batch_size*sequence_len,-1))

        return output, hidden

    def init_hidden(self, batch_size):
        '''
        Initialize the hidden layer to 0s.
        '''

        return (torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device),
            torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device))

    def create_emb_layer(self, weights_matrix, non_trainable=False):
        num_embeddings, embedding_dim = weights_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer, num_embeddings, embedding_dim

class quote_dataset(torch.utils.data.dataset.Dataset):
    '''
    Create a Dataset to use with DataLoaders. We need to define __getitem__
    and __len__.
    '''

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        dat = self.data[index]
        return (dat,self.labels[index])

    def __len__(self):
        return len(self.data)

def create_data(datas, vocab, sequence_size):
    '''
    Format the datas in sequences.

    Parameters:
    -----------
    datas: The data we want to split in sequences.
    vocab: Dictionary of the words in datas.
    sequence_size: The length of the output sequences.

    Returns:
    --------
    data: tensor containing the data, shape(number of sequence, sequence_size-1)
    labels: tensor containing the labels, same shape as data
    '''

    # Calculate the number of sequences.
    num_data = (len(datas) - sequence_size) // sequence_size
    # Initialize the tensors.
    sequence = torch.zeros(sequence_size).long()
    data = torch.zeros(num_data, sequence_size-1).long()
    labels = torch.zeros(num_data, sequence_size-1).long()
    for i in range(num_data):
        for s in range(sequence_size):
            sequence[s] = vocab[datas[i * sequence_size + s]]
        data[i,:] , labels[i,:] = sequence[:-1], sequence[1:]

    return data, labels

def char_tensor(string, target_vocab):
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
    string = string.split()
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = target_vocab.index(string[c])
        #tensor[c] = all_characters.index(string[c])   

    return tensor

def evaluate(model,device, target_vocab, init_str='W', predict_len=100,
        temperature=0.7):
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
        hidden = model.init_hidden(1)
        init = char_tensor(init_str, target_vocab)
        predicted = init_str+' '

        # Build up the hidden state with inputs.
        _, hidden = model(init.to(device), hidden, len(init),1)
        # Take the last element of init as input.
        inp = init[-1].reshape(1)

        for p in range(predict_len):
            output, hidden = model(inp.to(device), hidden, 1, 1)
            # Sample from the network as a multinomial distribution.
            output_dist = output.data.view(-1).div(temperature).exp()
            retained = torch.multinomial(output_dist, 1)[0]

            # Add predicted character to string and use as next input.
            predicted_char = target_vocab[retained]+' '
            predicted += predicted_char
            inp = char_tensor(predicted_char, target_vocab)

    return predicted

def cross_loss(model, device, dataset, t_vocab, sequence_size, batch_size):
    data, labels = create_data(dataset[:100000], t_vocab, sequence_size)
    testloader = torch.utils.data.DataLoader(quote_dataset(data,labels),
        batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss_avg = 0
    with torch.no_grad():
        # Calculate the training loss
        for i, data in enumerate(testloader):
            inputs, targets = data
            hidden = model.init_hidden(inputs.shape[0])
            output, hidden = model(inputs.to(device), hidden,
                sequence_size-1, inputs.shape[0])
            targets = targets.contiguous()
            targets = targets.view(inputs.shape[0] * (sequence_size-1))
            loss = criterion(output.to(device), targets.to(device))
            loss_avg += loss.item()
        
    return loss_avg/len(testloader)


def train(model, device, dataset, t_vocab, target_vocab, num_epoch,
        cross_dataset=None, sequence_size=20, batch_size=200, lr=0.005):
    '''
    Function used to train the model on the joke dataset.

    Parameters:
    -----------
    model: The model we are training.
    num_epoch: The number of epoch we want to do. In fact it's not really an
                epoch because we are not going through the whole dataset.
    sequence_size: length of the sequences.
    batch_size: The size of the sequence we are passing to the network.
    lr: The learning rate.

    Returns:
    --------
    loss_train: list of the losses for the traiset.
    loss_test: list of the losses for the testset.
    '''

    # Create DataLoaders to facilitate the data manipulation via minibatches.
    data, labels = create_data(dataset[:350000], t_vocab, sequence_size)
    trainloader = torch.utils.data.DataLoader(quote_dataset(data,labels),
        batch_size=batch_size, shuffle=True, num_workers=0)

    data, labels = create_data(dataset[350000:400000], t_vocab, sequence_size)
    testloader = torch.utils.data.DataLoader(quote_dataset(data,labels),
        batch_size=batch_size, shuffle=False, num_workers=0)

    # We use Cross entropy loss. This combine negative loss likelihood with a
    # softmax function for the prediction.
    criterion = nn.CrossEntropyLoss()

    loss_train = []
    loss_test = []
    loss_cross = []
    for epoch in range(num_epoch):
        loss_avg_train = 0
        loss_avg_test = 0
        for i, data in enumerate(trainloader):
            model.train()
            # Learning rate decay.
            lrd = lr * (1./(1 + 9 * epoch / num_epoch))
            # Define the optimizing method and pass the parameters to optimize.
            optimizer = torch.optim.Adam(model.parameters(), lr=lrd)
            # zero the gradient after each step.
            model.zero_grad()
            # get a training exemple.
            inputs, targets = data
            # init the hidden state.
            hidden = model.init_hidden(inputs.shape[0])
            # pass it through the network.
            output, hidden = model(inputs.to(device), hidden, sequence_size-1,
                inputs.shape[0])
            # calculate the loss.
            targets = targets.contiguous()
            targets = targets.view(inputs.shape[0] * (sequence_size-1))
            loss = criterion(output.to(device), targets.to(device))
            # populate the gradients.
            loss.backward()
            # make a step.
            optimizer.step()

        # Pout the model on eval to calculate the losses   
        model.eval()
        with torch.no_grad():
            # Calculate the training loss
            for i, data in enumerate(trainloader):
                inputs, targets = data
                hidden = model.init_hidden(inputs.shape[0])
                output, hidden = model(inputs.to(device), hidden,
                    sequence_size-1, inputs.shape[0])
                targets = targets.contiguous()
                targets = targets.view(inputs.shape[0] * (sequence_size-1))
                loss = criterion(output.to(device), targets.to(device))
                loss_avg_train += loss.item()
            loss_train.append(loss_avg_train/len(trainloader))
            # Calculate the test loss
            for i, data in enumerate(testloader):
                inputs, targets = data
                hidden = model.init_hidden(inputs.shape[0])
                output, hidden = model(inputs.to(device), hidden,
                    sequence_size-1, inputs.shape[0])
                targets = targets.contiguous()
                targets = targets.view(inputs.shape[0] * (sequence_size-1))
                loss = criterion(output.to(device), targets.to(device))
                loss_avg_test += loss.item()
            loss_test.append(loss_avg_test/len(testloader))
        if cross_dataset is not None:
            loss_cross.append(cross_loss(model, device, cross_dataset, t_vocab,
                sequence_size, batch_size))
        # Print an exemple of generated sequence.
        print('Epoch: {}'.format(epoch))
        print(evaluate(model,device,target_vocab,'i', 40))
        print('Train error: {0:.2f} Test error: {1:.2f}\n'.format(
                    loss_train[epoch], loss_test[epoch]))
        if cross_dataset is not None:
            print('Train error: {0:.2f} Test error: {1:.2f} Cross error: {2:.2f}\
                \n'.format(loss_train[epoch], loss_test[epoch],
                loss_cross[epoch]))
    return loss_train, loss_test, loss_cross

