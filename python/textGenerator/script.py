#!/usr/bin/env python

"""
Project for IFT6269.
"""

__authors__ = "Jimmy Leroux, Nicolas Laliberte, Frederic Boileau"
__version__ = "1.0"
__maintainer__ = "Jimmy Leroux, Nicolas Laliberte, Frederic Boileau"
__email__ = "jim.leroux1@gmail.com, n.laliberte01@gmail.com, "
__studentid__ = "1024610, 1005803, "

from textGenerator import *

def main(*args,**kwargs):

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
  dataset = pd.read_csv('data/shortjokes.csv')
  dataset = ' '.join(dataset.values[:,1].tolist())
  # create the network.
  rnn = RNN(input_size=n_characters, hidden_size=512, output_size=n_characters,
      n_layers=1).to(device)

  losses = train(dataset, rnn, num_epoch=200, mini_batch_size=200, lr=0.01)

  plt.figure()
  plt.plot(losses, 'sk-')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.show()


if __name__ == '__main__':
  main()
