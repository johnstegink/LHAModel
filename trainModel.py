# Train the model for phase 3

import argparse
import math
import os

import numpy as np
import torch
from numpy.random import multivariate_normal
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

import functions
from network.SectionDataset import SectionDataset
from texts.corpus import Corpus


def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Create a histogram containing the count of the sections per corpus')
    parser.add_argument('-N', '--sections', help='The number of sections to consider', required=True, type=int)
    parser.add_argument('-c', '--corpus_dir', help='The directory of the corpus to train on', required=True)
    parser.add_argument('-s', '--cache_dir', help='The directory used for caching', required=True)
    parser.add_argument('-v', '--vectors_dir', help='The directory containing the document and section vectors of this corpus', required=True)
    parser.add_argument('-m', '--heatmap_dir', help='The directory containing the images with the heatmaps', required=True)
    parser.add_argument('-t', '--transformation', help='The transformation to apply to sections with an index larger than N', required=True, choices=['truncate', 'avg'])

    args = vars(parser.parse_args())

    os.makedirs( args["cache_dir"], exist_ok=True)
    if os.path.exists( args["heatmap_dir"]):
        functions.remove_redirectory_recursivly(args["heatmap_dir"])

    return ( args["sections"], args["corpus_dir"], args["cache_dir"], args["vectors_dir"], args["transformation"], args["heatmap_dir"])


class NeuralNetwork(nn.Module):
    def __init__(self, N):
        super(NeuralNetwork, self).__init__()

        self.hidden1 = nn.Linear(N*(N+1), N*(N+1))
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(N*(N+1), 5)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(5, 1)
        self.act_output = nn.Sigmoid()
    def forward(self, x):
        x1 = self.act1(self.hidden1(x))
        x2 = self.act2(self.hidden2(x1))
        x_out = self.act_output(self.output(x2))

        return x_out.flatten()

def make_heatmap( X, title, heatmap_dir, filename, predicted, actual):
    """
    Create a single heatmap
    :param X:
    :param title:
    :param heatmap_dir:
    :param filename:
    :param predicted:
    :param actual:
    :return:
    """
    n = int( math.sqrt (len(X)))
    matrix = X.reshape( n, n + 1)[:,:(n-1)]
    inverted = 1 - matrix
    plt.pcolormesh( inverted, cmap="hot")
    plt.title( title)

    dir = os.path.join( heatmap_dir, f"{predicted} - {actual}")
    os.makedirs( dir, exist_ok=True)

    plt.savefig( os.path.join(dir ,filename))





## Main part
if __name__ == '__main__':
    (N, corpusdir, cache_dir, vectors_dir, transformation, heatmap_dir) = read_arguments()

    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    device = torch.device("cpu")

    cache_file = os.path.join( cache_dir, f"dataset_{N}.pkl")
    train_ds = SectionDataset( N=N, device=device, corpus_dir=corpusdir, dataset='train', documentvectors_dir=vectors_dir, transformation=transformation, cache_file=cache_file)
    validation_ds = SectionDataset( N=N, device=device, corpus_dir=corpusdir, dataset='validation', documentvectors_dir=vectors_dir, transformation=transformation, cache_file=cache_file)

    # parameters
    batch_size = 32
    n_epochs = 100
    learning_rate = 0.001
    nr_of_heatmaps = 100


    train_dl = DataLoader( train_ds, batch_size=batch_size, shuffle=True)
    validation_dl = DataLoader( validation_ds, batch_size=batch_size, shuffle=False)


    # Define the model
    model = NeuralNetwork( N)
    model.to( device)
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # Train the model
    for epoch in range(n_epochs):
        for batch in train_dl:
            (X, Y) = batch

            y_pred = model(X)
            loss = loss_fn(y_pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')


    # Evaluate the model
    correct = 0
    total = len(validation_ds)
    with tqdm(total=nr_of_heatmaps, desc="Creating heatmaps") as progress:
        for i in range(0, total):

            (X, Y) = validation_ds[i]
            prediction = 1 if model(X) >= 0.5 else 0

            if int(prediction) == int(Y):
                correct += 1

            if nr_of_heatmaps > 0:
                equal = int(Y)
                title = validation_ds.get_title(i) + " "
                title += "(Equal)" if equal else "(NOT equal)"

                filename = validation_ds.get_pair(i) + ".png"
                make_heatmap(X, title, heatmap_dir, filename, prediction, equal)

                nr_of_heatmaps -= 1
                progress.update()




    # Print the accuracy
    print( correct)
    print( total)
    accuracy = int((correct / total) * 100)
    print(f"Accuracy: {accuracy}%")
