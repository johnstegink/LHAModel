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
from network.SectionDatasetStat import SectionDatasetStat
from network.SectionDatasetTest import SectionDatasetTest
from texts.corpus import Corpus


def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Create a histogram containing the count of the sections per corpus')
    parser.add_argument('-N', '--sections', help='The number of sections to consider', required=True, type=int)
    parser.add_argument('-c', '--corpus_dir', help='The directory of the corpus to train on', required=True)
    parser.add_argument('-nn', '--neuralnetworktype', help='The type of neural network', required=True, choices=["plain", "masked", "lstm", "stat", "test"])
    parser.add_argument('-s', '--cache_dir', help='The directory used for caching', required=True)
    parser.add_argument('-v', '--vectors_dir', help='The directory containing the document and section vectors of this corpus', required=False)
    parser.add_argument('-r', '--relationsfile', help='The xml file containing the relations between the documents', required=False)
    parser.add_argument('-m', '--heatmap_dir', help='The directory containing the images with the heatmaps', required=True)
    parser.add_argument('-t', '--transformation', help='The transformation to apply to sections with an index larger than N', required=F, choices=['truncate', 'avg'])

    args = vars(parser.parse_args())

    os.makedirs( args["cache_dir"], exist_ok=True)
    if os.path.exists( args["heatmap_dir"]):
        functions.remove_redirectory_recursivly(args["heatmap_dir"])

    return ( args["sections"], args["corpus_dir"], args["cache_dir"], args["vectors_dir"], args["transformation"], args["heatmap_dir"], args["neuralnetworktype"].lower(), args["relationsfile"])

class NeuralNetworkPlain(nn.Module):
    """
    Basic neural network, without masks
    """
    def __init__(self, N):
        super(NeuralNetworkPlain, self).__init__()

        self.hidden1 = nn.Linear(N*(N+1), N*(N+1))
        self.dropout = nn.Dropout( 0.2)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(N*(N+1), 5)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(5, 1)
        self.act_output = nn.Sigmoid()
    def forward(self, x):
        x1 = self.act1(self.hidden1(x))
        # do = self.dropout( x1)
        # x2 = self.act2(self.hidden2(do))
        x2 = self.act2(self.hidden2(x1))
        x_out = self.act_output(self.output(x2))

        return x_out.flatten()

class NeuralNetworkStat(nn.Module):
    """
    Basic neural network, without masks
    """
    def __init__(self, N):
        super(NeuralNetworkStat, self).__init__()

        self.hidden1 = nn.Linear(N, N)
        self.dropout1 = nn.Dropout( 0.2)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(N, N)
        self.dropout2 = nn.Dropout( 0.2)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(N, 1)
        self.act_output = nn.Sigmoid()
    def forward(self, x):
        x1 = self.act1(self.hidden1(x))
        do1 = self.dropout1( x1)
        x2 = self.act2(self.hidden2(do1))
        do2 = self.dropout2( x2)
        x_out = self.act_output(self.output(do2))

        return x_out.flatten()


class NeuralNetworkMask(nn.Module):
    """
    Basic neural network, with mask added
    """
    def __init__(self, N):
        super(NeuralNetworkMask, self).__init__()
        vector_len = 2*N*N

        self.linear = nn.Sequential(
            nn.Linear(vector_len, int( vector_len / 2)),
            nn.Dropout( 0.2),
            nn.ReLU(),
            nn.Linear(int( vector_len / 2), int( vector_len / 10)),
            nn.ReLU(),
            nn.Linear(int( vector_len / 10), 1)
        )

    def forward(self, x):
        out = self.linear( x)
        return torch.sigmoid( out)

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
    if nntype == "masked":
        n = int(n /2)

    matrix = X.reshape( n, n + 1)[:,:(n-1)]
    inverted = 1 - matrix
    plt.pcolormesh( inverted, cmap="hot")
    plt.title( title)

    dir = os.path.join( heatmap_dir, f"{predicted} - {actual}")
    os.makedirs( dir, exist_ok=True)

    plt.savefig( os.path.join(dir ,filename))


def evaluate_the_model( model, validation_ds, nr_of_heatmaps ):
    """
    Make an evaluation of the model
    :param model:
    :param validation_ds:
    :param nr_of_heatmaps:
    :return:
    """
    # Evaluate the model
    correct = 0.0
    fp = 0.0
    tp = 0.0
    fn = 0.0
    total = len(validation_ds)
    with tqdm(total=nr_of_heatmaps, desc="Creating heatmaps") as progress:
        for i in range(0, total):

            (X, Y) = validation_ds[i]
            prediction = 1 if model(X) >= 0.5 else 0

            if prediction == 0 and int(Y) == 1:
                fp += 1.0

            elif prediction == 1 and int(Y) == 1:
                correct += 1.0
                tp += 1.0

            elif prediction == 0 and int(Y) == 1:
                fn += 1.0

            if int(prediction) == int(Y):
                correct += 1.0

            if nr_of_heatmaps > 0:
                equal = int(Y)
                title = validation_ds.get_title(i) + " "
                title += "(Equal)" if equal else "(NOT equal)"

                filename = validation_ds.get_pair(i) + ".png"
                # make_heatmap(X, title, heatmap_dir, filename, prediction, equal)

                nr_of_heatmaps -= 1
                progress.update()

    # Print the accuracy
    if tp > 0:
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        F1 = (2 * recall * precision) / (recall + precision)
        accuracy = int((correct / total) * 100)
        print(f"Accuracy: {accuracy}%\ntp: {tp}, \nfp:{fp}\nF1:{F1}\nPrecision: {precision}\nRecall: {recall}")
        print(f"{F1}\t{accuracy}\t{precision}\t{recall}\t{tp}\t{fp}\t{fn}")




## Main part
if __name__ == '__main__':
    (N, corpusdir, cache_dir, vectors_dir, transformation, heatmap_dir, nntype, relations_file) = read_arguments()

    device = torch.device("cpu")


    cache_file = os.path.join( cache_dir, f"dataset_{N}_{nntype}.pkl")
    if( nntype=="test"):
        train_ds = SectionDatasetTest(device=device, dataset='train', set_size=5000, cache_file=cache_file)
        validation_ds = SectionDatasetTest(device=device, dataset='validation', set_size=5000, cache_file=cache_file)
    elif (nntype == "stat"):
        threshold = 0.3
        train_ds = SectionDatasetStat(device=device, corpus_dir=corpusdir, dataset='train',
                                      relationsfile=relations_file, top_threshold=threshold, cache_file=cache_file)
        validation_ds = SectionDatasetStat(device=device, corpus_dir=corpusdir, dataset='validation',
                                           relationsfile=relations_file, top_threshold=threshold,
                                           cache_file=cache_file)
    else:
        train_ds = SectionDataset(N=N, device=device, corpus_dir=corpusdir, dataset='train',
                                  documentvectors_dir=vectors_dir, transformation=transformation, nntype=nntype,
                                  cache_file=cache_file)
        validation_ds = SectionDataset(N=N, device=device, corpus_dir=corpusdir, dataset='validation',
                                   documentvectors_dir=vectors_dir, transformation=transformation, nntype=nntype,
                                   cache_file=cache_file)


    # parameters
    batch_size = 100
    n_epochs = 50
    learning_rate = 0.01
    nr_of_heatmaps = 0


    train_dl = DataLoader( train_ds, batch_size=batch_size, shuffle=False)
    validation_dl = DataLoader( validation_ds, batch_size=batch_size, shuffle=False)


    # Define the model
    if nntype == "plain":
        model = NeuralNetworkPlain( N)
    elif nntype == "masked":
        model = NeuralNetworkMask(N)
    elif nntype == "stat" or nntype=="test":
        model = NeuralNetworkStat( 4)
    else:
        raise f"Unknown neural network type: {nntype}"


    model.to( device)
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = []

        for batch in train_dl:
            (X, Y, titles, pairs) = batch

            y_pred = model(X)
            loss = loss_fn(y_pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append( loss.item())

        # evaluate_the_model(model=model, validation_ds=validation_ds, nr_of_heatmaps=nr_of_heatmaps)
        losses.append( epoch_loss)
        print(f'Finished epoch {epoch}, latest loss {epoch_loss}')

    # Plot the loss graph.
    plt.plot([epoch for epoch in range(n_epochs)], [loss for loss in losses])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    evaluate_the_model(model=model, validation_ds=train_ds, nr_of_heatmaps=nr_of_heatmaps)







