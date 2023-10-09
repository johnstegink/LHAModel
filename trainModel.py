# Train the model for phase 3

import argparse
import math
import os

import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
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
    parser.add_argument('-o', '--results_dir', help='The director containing the results', required=True)

    args = vars(parser.parse_args())

    os.makedirs( args["cache_dir"], exist_ok=True)
    os.makedirs( args["results_dir"], exist_ok=True)
    if os.path.exists( args["heatmap_dir"]):
        functions.remove_redirectory_recursivly(args["heatmap_dir"])

    return ( args["sections"], args["corpus_dir"], args["cache_dir"], args["vectors_dir"], args["transformation"], args["heatmap_dir"], args["neuralnetworktype"].lower(), args["relationsfile"], args["results_dir"])

class NeuralNetworkPlain(nn.Module):
    """
    Basic neural network, without masks
    """
    def __init__(self, N):
        super(NeuralNetworkPlain, self).__init__()

        self.hidden1 = nn.Linear(N*N, 5)
        self.dropout1 = nn.Dropout(0.0)
        self.act1 = nn.ReLU()
        # self.hidden2 = nn.Linear(N, 5)
        # self.dropout2 = nn.Dropout(0.1)
        # self.act2 = nn.ReLU()
        self.output = nn.Linear(5, 1)
        self.act_output = nn.Sigmoid()

        torch.nn.init.xavier_uniform( self.hidden1.weight)
        torch.nn.init.xavier_uniform( self.output.weight)

    def forward(self, x):
        x1 = self.act1(self.hidden1(x))
        do1 = self.dropout1(x1)
        # x2 = self.act2(self.hidden2(do1))
        # do2 = self.dropout2(x2)
        x_out = self.act_output(self.output(do1))

        return x_out


class NeuralNetworkStat(nn.Module):
    """
    Basic neural network, without masks
    """
    def __init__(self, N):
        super(NeuralNetworkStat, self).__init__()

        self.hidden1 = nn.Linear(N, N)
        self.dropout1 = nn.Dropout( 0.0)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(N, N)
        self.dropout2 = nn.Dropout( 0.0)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(N, 1)
        self.act_output = nn.Sigmoid()
    def forward(self, x):
        x1 = self.act1(self.hidden1(x))
        do1 = self.dropout1( x1)
        x2 = self.act2(self.hidden2(do1))
        do2 = self.dropout2( x2)
        x_out = self.act_output(self.output(do2))

        return x_out

class NeuralNetworkTest(nn.Module):
    """
    Basic neural network, without masks
    """
    def __init__(self, N):
        super(NeuralNetworkTest, self).__init__()

        self.hidden1 = nn.Linear(N, N)
        self.dropout1 = nn.Dropout( 0.0)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(N, N)
        self.dropout2 = nn.Dropout( 0.0)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(N, 1)
        self.act_output = nn.Sigmoid()
    def forward(self, x):
        x1 = self.act1(self.hidden1(x))
        do1 = self.dropout1( x1)
        # x2 = self.act2(self.hidden2(do1))
        # do2 = self.dropout2( x2)
        x_out = self.act_output(self.output(do1))

        return x_out


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

def create_heatmap(X, title, heatmap_dir, filename, predicted, actual):
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

    matrix = X.reshape(n, n)[:,:(n)]
    inverted = 1 - matrix
    plt.pcolormesh( inverted, cmap="hot")
    plt.title( title)

    dir = os.path.join( heatmap_dir, f"{predicted} - {actual}")
    os.makedirs( dir, exist_ok=True)

    plt.savefig( os.path.join(dir ,filename))


def evaluate_the_model( model, Y, X_test, latest_loss, results_file, titles, pairs, nr_of_heatmaps, batch_size, n_epochs, learning_rate, first):
    """
    Make an evaluation of the model
    :param model:
    :param y:
    :param x_test:
    :return:
    """

    # change the y_pred to 0 or 1
    y_pred = [True if model( x) >= 0.5 else False for x in X_test]
    y_act = [True if y >= 0.5 else False for y in Y]

    (precision, recall, F1, _) =sklearn.metrics.precision_recall_fscore_support( y_act, y_pred, average="binary")
    accuracy = sklearn.metrics.accuracy_score( y_act, y_pred)

    to_be_created = min(nr_of_heatmaps, len(y_pred))
    with tqdm(total=to_be_created, desc="Creating heatmaps") as progress:
        for i in range(0, to_be_created):
            title = titles[i] + " "
            title += "(Equal)" if y_act[i] else "(NOT equal)"

            filename = pairs[i] + ".png"
            create_heatmap(X_test[i], title, heatmap_dir, filename, y_pred[i], y_act[i])

            nr_of_heatmaps -= 1
            progress.update()


    tp = fp = fn = 0
    for i in range(0, len(y_pred)):
        if y_pred[i] and y_act[i] : tp += 1
        elif y_pred[i] and not y_act[i]: fp += 1
        elif not y_pred[i] and y_act[i]: fn += 1

    readable = f"Batch size: {batch_size}\nEpochs: {n_epochs}, Learning rate: {learning_rate}\nAccuracy: {accuracy * 100:.2f}%\ntp: {tp}, \nfp:{fp}\nF1:{F1 * 100:.2f}\nPrecision: {precision}\nRecall: {recall}\nLatest loss: {latest_loss}"
    tsv = f"{batch_size}\t{n_epochs}\t{learning_rate}\t{F1}\t{accuracy}\t{precision}\t{recall}\t{tp}\t{fp}\t{fn}\t{latest_loss}\n";
    header = "Batch size\tEpochs\tLearning rate\tF1\tAccuracy\tPrecision\tRecall\tTrue Positives\tFalse Positives\tLatest loss\n";
    print( readable)

    if first:
        functions.write_file(results_file + ".txt", readable)
        functions.write_file(results_file + ".tsv", header + tsv)
    else:
        functions.append_file(results_file + ".txt", readable)
        functions.append_file(results_file + ".tsv", tsv)



## Main part
if __name__ == '__main__':
    (N, corpusdir, cache_dir, vectors_dir, transformation, heatmap_dir, nntype, relations_file, output_dir) = read_arguments()

    device = torch.device("cpu")

    cache_file = os.path.join( cache_dir, f"dataset_{N}_{nntype}.pkl")
    base_file = os.path.basename(relations_file).split('.')[0].replace("_pairsonly", "")
    results_file = os.path.join( output_dir, f"results_{base_file}_{N}_{nntype}")

    if( nntype=="test"):
        ds = SectionDatasetTest(device=device, set_size=5000, cache_file=cache_file)
    elif (nntype == "stat"):
        threshold = 0.3
        ds = SectionDatasetStat(device=device, corpus_dir=corpusdir, relations_file=relations_file, top_threshold=threshold, cache_file=cache_file)
    else:
        ds = SectionDataset(N=N, device=device, corpus_dir=corpusdir,
                                  documentvectors_dir=vectors_dir, transformation=transformation, nntype=nntype,
                                  cache_file=cache_file)


    X = torch.tensor( [data[0] for data in ds], dtype=torch.float32, device=device)
    Y = torch.tensor( [[data[1]] for data in ds], dtype=torch.float32, device=device)
    titles = [data[2] for data in ds]
    pairs = [data[3] for data in ds]

    # split the dataset into training and test sets
    validation_start = len(X) - int(len(X) / 10)

    X_train = X[:validation_start]
    y_train = Y[:validation_start]
    titles_train = titles[:validation_start]
    pairs_train = pairs[:validation_start]
    X_test = X[validation_start:]
    Y_test = Y[validation_start:]
    titles_test = titles[validation_start:]
    pairs_test = pairs[validation_start:]


    # Define the model
    if nntype == "plain":
        model = NeuralNetworkPlain(N)
    elif nntype == "masked":
        model = NeuralNetworkMask(N)
    elif nntype == "stat":
        model = NeuralNetworkTest(4)
    elif nntype == "test":
        model = NeuralNetworkTest(4)
    else:
        raise f"Unknown neural network type: {nntype}"

    model.to(device)

    nr_of_heatmaps = 0
    plot_graph = False
    first = True
    for batch_size in [20, 50, 100, 200]:
        for n_epochs in [10, 60, 100, 200]:
            for learning_rate in [0.001, 0.01, 0.1]:
                loss_fn = nn.BCELoss()  # binary cross entropy
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                batches_per_epoch = math.ceil(len(X_train) / batch_size)

                # Train the model
                losses = []
                for epoch in range(n_epochs):
                    for i in range( batches_per_epoch):
                        start = i * batch_size

                        # take a batch
                        Xbatch = X_train[start:min(start + batch_size, len( X_train))]
                        y_batch = y_train[start:min(start + batch_size, len( X_train))]

                        # forward pass
                        y_pred = model(Xbatch)
                        loss = loss_fn(y_pred, y_batch)

                        # backward pass
                        optimizer.zero_grad()
                        loss.backward()

                        # update weights
                        optimizer.step()

                    epoch_loss = loss.item()
                    losses.append( epoch_loss)
                    # print(f'Finished epoch {epoch}, latest loss {epoch_loss}')

                latest_loss = epoch_loss

                # Plot the loss graph.
                if plot_graph:
                    plt.plot([epoch for epoch in range(n_epochs)], [loss for loss in losses])
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.show()

                evaluate_the_model(
                    model=model,
                    X_test=X_test,
                    Y=Y_test,
                    latest_loss=latest_loss,
                    titles=titles_test,
                    pairs=pairs_test,
                    results_file=results_file,
                    nr_of_heatmaps=nr_of_heatmaps,
                    first=first,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    learning_rate=learning_rate
                )
                first = False






