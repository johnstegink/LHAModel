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
from sklearn.model_selection import StratifiedKFold
import statistics


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

    parser = argparse.ArgumentParser(description='Train the optimal model')
    parser.add_argument('-N', '--sections', help='The number of sections to consider', required=True, type=int)
    parser.add_argument('-c', '--corpus_dir', help='The directory of the corpus to train on', required=True)
    parser.add_argument('-nn', '--neuralnetworktype', help='The type of neural network', required=True, choices=["plain", "masked", "lstm", "stat", "test"])
    parser.add_argument('-s', '--cache_dir', help='The directory used for caching', required=True)
    parser.add_argument('-v', '--vectors_dir', help='The directory containing the document and section vectors of this corpus', required=False)
    parser.add_argument('-r', '--relationsfile', help='The xml file containing the relations between the documents', required=False)
    parser.add_argument('-m', '--heatmap_dir', help='The directory containing the images with the heatmaps', required=True)
    parser.add_argument('-t', '--transformation', help='The transformation to apply to sections with an index larger than N', required=F, choices=['truncate', 'avg'])
    parser.add_argument('-o', '--results_dir', help='The director containing the results', required=True)
    parser.add_argument('-mo', '--models_dir', help='The director containing the models, optional', required=False)


    args = vars(parser.parse_args())

    os.makedirs( args["cache_dir"], exist_ok=True)
    os.makedirs( args["results_dir"], exist_ok=True)
    if not args["models_dir"] is None:
        os.makedirs( args["models_dir"], exist_ok=True)

    if os.path.exists( args["heatmap_dir"]):
        functions.remove_redirectory_recursivly(args["heatmap_dir"])

    return ( args["sections"], args["corpus_dir"], args["cache_dir"], args["vectors_dir"], args["transformation"], args["heatmap_dir"], args["neuralnetworktype"].lower(), args["relationsfile"], args["results_dir"], args["models_dir"])

class NeuralNetworkPlain(nn.Module):
    """
    Basic neural network, without masks
    """
    def __init__(self, N):
        super(NeuralNetworkPlain, self).__init__()

        self.hidden1 = nn.Linear(N*N, 5)
        self.dropout1 = nn.Dropout(0.0)
        self.act1 = nn.ReLU()
        self.output = nn.Linear(5, 1)
        self.act_output = nn.Sigmoid()

        torch.nn.init.xavier_uniform( self.hidden1.weight)
        torch.nn.init.xavier_uniform( self.output.weight)

    def forward(self, x):
        x1 = self.act1(self.hidden1(x))
        do1 = self.dropout1(x1)
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


def calculate_metrics( model, Y, X_test):
    """
    Calculate the metrics and return a list with the metrics
    :param model:
    :param y:
    :param x_test:
    :return: [F1, accuracy, precision, recall, tp, fp, fn]
    """

    # change the y_pred to 0 or 1
    y_pred = [True if model( x) >= 0.5 else False for x in X_test]
    y_act = [True if y >= 0.5 else False for y in Y]

    (precision, recall, F1, _) =sklearn.metrics.precision_recall_fscore_support( y_act, y_pred, average="binary")
    accuracy = sklearn.metrics.accuracy_score( y_act, y_pred)

    tp = fp = fn = 0
    for i in range(0, len(y_pred)):
        if y_pred[i] and y_act[i] : tp += 1
        elif y_pred[i] and not y_act[i]: fp += 1
        elif not y_pred[i] and y_act[i]: fn += 1



    return [F1, accuracy, precision, recall, tp, fp, fn]




def evaluate_the_model( model, X, Y, metrics,  results_file, batch_size, n_epochs, learning_rate, first):
    """
    Make an evaluation of the model
    :param metrics:
    :param results_file:
    :param batch_size:
    :param n_epochs:
    :param learning_rate:
    :param first:
    :return:
    """

    y_pred = []
    y_act = []
    for i in range( len(X)):
        x = torch.tensor([X[i]], dtype=torch.float32, device=device);
        y_pred.append( float(model(x)[0]) >= 0.5)
        y_act.append( Y[i] >= 0.5)

    F1 = sklearn.metrics.f1_score( y_act, y_pred)
    accuracy = sklearn.metrics.accuracy_score( y_act, y_pred)
    precision = sklearn.metrics.precision_score(y_act, y_pred)
    recall = sklearn.metrics.recall_score(y_act, y_pred)
    tp = sum( [1 for i in range(len(X)) if y_pred[i] and y_act[i]])
    fp = sum( [1 for i in range(len(X)) if y_pred[i] and not y_act[i]])
    fn = sum( [1 for i in range(len(X)) if not y_pred[i] and y_act[i]])

    readable = f"Batch size: {batch_size}\nEpochs: {n_epochs}, Learning rate: {learning_rate}\nAccuracy: {accuracy * 100:.2f}%\ntp: {tp}, \nfp:{fp}\nF1:{F1 * 100:.2f}\nPrecision: {precision}\nRecall: {recall}\n"
    tsv = f"{batch_size}\t{n_epochs}\t{learning_rate}\t{F1}\t{accuracy}\t{precision}\t{recall}\t{tp}\t{fp}\t{fn}\n"
    header = "Batch size\tEpochs\tLearning rate\tF1\tAccuracy\tPrecision\tRecall\tTrue Positives\tFalse Positives\tFalse Negatives\n"
    print( readable)

    if first:
        functions.write_file(results_file + ".txt", readable)
        functions.write_file(results_file + ".tsv", header + tsv)
    else:
        functions.append_file(results_file + ".txt", readable)
        functions.append_file(results_file + ".tsv", tsv)




## Main part
if __name__ == '__main__':
    (N, corpusdir, cache_dir, vectors_dir, transformation, heatmap_dir, nntype, relations_file, output_dir, models_dir) = read_arguments()

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



    nr_of_heatmaps = 0
    trainingset_percentage = 80  # percentage of the dataset that is used for training
    X = [data[0] for data in ds]
    Y = [data[1] for data in ds]


    # Make a training set and a testset
    trainingset_size = int(len(X) * .80)
    X_test = X[(trainingset_size + 1):]
    X = X[:trainingset_size]
    Y_test = Y[(trainingset_size + 1):]
    Y = Y[:trainingset_size]

    plot_graph = False
    first = True
    for batch_size in [20, 50, 100]:
        for n_epochs in [10, 60, 100, 200]:
            for learning_rate in [0.001, 0.01, 0.1]:
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

                metrics = []
                skf = StratifiedKFold(n_splits=5)
                skf.get_n_splits(X, Y)
                for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
                    print(f"Fold {i}:")

                    TrainX = torch.tensor([X[i] for i in train_index], dtype=torch.float32, device=device)
                    TrainY = torch.tensor([[Y[i]] for i in train_index], dtype=torch.float32, device=device)
                    TestX  = torch.tensor([[X[i]] for i in test_index], dtype=torch.float32, device=device)
                    TestY  = torch.tensor([[Y[i]] for i in test_index], dtype=torch.float32, device=device)

                    loss_fn = nn.BCELoss()  # binary cross entropy
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                    batches_per_epoch = math.ceil(len(TrainX) / batch_size)

                    # Train the model
                    losses = []
                    for epoch in range(n_epochs):
                        for i in range( batches_per_epoch):
                            start = i * batch_size

                            # take a batch
                            Xbatch = TrainX[start:min(start + batch_size, len( TrainX))]
                            y_batch = TrainY[start:min(start + batch_size, len( TrainY))]

                            # forward pass
                            y_pred = model(Xbatch)
                            loss = loss_fn(y_pred, y_batch)

                            # backward pass
                            optimizer.zero_grad()
                            loss.backward()

                            # update weights
                            optimizer.step()

                    # Plot the loss graph.
                    if plot_graph:
                        plt.plot([epoch for epoch in range(n_epochs)], [loss for loss in losses])
                        plt.xlabel('Epoch')
                        plt.ylabel('Loss')
                        plt.show()

                    # get the metrics
                    this_metric = calculate_metrics( model, TestY, TestX)
                    metrics.append( this_metric)

                    # Save the model if wanted
                    if not models_dir is None:
                        model_file = f"{os.path.basename(relations_file).split('.')[0]}_{batch_size}_{n_epochs}_{learning_rate}.pt";
                        torch.save( model, os.path.join(models_dir, model_file))


                evaluate_the_model(
                model = model,
                X=X_test,
                Y=Y_test,
                metrics = metrics,
                results_file=results_file,
                first=first,
                batch_size=batch_size,
                n_epochs=n_epochs,
                learning_rate=learning_rate
            )


            first = False
