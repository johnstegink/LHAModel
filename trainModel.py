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
from sklearn.model_selection import StratifiedKFold
from models import nn_stat
from models import nn_mask
from models import nn_plain
from models import nn_test

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
    parser.add_argument('-nn', '--neuralnetworktype', help='The type of neural network', required=True, choices=["plain", "stat", "test"])
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

    (precision, recall, F1, _) =sklearn.metrics.precision_recall_fscore_support( y_act, y_pred, average="binary", zero_division=0.0)
    accuracy = sklearn.metrics.accuracy_score( y_act, y_pred)

    tp = fp = fn = 0
    for i in range(0, len(y_pred)):
        if y_pred[i] and y_act[i] : tp += 1
        elif y_pred[i] and not y_act[i]: fp += 1
        elif not y_pred[i] and y_act[i]: fn += 1



    return [F1, accuracy, precision, recall, tp, fp, fn]



def mean_of_column( metrics, column):
    """
    Calculates the mean of the given column in the metrics list (two dimensional)
    :param metrics:
    :param column:
    :return:
    """

    values = [metric[column] for metric in metrics]
    return sum(values) / len( values)


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

    F1 = mean_of_column(metrics, 0)
    accuracy = mean_of_column(metrics, 1)
    precision = mean_of_column(metrics, 2)
    recall = mean_of_column(metrics, 3)
    tp = mean_of_column(metrics, 4)
    fp = mean_of_column(metrics, 5)
    fn = mean_of_column(metrics, 6)

    readable = f"Batch size: {batch_size}, Epochs: {n_epochs}, Learning rate: {learning_rate}, F1: {F1 * 100:.0f} Accuracy: {accuracy * 100:.0f}%"
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
        ds = SectionDatasetStat(device=device, corpus_dir=corpusdir, relations_file=relations_file, top_threshold=threshold, cache_file=cache_file, train=True)
    else:
        ds = SectionDataset(N=N, device=device, corpus_dir=corpusdir,
                                  documentvectors_dir=vectors_dir, transformation=transformation, nntype=nntype,
                                  cache_file=cache_file, train=True)



    nr_of_heatmaps = 0
    X = [data[0] for data in ds]
    Y = [data[1] for data in ds]


    # Make a training set and a testset
    trainingset_size = int(len(X) * .80)

    plot_graph = False
    first = True
    for batch_size in [20, 50, 100]:
        for n_epochs in [10, 60, 100, 200]:
            for learning_rate in [0.001, 0.01, 0.1]:
                # Define the model
                if nntype == "plain":
                    model = nn_plain.NeuralNetworkPlain(N)
                elif nntype == "masked":
                    model = nn_mask.NeuralNetworkMask(N)
                elif nntype == "stat":
                    model = nn_stat.NeuralNetworkStat(4)
                elif nntype == "test":
                    model = nn_test.NeuralNetworkTest(4)
                else:
                    raise f"Unknown neural network type: {nntype}"
                model.to(device)

                metrics = []
                skf = StratifiedKFold(n_splits=5)
                skf.get_n_splits(X, Y)
                for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
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
                        model_file = f"{os.path.basename(relations_file).split('.')[0]}_{N}_{batch_size}_{n_epochs}_{learning_rate}.pt";
                        torch.save( model, os.path.join(models_dir, model_file))


                evaluate_the_model(
                model = model,
                X=X,
                Y=Y,
                metrics = metrics,
                results_file=results_file,
                first=first,
                batch_size=batch_size,
                n_epochs=n_epochs,
                learning_rate=learning_rate
            )


            first = False
