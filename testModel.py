# Test the model for phase 3 of the SADDLE algorithm

import argparse
import math
import os

import sklearn
import torch
import torch.nn.functional as F
from models import nn_stat
from models import nn_plain

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

    parser = argparse.ArgumentParser(description='Test the model')
    parser.add_argument('-N', '--sections', help='The number of sections to consider', required=True, type=int)
    parser.add_argument('-c', '--corpus_dir', help='The directory of the corpus to be tested', required=True)
    parser.add_argument('-mo', '--model_file', help='The model to be tested', required=True)
    parser.add_argument('-nn', '--neuralnetworktype', help='The type of neural network', required=True, choices=["plain", "stat"])
    parser.add_argument('-r', '--relationsfile', help='The xml file containing the relations between the documents', required=False)
    parser.add_argument('-t', '--transformation', help='The transformation to apply to sections with an index larger than N', required=F, choices=['truncate', 'avg'])
    parser.add_argument('-s', '--cache_dir', help='The directory used for caching', required=True)
    parser.add_argument('-v', '--vectors_dir', help='The directory containing the document and section vectors of this corpus', required=False)

    args = vars(parser.parse_args())

    return ( args["sections"], args["corpus_dir"], args["model_file"], args["neuralnetworktype"], args["relationsfile"], args["transformation"], args["cache_dir"], args["vectors_dir"])



def calculate_metrics(model, Y_test, X_test):
    """
    Calculate the metrics and return a list with the metrics
    :param model:
    :param y:
    :param x_test:
    :return: [F1, accuracy, precision, recall, tp, fp, fn]
    """

    # change the y_pred to 0 or 1
    y_pred = [True if model( x) >= 0.5 else False for x in X_test]
    y_act = [True if y >= 0.5 else False for y in Y_test]

    (precision, recall, F1, _) =sklearn.metrics.precision_recall_fscore_support( y_act, y_pred, average="binary", zero_division=0.0)
    accuracy = sklearn.metrics.accuracy_score( y_act, y_pred)

    tp = fp = fn = 0
    for i in range(0, len(y_pred)):
        if y_pred[i] and y_act[i] : tp += 1
        elif y_pred[i] and not y_act[i]: fp += 1
        elif not y_pred[i] and y_act[i]: fn += 1

    return [F1, accuracy, precision, recall, tp, fp, fn]



## Main part
if __name__ == '__main__':
    (N, corpus_dir, model_file, nntype, relations_file, transformation, cache_dir, vectors_dir) = read_arguments()

    device = torch.device("cpu")
    cache_file = os.path.join( cache_dir, f"dataset_{N}_{nntype}.pkl")

    if (nntype == "stat"):
        threshold = 0.3
        ds = SectionDatasetStat(device=device, corpus_dir=corpus_dir, relations_file=relations_file, top_threshold=threshold, cache_file=cache_file, train=False)
    else:
        ds = SectionDataset(N=N, device=device, corpus_dir=corpus_dir,
                            documentvectors_dir=vectors_dir, transformation=transformation, nntype=nntype,
                            cache_file=cache_file, train=False)



    X = [data[0] for data in ds]
    Y = [data[1] for data in ds]

    # Define the model
    if nntype == "plain":
        model = nn_plain.NeuralNetworkPlain(N)
    elif nntype == "stat":
        model = nn_stat.NeuralNetworkStat(4)
    else:
        raise f"Unknown neural network type: {nntype}"

    torch.load(model_file)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32, device=device)

    [F1, accuracy, precision, recall, tp, fp, fn] = calculate_metrics( model, Y_tensor, X_tensor)

    # Read the info from the vectordir name
    parts = os.path.basename(vectors_dir).split('_')
    lang = parts[1]
    corpus = parts[0]
    method = parts[2][:-4]
    print(f"{lang} | {corpus}  | {method} | {F1:.2f} | {accuracy:.2f}")