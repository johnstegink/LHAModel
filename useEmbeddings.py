# Use embeddings without any sections to determine the maximum score that can be obtained using embeddings

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
from Distances.DocumentRelations import DocumentRelations
from Distances.DocumentSectionRelations import DocumentSectionRelations
from network.SectionDataset import SectionDataset
from network.SectionDatasetStat import SectionDatasetStat
from network.SectionDatasetTest import SectionDatasetTest
from texts.corpus import Corpus


def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Create sematic links based on the embeddings only')
    parser.add_argument('-c', '--corpus_dir', help='The directory of the corpus to train on', required=True)
    parser.add_argument('-r', '--relations_file', help='The file with the relations and similarities', required=False)
    parser.add_argument('-o', '--results_dir', help='The director containing the results', required=True)

    args = vars(parser.parse_args())

    os.makedirs( args["results_dir"], exist_ok=True)

    return ( args["corpus_dir"], args["relations_file"], args["results_dir"])


def evaluate_the_model(  threshold, dr, pairs, results_file, first):
    """
    Make an evaluation
    :param first:
    :param threshold:
    :param pairs:       // All pairs
    :param results_file: // Where the results are written
    :return:
    """

    y_pred = []
    y_act = []
    for pair in pairs:
        src = pair.get_src()
        dest = pair.get_dest()
        sim = pair.get_similarity()

        pred_sim = dr.get_similarity( src, dest)
        if pred_sim == 0: continue

        y_act.append( True if sim == 1.0 else False)
        y_pred.append( True if pred_sim >= (threshold / 100) else False)

    (precision, recall, F1, _) =sklearn.metrics.precision_recall_fscore_support( y_act, y_pred, average="binary")
    accuracy = sklearn.metrics.accuracy_score( y_act, y_pred)

    tp = fp = fn = 0
    for i in range(0, len(y_pred)):
        if y_pred[i] and y_act[i] : tp += 1
        elif y_pred[i] and not y_act[i]: fp += 1
        elif not y_pred[i] and y_act[i]: fn += 1

    readable = f"Threshold\nAccuracy: {accuracy * 100:.2f}%\ntp: {tp}, \nfp:{fp}\nF1:{F1 * 100:.2f}\nPrecision: {precision}\nRecall: {recall}"
    tsv = f"{threshold}\t{F1}\t{accuracy}\t{precision}\t{recall}\t{tp}\t{fp}\t{fn}\n";
    header = "Threshold\tF1\tAccuracy\tPrecision\tRecall\tTrue Positives\tFalse Positives\t";
    print( readable)

    if first:
        functions.write_file(results_file + ".txt", readable)
        functions.write_file(results_file + ".tsv", header + tsv)
    else:
        functions.append_file(results_file + ".txt", readable)
        functions.append_file(results_file + ".tsv", tsv)



## Main part
if __name__ == '__main__':
    (corpusdir, relations_file, output_dir) = read_arguments()

    results_file = os.path.join( output_dir, f"results_{os.path.basename(relations_file).split('.')[0]}")


    print("Reading corpus")
    corpus = Corpus(directory=corpusdir)

    print("Reading document relations")
    (dr, dr_info) = DocumentRelations.read(relations_file)

    print("Reading document pairs")
    pairs = corpus.read_document_pairs(True)

    first = True
    for threshold in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        evaluate_the_model(
            threshold=threshold,
            dr = dr,
            pairs=pairs,
            results_file=results_file,
            first=first
        )
        first = False






