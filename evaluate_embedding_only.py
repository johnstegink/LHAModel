# Script to evalute the embedding algortihm

import argparse

import sys
import os

from tqdm import tqdm

import functions
from Distances.DocumentVectors import  DocumentVectors
from Distances.DistanceIndex import  DistanceIndex
from texts.corpus import Corpus



def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Evaluate the similarity, only based on the embedding algorithm "createvectors.py"')
    parser.add_argument('-c', '--corpusdirectory', help='The corpus directory in the Common File Format', required=True)
    parser.add_argument('-i', '--documentvectorfile', help='The xml file containing the documentvectors', required=True)
    parser.add_argument('-s', '--similarity', help='Minimum similarity between the files (actual similarity times 100)', required=True)
    args = vars(parser.parse_args())

    corpusdir = args["corpusdirectory"] if "corpusdirectory" in args else None
    if corpusdir is not None and len( functions.read_all_files_from_directory(corpusdir , "xml")) == 0:
        sys.stderr.write(f"Directory '{corpusdir}' doesn't contain any files\n")
        exit( 2)

    return (corpusdir, args["documentvectorfile"], int(args["similarity"]))


# Main part of the script
if __name__ == '__main__':
    (corpusdir, input, similarity) = read_arguments()

    functions.show_message("Reading document vectors")
    dv = DocumentVectors.read(input)
    distance_index = DistanceIndex( dv)
    distance_index.build()

    functions.show_message("Reading corpus")
    corpus = Corpus(directory=corpusdir)
    functions.show_message(f"The corpus contains {corpus.get_number_of_documents()} documents")

    correct = 0.0
    fp = 0.0
    tp = 0.0
    tn = 0.0
    fn = 0.0
    pairs = corpus.read_document_pairs()
    total = pairs.count()
    with tqdm(total=total, desc="Predicting") as progress:
        for pair in pairs:
            src = pair.get_src()
            dest = pair.get_dest()
            sim = pair.get_similarity()

            cosine_sim = distance_index.cosine_sim(src, dest)
            if not cosine_sim is None:
                prediction = 1 if sim * 100 >= similarity else 0
            else:
                print( "None")
                prediction = 0


            if prediction == 0 and int(sim) == 1:
                fp += 1.0

            elif prediction == 1 and int(sim) == 1:
                correct += 1
                tp += 1.0

            elif prediction == 0 and int(sim) == 1:
                fn += 1.0

            else:
                correct += 1.0
                tn += 1.0

            progress.update()

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    F1 = (2 * recall * precision) / (recall + precision)
    accuracy = int((correct / total) * 100)
    print(f"Accuracy: {accuracy}%\ntp: {tp}, \nfp:{fp}\nF1:{F1}\nPrecision: {precision}\nRecall: {recall}")

    functions.show_message("Done")

