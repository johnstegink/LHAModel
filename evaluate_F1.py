# Script to evaluate based on the F1-score
import argparse

import pandas
import sklearn.metrics

from texts.corpus import Corpus
from Distances.DocumentRelations import DocumentRelations
from texts.similarities import Similarities
import sys
import os
import functions


def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Evaluate the relations based on the F1 score')
    parser.add_argument('-c', '--corpusdirectory', help='The corpus directory in the Common File Format', required=True)
    parser.add_argument('-r', '--relationsxml', help='Xml file containing document relations', required=True)
    parser.add_argument('-t', '--iteration', help='Iteration, is put in filename for thesis', required=True)
    parser.add_argument('-l', '--latexdir', help='Directory containing the images for the latex version of the thesis', required=True)
    args = vars(parser.parse_args())

    corpusdir = args["corpusdirectory"] if "corpusdirectory" in args else None
    if corpusdir is not None and len( functions.read_all_files_from_directory(corpusdir , "xml")) == 0:
        sys.stderr.write(f"Directory '{corpusdir}' doesn't contain any files\n")
        exit( 2)

    return (corpusdir, args["relationsxml"], args["iteration"], args["latexdir"])

def evaluate(corpus, relations):
    """
    Evaluate the relations compared to the similarities, creates a pandas datatable with folowing columns
    simvalue,F1,precision,recall
    :param similarities: the corpus
    :param relations: DocumentRelations object
    :return: datatable
    """

    df = pandas.DataFrame(columns=["F1", "precision", "recall"])

    rels = {}
    corpus_sims = corpus.read_similarities()

    for relation in relations:
        rels[f"{relation.get_src()}#{relation.get_dest()}"] = relation.get_similarity()

    for treshold in range(10, 99):
        y_true = []
        y_pred = []
        tp = fp = tn = fn = 0
        for src in corpus.get_ids():
            for src_sim in corpus_sims.get_similiarties( src):
                dest = src_sim.get_dest()
                corpus_similarity = src_sim.get_similarity()

                # Find the similarity in the relations
                key = f"{src}#{dest}"
                rel_similarity = 1 if key in rels and rels[key] >= (treshold / 100) else 0

                y_true.append(corpus_similarity)
                y_pred.append( rel_similarity)

        #         if corpus_similarity == 1:
        #             if rel_similarity == 1:
        #                 tp += 1
        #             else:
        #                 fn += 1
        #         else:
        #             if rel_similarity == 0:
        #                 tn += 1
        #             else:
        #                 fp += 1
        #
        # precision = tp / (tp + fp)
        # recall = tp / (fn + tp)
        # f1 = 2 * (precision * recall) / (precision + recall)
        f1 = sklearn.metrics.f1_score( y_true, y_pred)
        print(f1)
        # df.append( {"F1": f1, "precision": precision, "recall": recall})

    return df



# Main part of the script
if __name__ == '__main__':
    (inputdir, input, iteration, latexdir) = read_arguments()

    functions.show_message("Reading corpus")
    corpus = Corpus(directory=inputdir)
    functions.show_message(f"The corpus contains {corpus.get_number_of_documents()} documents")

    functions.show_message("Reading document relations")
    (relations, parameters) = DocumentRelations.read( input)
    functions.show_message(f"The corpus contains {relations.count()} relations")

    dt = evaluate(corpus, relations)
