# Script to create embeddings from a document, similar to the LHA algorithm (Nikola I. Nikolov and Richard H.R. Hahnloser)

import argparse

from texts.corpus import Corpus
from Distances.DocumentRelations import DocumentRelations
import sys
import functions


def create_key( src, dest):
    """
    Create a key of the source and destination
    :param src:
    :param dest:
    :return:
    """
    return src + "###" + dest


def evaluate( similarities, relations, sim1, topk):
    """
    Evaluate the relations compared to the similarities
    :param similarities: Similarities object
    :param relations: DocumentRelations object
    :param sim1: what to do with a similarity of 1, positive, negative
    :param topk: the top number of relevant items to compare for calculation of the F1-value
    :return: (tp, fp, fn, tn)   True positives, False positives, False negatives
    """

    rels = set()
    for relation in relations:
        key = create_key(relation.get_src(), relation.get_dest())
        rels.add( key )

    sim_pos = set()
    sim_neg = set()
    for sim in similarities:
        value = sim.get_similarity()
        key = create_key( sim.get_src(), sim.get_dest())
        if value == 2  or (value == 1 and sim1 == "positive"):
            sim_pos.add( key)
        elif value == 0 or (value == 1 and sim1 == "negative"):
            sim_neg.add( key)


    tp = len( rels.intersection(sim_pos))
    fp = len( rels.intersection(sim_neg))
    fn = len( sim_pos.difference( rels))

    return (tp, fp, fn)

def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Evaluate the score.')
    parser.add_argument('-c', '--corpusdirectory', help='The corpus directory in the Common File Format', required=True)
    parser.add_argument('-i', '--relationsxml', help='Xml file containing document relations', required=True)
    parser.add_argument('-k', '--topk', help='The number ', required=True, default=5, type=int)
    parser.add_argument('-s', '--similarity_1', help="ignore = ignore similarity 1, positive or negative", choices=["positive", "negative"], default="positive")
    args = vars(parser.parse_args())

    corpusdir = args["corpusdirectory"] if "corpusdirectory" in args else None
    if corpusdir is not None and len( functions.read_all_files_from_directory(corpusdir , "xml")) == 0:
        sys.stderr.write(f"Directory '{corpusdir}' doesn't contain any files\n")
        exit( 2)

    return (corpusdir, args["relationsxml"], args["similarity_1"], args["topk"])


# Main part of the script
if __name__ == '__main__':
    (inputdir, input, sim1, topk) = read_arguments()

    functions.show_message("Reading corpus")
    corpus = Corpus(directory=inputdir)
    functions.show_message(f"The corpus contains {corpus.get_number_of_documents()} documents")

    functions.show_message("Reading similarities")
    sim = corpus.read_similarities()
    functions.show_message(f"The corpus contains {sim.count()} similarities")

    functions.show_message("Reading document relations")
    relations = DocumentRelations.read( input)
    functions.show_message(f"The corpus contains {relations.count()} relations")


    functions.show_message("Calculating score")
    (tp, fp, fn) = evaluate(sim, relations, sim1, topk)
    F1 = float(tp) / (float(tp) + 0.5 * (float(fp) + float(fn))) * 100.0
    print(f"tp: {tp}")
    print(f"fp: {fp}")
    print(f"fn: {fn}")
    print(f"F1: {F1}")
