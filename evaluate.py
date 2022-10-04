# Script to create embeddings from a document, similar to the LHA algorithm (Nikola I. Nikolov and Richard H.R. Hahnloser)

import argparse

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

    parser = argparse.ArgumentParser(description='Evaluate the score.')
    parser.add_argument('-c', '--corpusdirectory', help='The corpus directory in the Common File Format', required=True)
    parser.add_argument('-i', '--relationsxml', help='Xml file containing document relations', required=True)
    parser.add_argument('-s', '--similarity_1', help="ignore = ignore similarity 1, positive or negative", choices=["positive", "negative"], default="positive")
    parser.add_argument('-o', '--output', help="TSV file with output metrics", required=True)
    args = vars(parser.parse_args())

    corpusdir = args["corpusdirectory"] if "corpusdirectory" in args else None
    if corpusdir is not None and len( functions.read_all_files_from_directory(corpusdir , "xml")) == 0:
        sys.stderr.write(f"Directory '{corpusdir}' doesn't contain any files\n")
        exit( 2)

    return (corpusdir, args["relationsxml"], args["output"], args["similarity_1"])


def create_relations_grouped_on_src( relations):
    """
    Group all relations on the src
    :param relations:
    :return: dictionary with srcid as key and a list of relations as value
    """
    grouped_on_src = {}
    for relation in relations:
        src = relation.get_src()
        if not src in grouped_on_src:
            grouped_on_src[src] = []

        grouped_on_src[src].append( relation)

    return grouped_on_src


def create_related( grouped_on_src, topk ):
    """
    Create a dictionary with the srcid as key and a set of destinations as value
    By getting the topk values of the relations
    :param relations:
    :param topk:
    :return:
    """

    # determine the topk relations of every document
    generated = {}
    for (src, rels) in grouped_on_src.items():
        sorted_rels = sorted( rels, key=lambda x : x.get_similarity(), reverse=True)
        generated[src] = set()
        for i in range(0, min( topk, len(sorted_rels))):
            generated[src].add( sorted_rels[i].get_dest())

    return generated



def evaluate(corpus, sims, grouped_on_src, sim1, topk):
    """
    Evaluate the relations compared to the similarities
    :param similarities: the corpus
    :param relations: DocumentRelations object
    :param sim1: what to do with a similarity of 1, positive, negative
    :param topk: the top number of relevancy items to compare for calculation of the F1-value
    :return: mean f1
    """


    generated_all = create_related( grouped_on_src, topk)

    consider_1_as_positive = sim1 == "positive"
    f1s = []
    recalls = []
    precisions = []

    for src in corpus.get_ids():
        relevant = set( [sim.get_dest() for sim in sims.get_similiarties(src) if sim.get_similarity() == 2 or (consider_1_as_positive and sim.get_similarity() == 1)] )
        recommended = generated_all[src] if src in generated_all else set()

        relevant_items = float( len( recommended.intersection( relevant)))
        precision = relevant_items / float( len( recommended)) if len(recommended) > 0 else 1
        recall = relevant_items / float( len( relevant)) if len(relevant) > 0 else 1
        if (recall + precision) != 0:
            f1 = (2 * recall * precision) / (recall + precision)
        else:
            f1 = 0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append( f1)


    return (sum(f1s) / len(f1s), sum(precisions) / len(precisions), sum(recalls) /len(recalls))





# Main part of the script
if __name__ == '__main__':
    (inputdir, input, output, sim1) = read_arguments()

    functions.show_message("Reading corpus")
    corpus = Corpus(directory=inputdir)
    functions.show_message(f"The corpus contains {corpus.get_number_of_documents()} documents")

    functions.show_message("Reading similarities")
    sims = corpus.read_similarities()
    functions.show_message(f"The corpus contains {sims.count()} similarities")

    functions.show_message("Reading document relations")
    (relations, parameters) = DocumentRelations.read( input)
    grouped_on_src = create_relations_grouped_on_src( relations)
    functions.show_message(f"The corpus contains {relations.count()} relations")

    functions.show_message("Calculating score")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    if not os.path.isfile( output):
        with open(output, mode="w", encoding="utf-8-sig") as file:
            file.write("corpus\tlanguage\ttopk\tsim\tavgrel\t\tF1\tprecision\trecall\n")

    for topk in range(1, 25, 1):
        (F1, precision, recall) = evaluate(corpus, sims, grouped_on_src, sim1, topk)
        with open(output, mode="a", encoding="utf-8-sig") as file:
            file.write(f"{corpus.get_name()}\t{corpus.get_language()}\t{topk}\t{parameters['similarity']}\t{parameters['avgrel']}\t{F1}\t{precision}\t{recall}\n")

