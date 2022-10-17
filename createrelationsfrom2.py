# Script to create relations based on documentvectors from two different corpora

import argparse

import sys
import os
import functions
from Distances.DocumentRelations import DocumentRelations
from Distances.DocumentVectors import  DocumentVectors
from Distances.DistanceIndex import  DistanceIndex
from texts.corpus import Corpus
import re



def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Create document relations based on the documentvectors that were created with "createvectors.py"')
    parser.add_argument('-c1', '--corpusdirectory1', help='The first corpus directory in the Common File Format', required=True)
    parser.add_argument('-c2', '--corpusdirectory2', help='The second corpus directory in the Common File Format', required=True)
    parser.add_argument('-i1', '--documentvectorfile1', help='The first xml file containing the documentvectors', required=True)
    parser.add_argument('-i2', '--documentvectorfile2', help='The second xml file containing the documentvectors', required=True)
    parser.add_argument('-d', '--distance', help='Minimum distance between the files (actual distance times 100)', required=True)
    parser.add_argument('-o', '--output', help='Output file for the xml file with the document relations', required=True)
    parser.add_argument('-r', '--html', help='Output file for readable html output', required=False)
    args = vars(parser.parse_args())

    # Create the output directory if it doesn't exist
    outputdir = os.path.dirname(args["output"])
    os.makedirs( outputdir, exist_ok=True)


    return (args["corpusdirectory1"], args["corpusdirectory2"], args["documentvectorfile1"], args["documentvectorfile2"], int(args["distance"]), args["output"], args["html"])

def print_scores( relations, src_corpus, dst_corpus):
    """
    Print the number of true positives
    :param relations:
    :param src_corpus:
    :param dst_corpus:
    :return:
    """
    tp = 0.0
    tn = 0.0
    for relation in relations:
        src = src_corpus.getDocument(relation.get_src())
        dst = dst_corpus.getDocument(relation.get_dest())

        src_id = src.get_id()
        dst_id = dst.get_id()

        if( src_id.startswith("a_") or dst_id.startswith("a_")):
            if( src_id == dst_id):
                tp += 1.0
            else:
                tn += 1.0

    perc = round((tp * 100) / (tn + tp))
    print(f"TP: {int(tp)}")
    print(f"TN: {int(tn)}")
    print(f"Percentage: {perc}")



# Main part of the script
if __name__ == '__main__':
    (corpusdir1, corpusdir2, input1, input2, distance, output, html) = read_arguments()

    functions.show_message("Reading document vectors")

    id_filter = re.compile(r"^[ar]")
    dv1 = DocumentVectors.read(input1, id_filter=id_filter)
    dv2 = DocumentVectors.read(input2, id_filter=id_filter)
    distance_index1 = DistanceIndex( dv1)
    distance_index2 = DistanceIndex( dv2)

    functions.show_message("Reading corpora")
    corpus1 = Corpus(directory=corpusdir1)
    functions.show_message(f"The first corpus contains {corpus1.get_number_of_documents()} documents")
    corpus2 = Corpus(directory=corpusdir2)
    functions.show_message(f"The second corpus contains {corpus2.get_number_of_documents()} documents")

    functions.show_message("Building indexes")
    distance_index1.build()
    distance_index2.build()

    functions.show_message("Calculating distances")
    relations = distance_index1.calculate_relations_less_slow( (float(distance) / 100.0), second_index=distance_index2, maximum_number_of_results=1)
    # relations = distance_index1.calculate_relations( (float(distance) / 100.0), second_index=distance_index2, maximum_number_of_results=1)
    relations.save( output, {})
    (relations, params) = DocumentRelations.read( output)
    if not html is None:
        relations.save_html( corpus1, html, corpus2, "a_")

    print_scores( relations, corpus1, corpus2)

    functions.show_message("Done")

