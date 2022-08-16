# Script to create relations based on documentvectors from two different corpora

import argparse

import sys
import os
import functions
from Distances.DocumentVectors import  DocumentVectors
from Distances.DistanceIndex import  DistanceIndex
from texts.corpus import Corpus



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


# Main part of the script
if __name__ == '__main__':
    (corpusdir1, corpusdir2, input1, input2, distance, output, html) = read_arguments()

    functions.show_message("Reading document vectors")
    dv1 = DocumentVectors.read(input1)
    dv2 = DocumentVectors.read(input2)
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
    relations = distance_index1.calculate_relations( (float(distance) / 100.0), nearest_lim=2, second_index=distance_index2)
    relations.save( output)
    if not html is None:
        relations.save_html( corpus1, html, corpus2)

    functions.show_message("Done")

    a = 0

