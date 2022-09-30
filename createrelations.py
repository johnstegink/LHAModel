# Script to create relations based on documentvectors

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
    parser.add_argument('-c', '--corpusdirectory', help='The corpus directory in the Common File Format', required=True)
    parser.add_argument('-i', '--documentvectorfile', help='The xml file containing the documentvectors', required=True)
    parser.add_argument('-s', '--similarity', help='Minimum similarity between the files (actual similarity times 100)', required=True)
    parser.add_argument('-o', '--output', help='Output file for the xml file with the document relations', required=True)
    parser.add_argument('-r', '--html', help='Output file for readable html output', required=False)
    args = vars(parser.parse_args())

    # Create the output directory if it doesn't exist
    outputdir = os.path.dirname(args["output"])
    os.makedirs( outputdir, exist_ok=True)

    corpusdir = args["corpusdirectory"] if "corpusdirectory" in args else None
    if corpusdir is not None and len( functions.read_all_files_from_directory(corpusdir , "xml")) == 0:
        sys.stderr.write(f"Directory '{corpusdir}' doesn't contain any files\n")
        exit( 2)

    return (corpusdir, args["documentvectorfile"], int(args["similarity"]), args["output"], args["html"])


# Main part of the script
if __name__ == '__main__':
    (corpusdir, input, similarity, output, html) = read_arguments()

    functions.show_message("Reading document vectors")
    dv = DocumentVectors.read(input)
    distance_index = DistanceIndex( dv)

    functions.show_message("Reading corpus")
    corpus = Corpus(directory=corpusdir)
    functions.show_message(f"The corpus contains {corpus.get_number_of_documents()} documents")

    functions.show_message("Building index")
    distance_index.build()

    functions.show_message("Calculating distances")
    relations = distance_index.calculate_relations_less_slow((float(similarity) / 100.0))
    relations.save( output)
    if not html is None:
        relations.save_html( corpus, html)

    functions.show_message("Done")

