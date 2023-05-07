# Script to create relations the sections

import argparse

import sys
import os
import functions
from Distances.DocumentRelations import DocumentRelations
from Distances.DocumentVectors import  DocumentVectors
from Distances.SimilarityMatrix import  SimilarityMatrix
from texts.corpus import Corpus



def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Create document relations between the sections using the files created with "createvectors.py" and "createrelations.py"')
    parser.add_argument('-c', '--corpusdirectory', help='The corpus directory in the Common File Format', required=True)
    parser.add_argument('-i', '--documentvectorfile', help='The xml file containing the documentvectors', required=True)
    parser.add_argument('-r', '--relationsfiles', help='The xml file containing the relations between the documents', required=True)
    parser.add_argument('-s', '--similarity', help='Minimum similarity used to select the sections (actual similarity times 100)', required=True)
    parser.add_argument('-o', '--output', help='Output file for the xml file with the section relations', required=True)
    parser.add_argument('-d', '--html', help='Output file for readable html output (debug)', required=False)
    args = vars(parser.parse_args())

    # Create the output directory if it doesn't exist
    outputdir = os.path.dirname(args["output"])
    os.makedirs( outputdir, exist_ok=True)

    corpusdir = args["corpusdirectory"] if "corpusdirectory" in args else None
    if corpusdir is not None and len( functions.read_all_files_from_directory(corpusdir , "xml")) == 0:
        sys.stderr.write(f"Directory '{corpusdir}' doesn't contain any files\n")
        exit( 2)

    return (corpusdir, args["documentvectorfile"], args["relationsfiles"], int(args["similarity"]), args["output"], args["html"])


# Main part of the script
if __name__ == '__main__':
    (corpusdir, documentvectors, documentrelations, similarity, output, html) = read_arguments()

    functions.show_message("Reading document vectors")
    dv = DocumentVectors.read(documentvectors)

    functions.show_message("Reading document relations")
    dr = DocumentRelations.read(documentrelations)

    functions.show_message("Reading corpus")
    corpus = Corpus(directory=corpusdir)
    functions.show_message(f"The corpus contains {corpus.get_number_of_documents()} documents")

    for relation in dr:
        src = relation.get_src();
        dest = relation.get_dest()



    functions.show_message("Done")

