# Script to create embeddings from a document, similar to the LHA algorithm (Nikola I. Nikolov and Richard H.R. Hahnloser)

import argparse

from texts.corpus import Corpus
from Distances.DistanceIndex import DistanceIndex
import sys
import os
import functions
from documentencoders.Sent2VecEncoder import Sent2VecEncoder
#from documentencoders.AvgWord2VecEncoder import AvgWord2VecEncoder
from Distances.DocumentVectors import  DocumentVectors
from tqdm import *

# import pydevd_pycharm
#
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Create document embeddings.')
    parser.add_argument('-c', '--corpusdirectory', help='The corpus directory in the Common File Format', required=True)
    parser.add_argument('-a', '--algorithm', help='The corpus directory in the Common File Format (default "sent2vec")', choices=["word2vec", "sent2vec"], default="sent2vec")
    parser.add_argument('-o', '--output', help='Output file for the xml file with the documentvectors', required=True)
    args = vars(parser.parse_args())

    corpusdir = args["corpusdirectory"] if "corpusdirectory" in args else None
    if corpusdir is not None and len( functions.read_all_files_from_directory(corpusdir , "xml")) == 0:
        sys.stderr.write(f"Directory '{corpusdir}' doesn't contain any files\n")
        exit( 2)

    # Create the output directory if it doesn't exist
    outputdir = os.path.dirname(args["output"])
    os.makedirs( outputdir, exist_ok=True)

    return (corpusdir, args["output"], args["algorithm"].lower())


def create_encoder( algorithm):
    """
    Create a embedding encoding based on the algorithm value
    :param algorithm:
    :return:
    """

    if algorithm == "sent2vec":
        encoder = Sent2VecEncoder(corpus.get_language_code())
    elif algorithm == "word2vec":
        encoder = AvgWord2VecEncoder(corpus.get_language_code())
    else:
        raise Exception(f"Unknown algorith: {algorithm}")

    print( f"Using {algorithm} ..." )
    return encoder


# Main part of the script
if __name__ == '__main__':
    (inputdir, output, algorithm) = read_arguments()

    functions.show_message("Reading corpus")
    corpus = Corpus(directory=inputdir)
    functions.show_message(f"The corpus contains {corpus.get_number_of_documents()} documents")

    functions.show_message("Loading encoder")
    encoder = create_encoder( algorithm)

    functions.show_message("Document vectors")
    documentvectors = DocumentVectors()
    with tqdm(total=corpus.get_number_of_documents(), desc="Total progress") as progress:
        for document in corpus:
            text = document.get_fulltext_in_one_line()
            if len( text) > 100:
                vector = encoder.embed_text(document.get_fulltext_in_one_line())
                documentvectors.add( document.get_id(), vector)

                # Add all sections
                for section in document:
                    section_text = section.get_fulltext_in_one_line()
                    vector = encoder.embed_text( section_text)
                    documentvectors.add_section( document.get_id(), vector)

            progress.update()

    functions.show_message("Save vectors")
    del encoder
    documentvectors.save( output)
    functions.show_message("Vectors saved")

