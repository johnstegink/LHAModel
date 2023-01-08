# Script to train a smash model

import argparse

import sys
import os
import functions
from SMASH.DocumentPreprocessor import DocumentPreprocessor
from texts.corpus import Corpus
from texts.similarities import Similarities


def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Trans a SMASH model based on the corpus')
    parser.add_argument('-c', '--corpusdirectory', help='The corpus directory in the Common File Format', required=True)
    parser.add_argument('-p', '--pairs', help='XML file containing the shuffled pairs from corpus, if it exists, it won''t be created.')
    parser.add_argument('-o', '--model', help='Output file for the model', required=True)
    args = vars(parser.parse_args())

    corpusdir = args["corpusdirectory"] if "corpusdirectory" in args else None
    if corpusdir is not None and len( functions.read_all_files_from_directory(corpusdir , "xml")) == 0:
        sys.stderr.write(f"Directory '{corpusdir}' doesn't contain any files\n")
        exit( 2)

    functions.create_directory_for_file_if_not_exists( args["pairs"])
    functions.create_directory_for_file_if_not_exists( args["model"])

    return (corpusdir, args["pairs"], args["model"])


def readdocument_pairs(corpus, pairsfile):
    """
    Create a list of document pairs of the corpus, matching and non matching items are evenly distributed.
    If the file exists it will be read otherwise it writes the data to the outputfile
    :param corpus:
    :return: Similarities
    """
    if not os.path.exists(pairsfile):
        pairs = corpus.read_similarities()
        corpus.add_dissimilarities( pairs)

        pairs.save(file=pairsfile, shuffled=True)

    return Similarities.read(pairsfile)




# Main part of the script
if __name__ == '__main__':
    (corpusdir, pairs_file, model) = read_arguments()

    functions.show_message("Reading corpus")
    corpus = Corpus(directory=corpusdir)
    functions.show_message(f"The corpus contains {corpus.get_number_of_documents()} documents")

    functions.show_message("Creating document pairs")
    pairs = readdocument_pairs( corpus, pairs_file)

    # Preprocess the documents
    preprocessor = DocumentPreprocessor( corpus=corpus, similarities=pairs)
    (sections, sentences, words) = preprocessor.get_statistics()

    functions.show_message("Done")

