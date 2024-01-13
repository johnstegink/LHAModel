# Count the document pairs in all corpora

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
    parser.add_argument('-c', '--corpusdirectory', help='The directory with all corpora', required=True)
    args = vars(parser.parse_args())

    return (args["corpusdirectory"])


# Main part of the script
if __name__ == '__main__':
    (corpusdir) = read_arguments()

    for dir in os.listdir( corpusdir):
        if not "_" in dir or "." in dir: continue

        corpus = Corpus(directory=os.path.join( corpusdir, dir))

        counter = 0
        error = 0
        for pair in corpus.read_document_pairs():
            if( corpus.has_document( pair.get_src())  and corpus.has_document( pair.get_dest())):
                counter += 1
            else:
                error += 1

        print( f"{dir}\t{counter}\t{error}")
