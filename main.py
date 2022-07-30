# Script to create embeddings from a document, similar to the LHA algorithm (Nikola I. Nikolov and Richard H.R. Hahnloser)

import argparse
from texts.corpus import Corpus
from tqdm import *
import sys
import functions


def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Create an index fo document embeddings.')
    parser.add_argument('-c', '--corpusdirectory', help='The corpus directory in the Common File Format', required=True)
    parser.add_argument('-n', '--name', help='Name of the corpus', required=True)
    parser.add_argument('-l', '--language', help='Language of the corpus, nl or en', required=True, choices=['en', 'nl'])
    parser.add_argument('-v', '--vectorsize', help='Size of the embedding vector', required=True, type=int)
    parser.add_argument('-o', '--output', help='Output directory for the index', required=True)
    args = vars(parser.parse_args())

    corpusdir = args["corpusdirectory"] if "corpusdirectory" in args else None
    if corpusdir is not None and len( functions.read_all_files_from_directory(corpusdir , "xml")) == 0:
        sys.stderr.write(f"Directory '{corpusdir}' doesn't contain any files\n")
        exit( 2)

    functions.create_directory_if_not_exists( args["output"])

    return (corpusdir, args["name"], args["language"], int(args["vectorsize"]), args["output"])


# Main part of the script
if __name__ == '__main__':
    (inputdir, name, language, vectorsize, outputdir) = read_arguments()
    corpus = Corpus(name=name, directory=inputdir)

    with tqdm(total=corpus.get_number_of_documents(), desc="Total progress") as progress:
        for document in corpus:
            progress.update()