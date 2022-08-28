#! Create files for the LHA scripts, so they can be used with
# the tools on https://github.com/ninikolov/lha
# creates two files. One file with an ID per line, and one file with the text on one line
import argparse
import os

import functions
from texts.corpus import Corpus


def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Create files that can be used with the tools on https://github.com/ninikolov/lha"')
    parser.add_argument('-c', '--corpusdirectory', help='The corpus directory in the Common File Format', required=True)
    parser.add_argument('-o', '--output', help='Output directory where the files will be created with the name of corpus', required=True)
    args = vars(parser.parse_args())

    # Create the output directory if it doesn't exist
    outputdir = os.path.dirname(args["output"])
    os.makedirs( outputdir, exist_ok=True)


    return (args["corpusdirectory"], args["output"])


# Main part of the script
if __name__ == '__main__':
    (corpusdir, outputdir) = read_arguments()

    functions.show_message("Reading corpus")
    corpus = Corpus(directory=corpusdir)
    functions.show_message(f"The corpus contains {corpus.get_number_of_documents()} documents")


    ids = open( os.path.join(outputdir, corpus.get_name() + "_ids.txt"), mode="w", encoding="utf-8-sig")
    texts = open( os.path.join(outputdir, corpus.get_name() + "_texts.txt"), mode="w", encoding="utf-8-sig")

    for document in corpus:
        ids.write( document.get_id() + "\n")
        texts.write( document.get_fulltext_in_one_line() )

    ids.close()
    texts.close()
