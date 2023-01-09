# Script to train a smash model

import argparse

import sys
import os
import functions
from SMASH.DocumentPreprocessor import DocumentPreprocessor
from texts.corpus import Corpus
from texts.similarities import Similarities
import plotly.express as px
import plotly.graph_objects as go
import time

def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Trans a SMASH model based on the corpus')
    parser.add_argument('-c', '--corpusdirectory', help='The corpus directory in the Common File Format', required=True)
    parser.add_argument('-p', '--pairs', help='XML file containing the shuffled pairs from corpus, if it exists, it won''t be created.', required=True)
    parser.add_argument('-i', '--histogramdir', help='Directory to which the histograms will be written', required=False)
    parser.add_argument('-c1', '--sections', help='Maximum number of sections per document', required=False, type=int, default=8)
    parser.add_argument('-c2', '--sentences', help='Maximum number of sentences per section', required=False, type=int, default=30)
    parser.add_argument('-c3', '--histogramdir', help='Maximum number of words per sentence', required=False, type=int, default=50)
    parser.add_argument('-o', '--model', help='Output file for the model', required=True)
    args = vars(parser.parse_args())

    corpusdir = args["corpusdirectory"] if "corpusdirectory" in args else None
    if corpusdir is not None and len( functions.read_all_files_from_directory(corpusdir , "xml")) == 0:
        sys.stderr.write(f"Directory '{corpusdir}' doesn't contain any files\n")
        exit( 2)

    functions.create_directory_for_file_if_not_exists( args["pairs"])
    functions.create_directory_for_file_if_not_exists( args["model"])

    return (corpusdir, args["pairs"], args["model"], args["histogramdir"])


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

def create_histograms( preprocessor, imagedir):
    """
    Creates the barcharts for thi
    :param imagedir:
    :return:
    """

    (sections, sentences, words) = preprocessor.get_statistics()
    create_histogram( sections, "Sections", "sections", os.path.join( imagedir, "sections_histogram.pdf"))
    create_histogram( sentences, "Sentences", "sentences", os.path.join( imagedir, "sentences_histogram.pdf"))
    create_histogram( words, "Words", "words", os.path.join( imagedir, "words_histogram.pdf"))


def create_histogram(data, title, xasis, imagefile):
    """
    Create a new barchart with the statistics
    :param data: dictionary with the counts
    :param title: title of the grpah
    :param xasis: title of the x-asis
    :param imagefile: pathname to the imagefile
    :return: 
    """""
    graphcolors = {
        "main": "rgb(255, 126, 132)",
        "sub1":  "rgb(129, 0, 0)",
        "sub2":  "rgb(93, 138, 216)",
        "background": "rgb(240, 238, 238)"
    }

    xs = sorted( data.keys())
    ys = [data[x] for x in xs]

    plot = go.Figure(data=[go.Bar(
            name=title,
            x = xs,
            y = ys,
            marker_color=graphcolors["main"]
    ),
    ])

    # Workaround to remove error message from pdf
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig.write_image(imagefile, format="pdf")
    time.sleep(2)

    plot.update_layout(
        {"xaxis": {
            "title": "Number of " + xasis
        },
        "yaxis": {
            "title": "Number of occurences"
        }
    })
    plot.write_image(imagefile)


# Main part of the script
if __name__ == '__main__':
    (corpusdir, pairs_file, model, imgdir) = read_arguments()

    functions.show_message("Reading corpus")
    corpus = Corpus(directory=corpusdir)
    functions.show_message(f"The corpus contains {corpus.get_number_of_documents()} documents")

    functions.show_message("Creating document pairs")
    pairs = readdocument_pairs( corpus, pairs_file)

    # Preprocess the documents
    preprocessor = DocumentPreprocessor( corpus=corpus, similarities=pairs)
    if not imgdir is None:
        functions.show_message("Creating histograms")
        create_histograms( preprocessor, imgdir)

    functions.show_message("Done")

