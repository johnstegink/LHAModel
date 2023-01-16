# Script to train a smash model

import argparse
import logging
import sys
import os

from tqdm import tqdm

import functions
from SMASH.DocumentPreprocessor import DocumentPreprocessor
from texts.corpus import Corpus
from texts.similarities import Similarities
import plotly.express as px
import plotly.graph_objects as go
import time
from SMASH.Siamese_network_trainer import SiameseNetworkTrainer

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Trans a SMASH model based on the corpus')
    parser.add_argument('-c', '--corpusdirectory', help='The corpus directory in the Common File Format', required=True)
    parser.add_argument('-s', '--cachedir', help='The directory in which files will be cached per language', required=True)
    parser.add_argument('-i', '--histogramdir', help='Directory to which the histograms will be written', required=False)
    parser.add_argument('-c1', '--sections', help='Maximum number of sections per document', required=False, type=int, default=6)
    parser.add_argument('-c2', '--sentences', help='Maximum number of sentences per section', required=False, type=int, default=15)
    parser.add_argument('-c3', '--words', help='Maximum number of words per sentence', required=False, type=int, default=30)
    parser.add_argument('-e', '--elmomodel', help='Directory containing the ELMO models per language', required=True)
    parser.add_argument('-o', '--model', help='Output file for the model', required=True)
    args = vars(parser.parse_args())

    corpusdir = args["corpusdirectory"] if "corpusdirectory" in args else None
    if corpusdir is not None and len( functions.read_all_files_from_directory(corpusdir , "xml")) == 0:
        sys.stderr.write(f"Directory '{corpusdir}' doesn't contain any files\n")
        exit( 2)

    functions.create_directory_for_file_if_not_exists( args["cachedir"])
    functions.create_directory_for_file_if_not_exists( args["model"])

    return (corpusdir, args["cachedir"], args["model"], args["elmomodel"], args["histogramdir"], args["sections"], args["sentences"], args["words"])


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

        functions.create_directory_for_file_if_not_exists(pairsfile)
        pairs.save(file=pairsfile, shuffled=True)

    return Similarities.read(pairsfile)

def create_histograms( preprocessor, imagedir):
    """
    Creates the barcharts for thi
    :param imagedir:
    :return:
    """

    (sections, sentences, words) = preprocessor.get_statistics()
    create_graphs(sections, "Sections", "sections", os.path.join(imagedir, "sections_histogram.pdf"), os.path.join(imagedir, "sections_cummulative.pdf"))
    create_graphs(sentences, "Sentences", "sentences", os.path.join(imagedir, "sentences_histogram.pdf"), os.path.join(imagedir, "sections_cummulative.pdf"))
    create_graphs(words, "Words", "words", os.path.join(imagedir, "words_histogram.pdf"), os.path.join(imagedir, "words_cummulative.pdf"))


def create_graphs(data, title, xasis, histogram_image, cummulative_image):
    """
    Create a new barchart and linechart with the statistics
    :param data: dictionary with the counts
    :param title: title of the grpah
    :param xasis: title of the x-asis
    :param histogram_image: pathname to the imagefile with the histogra
    :param cummulative_image: pathname to the imagefile with the cummulative score
    :return: 
    """""
    graphcolors = {
        "main": "rgb(255, 126, 132)",
        "sub1":  "rgb(129, 0, 0)",
        "sub2":  "rgb(93, 138, 216)",
        "background": "rgb(240, 238, 238)"
    }

    # Workaround to remove error message from pdf
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig.write_image(histogram_image, format="pdf")
    time.sleep(2)


    (histogram_xs, histogram_ys) = create_histogram(data, graphcolors, histogram_image, title, xasis)
    create_cummulative(cummulative_image, graphcolors, histogram_xs, histogram_ys, title, xasis)


def create_histogram(data, graphcolors, histogram_image, title, xasis):
    """
    Create a histogram of the data
    :param data:
    :param graphcolors:
    :param histogram_image:
    :param title:
    :param xasis:
    :return:
    """
    histogram_xs = sorted([x for x in data.keys() if x < 100])
    histogram_ys = [data[x] for x in histogram_xs]
    histogram = go.Figure(data=[go.Bar(
        name=title,
        x=histogram_xs,
        y=histogram_ys,
        marker_color=graphcolors["main"]
    ),
    ])
    histogram.update_layout(
        {"xaxis": {
            "title": "Number of " + xasis
        },
            "yaxis": {
                "title": "Number of occurences"
            }
        })
    histogram.write_image(histogram_image)
    return histogram_xs, histogram_ys


def create_cummulative(cummulative_image, graphcolors, histogram_xs, histogram_ys, title, xasis):
    """
    Create the graph with the cummulative data
    :param cummulative_image:
    :param graphcolors:
    :param histogram_xs:
    :param histogram_ys:
    :param title:
    :param xasis:
    :return:
    """
    cummulative_xs = histogram_xs
    cummulative_ys = []
    total = sum(histogram_ys)
    current_sum = 0
    for x in range(len(cummulative_xs)):
        current_sum += histogram_ys[x]
        cummulative_ys.append((current_sum * 100) / total)  # Calculate the percentage
    cummulative = go.Figure(data=[go.Line(
        name=title,
        x=cummulative_xs,
        y=cummulative_ys,
        marker_color=graphcolors["main"]
    ),
    ])
    cummulative.update_layout(
        {"xaxis": {
            "title": "Number of " + xasis
        },
            "yaxis": {
                "title": "Cummulative number of occurences"
            }
        })
    cummulative.write_image(cummulative_image)


# Main part of the script
if __name__ == '__main__':
    (corpusdir, cache_base_dir, model, elmo_model_dir, imgdir, max_sections, max_sentences, max_words) = read_arguments()

    functions.show_message("Reading corpus")
    corpus = Corpus(directory=corpusdir)
    functions.show_message(f"The corpus contains {corpus.get_number_of_documents()} documents")

    cache_dir = os.path.join(cache_base_dir, corpus.language_code)
    os.makedirs( cache_base_dir, exist_ok=True)

    functions.show_message("Creating document pairs")
    pairs = readdocument_pairs( corpus, os.path.join( cache_dir, "pairs.xml"))

    # Preprocess the documents
    preprocessor = DocumentPreprocessor( corpus=corpus,
                                         similarities=pairs,
                                         elmomodel=os.path.join(elmo_model_dir, corpus.language_code),
                                         embeddingsdir=cache_dir,
                                         max_sections=max_sections,
                                         max_sentences=max_sentences,
                                         max_words = max_words)

    # documentids = sorted( preprocessor.get_all_documentids())
    # for id in tqdm(documentids, desc="Creating embeddings"):
    #     preprocessor.create_or_load_embedding(id=id)

    trainer = SiameseNetworkTrainer()
    trainer.train( preprocessor, 3)



    if not imgdir is None:
        functions.show_message("Creating histograms")
        create_histograms( preprocessor, imgdir)

    functions.show_message("Done")

