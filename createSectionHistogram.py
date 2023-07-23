import argparse
import os
import numpy
import plotly.express as px
import plotly.graph_objects as go
import time

from texts.corpus import Corpus


def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Create a histogram containing the count of the sections per corpus')
    parser.add_argument('-cd', '--corpusdir', help='Directory containing all corpora', required=True)
    parser.add_argument('-m', '--max', help='The maximum number of sections per document', required=True, type=int)
    parser.add_argument('-i', '--histogramimage', help='The filename to which the histogram will be written', required=True)

    args = vars(parser.parse_args())

    return ( args["corpusdir"], args["max"], args["histogramimage"])


def create_DOT( graph, word, max, prefix):
    counter = 0
    prolog = ""
    dot = ""
    node = f"{prefix}_0"
    prolog += f'{node}[label="{word}"]\n'
    nbs = graph.neighbors( word)
    for i in range(0, min(max, len(nbs))):
        nb = f"{prefix}_{i+1}"
        prolog += f'{nb}[label="{nbs[i]}"]\n'
        dot += f'{node} -> {nb}\n'

    print(prolog + "\n\n\n")
    print(dot + "\n------------------------\n\n\n")

def read_subdirs( dir):
    """
    Reads all subdirectories from the directory
    :param dir: the root directory
    :return: a list of subdirectories
    """

    folders = []
    for entry in os.listdir(dir):
        fullpath = os.path.join(dir, entry)
        if os.path.isdir( fullpath):
            folders.append( fullpath)

    return folders


def count( corpus, max):
    """
    Counts the number of sections per document in a corpus
    :param corpus: The corpus to be counted
    :return: an array containing the nr of documents per count i.e.
             [0,2,5] means 0 documents have 0 sections, 2 documents with 1 section and 5 documents with 3 sections

    """

    counts = [0] * (max + 1)
    for document in corpus:
        nr_of_sections = 0
        for section in document.sections:
            nr_of_sections += 1

        # Cannot be above the max
        if nr_of_sections >= max:
            nr_of_sections = max

        # Make sure the array is large enough
        counts[nr_of_sections] += 1

    return counts

def create_histogram(data, max, histogram_image, title):
    """
    Create a histogram of the data
    :param data:
    :param max:
    :param histogram_image:
    :param title:
    :return:
    """

    graphcolors = {
        "main": "rgb(255, 126, 132)",
        "sub1":  "rgb(129, 0, 0)",
        "sub2":  "rgb(93, 138, 216)",
        "background": "rgb(240, 238, 238)"
    }


    histogram_xs = [x for x in range(0, max + 1)]
    histogram_ys = data
    histogram = go.Figure(data=[go.Bar(
        name=title,
        x=histogram_xs,
        y=histogram_ys,
        marker_color=graphcolors["main"]
    ),
    ])
    histogram.update_layout(
        {"xaxis": {
            "title": "Number of sections per document"
        },
        "yaxis": {
            "title": "Number of occurences"
        }
    })


    # Workaround to remove error message from pdf
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig.write_image(histogram_image, format="pdf")
    time.sleep(2)

    histogram.show()

    histogram.write_image(histogram_image)


## Main part
if __name__ == '__main__':
    (corpusdir, max, histogram_image) = read_arguments()

    # List all corpora in the given directory
    corpusdirs = sorted(read_subdirs(corpusdir), key=str.casefold)
    counts = {}

    # debugging is faster
    # corpusdirs.pop(0)
    # corpusdirs.pop(0)


    corpora_names = ""
    for i in range(0, len(corpusdirs)):
        corpus = Corpus(directory=corpusdirs[i])
        name = corpus.get_name()
        print(f"Counting sections in {name} ...")
        counts[name] = count( corpus, max)

        if i == len( corpusdirs) - 1:
            corpora_names += " and "
        elif i > 0:
            corpora_names += ", "
        corpora_names += name


    # Sum all counts in one list
    total = [0] * (max + 1)
    for vals in counts.values():
        total = numpy.add( total, vals)
    total = list( total)

    title = f"Number of sections per document for the corpora {corpora_names}."
    create_histogram( total, max, histogram_image, title)

    print( title)
