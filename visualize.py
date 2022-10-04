# Script to create embeddings from a document, similar to the LHA algorithm (Nikola I. Nikolov and Richard H.R. Hahnloser)

import argparse
import os
import pandas
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import functions


def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Evaluate the score.')
    parser.add_argument('-m', '--metricsdir', help='Directory containing the metric files', required=True)
    args = vars(parser.parse_args())

    return (args["metricsdir"])


def read_data( mdir):
    """
    Read all files in a dictionary of data with the name as a key
    The rows are dictionary objects
    :param mdir:
    :return: data
    """
    data = {}
    for file in functions.read_all_files_from_directory(mdir, "tsv"):
        df = pandas.read_csv(file, delimiter="\t")
        name = os.path.splitext(os.path.basename(file))[0]
        data[name] = df

    return data


def create_graph( data):
    colors = ["rgb(133,153,0)","rgb(42,161,152)","rgb(38,139,210)","rgb(108,113,196)","rgb(211,54,130)","rgb(220,50,47)","rgb(203,75,22)","rgb(181,137,0)"]

    for dataset in data.keys():
        df = data[dataset]
        fig = make_subplots()
        colorindex =0
        for topk in range(3, 20, 3):
            dftopk = df[(df.topk == topk)]
            layer   = go.Scatter(
                          x=dftopk["sim"],
                          y=dftopk["F1"],
                          name=f"Top {topk}",
                          marker=dict(
                              color=colors[colorindex]
                        )
                      )
            fig.add_trace( layer)
            colorindex += 1

        fig.update_layout(title=dict(
           text = dataset
        ))

        fig.show()


# Main part of the script
if __name__ == '__main__':
    (mdir) = read_arguments()

    data = read_data(mdir)
    create_graph(data)
