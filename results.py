# Script to interpret the results from trainModel.py it depends very much on the names of the files

import argparse
import os
import pandas
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import time
import functions


def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Combine the results from trianModel.py')
    parser.add_argument('-d', '--results_dir', help='Directory containing the resultsfile', required=True)
    parser.add_argument('-t', '--type', help='The output type can be "excel" or "tsv"', choices=["excel", "tsv"], required=True)
    parser.add_argument('-o', '--output_file', help='The output file', required=True)
    args = vars(parser.parse_args())

    return (args["results_dir"], args["type"], args["output_file"])


def add_to_list(data, path):
    """
    Adds the current file to the list
    :param data: The list of dictionaries
    :param path: The file to be added
    :return:
    """

    file = os.path.basename( path)
    file_name, file_extension = os.path.splitext(file)
    contents = functions.read_file( path)

    file_data = file.split("_")
    content_data = contents.split("\t")
    new_row = {"Corpus": file_data[0],
               "Language": file_data[1],
               "Method": file_data[2],
               "Similarity": file_data[3],
               "Max doc": file_data[4],
               "N": file_data[5],
               "NN type": file_data[6],
               "F1": content_data[0],
               "Accuracy": content_data[1],
               "Precision": content_data[2],
               "Recall": content_data[3],
               "True positives": content_data[4],
               "False positives": content_data[5],
               "True negatives": content_data[6],
               "Latest loss": content_data[7]
               }
    data.append(new_row)



# Main part of the script
if __name__ == '__main__':
    (results_dir, type, output_file) = read_arguments()

    data = []
    for file in os.listdir( results_dir):
        file_name, file_extension = os.path.splitext( file)
        if file_extension == ".tsv":
            add_to_list(data, os.path.join(results_dir, file))

    df = pandas.DataFrame( data)
    if type == "excel":
        df.to_excel(output_file)
    elif type == "tsv":
        df.to_csv(output_file, sep="\t")
    else:
        print( f"Unknown type {type}")
