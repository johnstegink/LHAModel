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

    file_data = os.path.splitext( file)[0].split("_")
    first = True
    for line in contents.split( "\n"):
        if "\t" in line:
            if not first:
                content_data = line.split("\t")
                new_row = {"Corpus": file_data[1],
                           "Language": file_data[2],
                           "Method": file_data[3],
                           "Similarity": file_data[4],
                           "Max doc": file_data[5],
                           "N": file_data[6],
                           "NN": file_data[7],
                           "NN_type": file_data[8],
                           "Batch size": content_data[0],
                           "Epochs": content_data[1],
                           "Learning rate": content_data[2],
                           "F1": content_data[3],
                           "Accuracy": content_data[4],
                           "Precision": content_data[5],
                           "Recall": content_data[6],
                           "True positives": content_data[7],
                           "False positives": content_data[8],
                           "Latest loss": content_data[9]
                           }
                data.append(new_row)
            else:
                first = False


# Main part of the script
if __name__ == '__main__':
    (results_dir, type, output_file) = read_arguments()

    data = []
    for file in os.listdir( results_dir):
        file_name, file_extension = os.path.splitext( file)
        if file_extension == ".tsv":
            add_to_list(data, os.path.join(results_dir, file))


    df = pandas.DataFrame( data)
    corpora = df['Corpus'].unique()
    languages = df['Language'].unique()
    methods = df['Method'].unique()
    NN_types = df["NN_type"].unique()

    for language in languages:
        for corpus in corpora:
            for method in methods:
                for NN_type in NN_types:
                    selection = df.query(f'Language == "{language}" & Corpus=="{corpus}"  &  Method=="{method}" & NN_type=="{NN_type}" ')
                    selection.sort_values("F1", ascending=False)
                    [filename, ext] = os.path.splitext( output_file)
                    file_to_write = f"{filename}_{language}_{corpus}_{method}_{NN_type}.{ext}"
                    if type == "excel":
                        selection.to_excel(file_to_write)
                    elif type == "tsv":
                        selection.to_csv(file_to_write, sep="\t")
                    else:
                        print( f"Unknown type {type}")

