# Script to interpret the results from trainModel.py it depends very much on the names of the files

import argparse
import os
import pandas
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import time
import functions
from sqlalchemy import create_engine
import psycopg2


def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Combine the results from trianModel.py')
    parser.add_argument('-d', '--results_dir', help='Directory containing the resultsfile', required=True)
    parser.add_argument('-t', '--type', help='The output type can be "excel", "sql" or "tsv"', choices=["excel", "tsv", "sql"], required=True)
    parser.add_argument('-o', '--output_file', help='The output file, in case of sql the connectionstring', required=True)
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
                new_row = {"corpus": file_data[1],
                           "language": file_data[2],
                           "method": file_data[3],
                           "similarity": int(file_data[4]),
                           "maxdoc": int(file_data[5]),
                           "nn": int(file_data[6]),
                           "n": int(file_data[7]),
                           "nntype": file_data[8],
                           "batchsize": int(content_data[0]),
                           "epochs": int(content_data[1]),
                           "learningrate": float(content_data[2]),
                           "f1": float(content_data[3]),
                           "accuracy": float(content_data[4]),
                           "precision": float(content_data[5]),
                           "recall": float(content_data[6]),
                           "truepositives": float(content_data[7]),
                           "falsepositives": float(content_data[8]),
                           "falsenegatives": float(content_data[9]),
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

    if type == "sql":
        table_name = "resultsTT"
        sqlEngine = create_engine(output_file, pool_recycle=3600)
        dbConnection = sqlEngine.connect()
        df = pandas.DataFrame(data)

        try:
            df.to_sql(table_name, sqlEngine, if_exists='replace')
        except ValueError as vx:
            print(vx)
        except Exception as ex:
            print(ex)
        else:
            print(f"Table {table_name} created successfully.");
        finally:
            dbConnection.close()

    elif type == "excel" or type == "tsv":
        corpora = df['corpus'].unique()
        languages = df['language'].unique()
        methods = df['method'].unique()
        NN_types = df["nn_type"].unique()
        for language in languages:
            for corpus in corpora:
                for method in methods:
                    for NN_type in NN_types:
                        selection = df.query(f'language == "{language}" & corpus=="{corpus}"  &  method=="{method}" & nntype=="{NN_type}" ')
                        selection.sort_values("f1", ascending=False)
                        [filename, ext] = os.path.splitext( output_file)
                        file_to_write = f"{filename}_{language}_{corpus}_{method}_{NN_type}.{ext}"
                        if type == "excel":
                            selection.to_excel(file_to_write)
                        elif type == "tsv":
                            selection.to_csv(file_to_write, sep="\t")
    else:
        print( f"Unknown type {type}")

