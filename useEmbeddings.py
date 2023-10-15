# Use embeddings without any sections to determine the maximum score that can be obtained using embeddings

import argparse
import math
import os

import pandas
import scipy.spatial.distance
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sqlalchemy import create_engine
import psycopg2
from scipy.spatial.distance import cosine


import functions
from Distances.DocumentRelations import DocumentRelations
from Distances.DocumentSectionRelations import DocumentSectionRelations
from Distances.DocumentVectors import DocumentVectors
from network.SectionDataset import SectionDataset
from network.SectionDatasetStat import SectionDatasetStat
from network.SectionDatasetTest import SectionDatasetTest
from texts.corpus import Corpus


def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Create sematic links based on the embeddings only')
    parser.add_argument('-c', '--corpora_dir', help='The directory containing the corpora to calculate embeddings from', required=True)
    parser.add_argument('-t', '--type', help='The output type can be "excel", "sql" or "tsv"', choices=["excel", "tsv", "sql"], required=True)
    parser.add_argument('-v', '--vector_dir', help='The directory containing the vectors of the corpora', required=False)
    parser.add_argument('-o', '--results_dir', help='The directory containing the results, or postgress database link when type ="postgress"', required=True)

    args = vars(parser.parse_args())

    if not args["type"] == "sql":
        os.makedirs( args["results_dir"], exist_ok=True)

    return ( args["corpora_dir"], args["type"], args["vector_dir"], args["results_dir"])


def list_vectors(vector_dir):
    """
    List all vectors files split into information
    :param vector_dir:
    :return: a list of the basic corpus name, the language and the method
    """

    vectors = []
    for file in os.listdir( vector_dir):
        if ".xml" in file:
            parts = os.path.splitext(file)[0].split("_")
            if len(parts) == 3:
                [corpus, language, method] = parts
                vectors.append( (corpus, language, method))

    return vectors

def evaluate_the_model(  threshold, method, dv, corpusname, language, pairs):
    """
        Make an evaluation
    :param threshold:
    :param method: the method used to calculate the embeddings
    :param dv: document vectors
    :param corpusname:
    :param language:
    :param pairs:
    :return: dictionary that can be added to a dataframe
    """

    y_pred = []
    y_act = []
    missing = {}
    for pair in pairs:
        src = pair.get_src()
        dest = pair.get_dest()
        sim = pair.get_similarity()

        if not dv.documentvector_exists( src):
            missing[src] = 1
        elif not dv.documentvector_exists( dest):
            missing[dest] = 1
        else:
            pred_sim = 1 - scipy.spatial.distance.cosine( dv.get_documentvector( src).get_vector(), dv.get_documentvector( dest).get_vector())
            y_act.append( True if sim == 1.0 else False)
            y_pred.append( True if pred_sim >= (threshold / 100) else False)

    tp = fp = fn = 0
    for i in range(0, len(y_pred)):
        if y_pred[i] and y_act[i] : tp += 1
        elif y_pred[i] and not y_act[i]: fp += 1
        elif not y_pred[i] and y_act[i]: fn += 1

    (precision, recall, F1, _) =sklearn.metrics.precision_recall_fscore_support( y_act, y_pred, average="binary")
    accuracy = sklearn.metrics.accuracy_score( y_act, y_pred)
    new_row = {"corpus": corpusname,
               "language": language,
               "method": method,
               "f1": float(F1),
               "accuracy": float(accuracy),
               "precision": float(precision),
               "recall": float(recall),
               "truepositives": int(tp),
               "falsepositives": int(fp),
               "falsenegatives": int(fn),
               }

    print( f"{len(missing)} missing")
    return new_row


## Main part
if __name__ == '__main__':
    (corpora_dir, type, vector_dir, output_dir) = read_arguments()

    # Fill the dataframe
    data = []
    for (corpusname, language, method) in list_vectors(vector_dir):
        if language == "nl" : continue

        print("Reading corpus")
        corpus = Corpus(directory=os.path.join( corpora_dir, f"{corpusname}_{language}" ))

        print("Reading document pairs")
        pairs = corpus.read_document_pairs(True)

        print(f"Reading vectors: {corpusname} {language} {method}")
        dv = DocumentVectors.read( os.path.join( vector_dir, f"{corpusname}_{language}_{method}.xml"))

        for threshold in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            row = evaluate_the_model(
                threshold=threshold,
                corpusname = corpusname,
                language = language,
                method=method,
                dv = dv,
                pairs=pairs,
            )
            data.append( row)

    df = pandas.DataFrame( data)

    if type == "sql":
        table_name = "embeddings"
        sqlEngine = create_engine(output_dir, pool_recycle=3600)
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
        for language in languages:
            for corpus in corpora:
                for method in methods:
                    selection = df.query(f'language == "{language}" & corpus=="{corpus}"  &  method=="{method}" ')
                    selection.sort_values("f1", ascending=False)
                    if type == "excel":
                        file_to_write = f"{language}_{corpus}_{method}.xlsx"
                        selection.to_excel(os.path.join( output_dir, file_to_write))
                    elif type == "tsv":
                        file_to_write = f"{language}_{corpus}_{method}.tsv"
                        selection.to_csv(os.path.join( output_dir, file_to_write), sep="\t")
    else:
        print( f"Unknown type {type}")






