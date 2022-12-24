import argparse
import math
import os
import time

import pandas as pd

from Graph.WikiGraph import WikiGraph
import gc
import os
import functions
import pandas
import plotly.express as px
import plotly.graph_objects as go
import kaleido
from plotly.subplots import make_subplots

wikipedia_dumpdir = "../../Corpora/WikipediaDump"


def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Create a diagram from the gWikiMatch files, with the Milne_Witten score and distance')
    parser.add_argument('-i', '--input', help='Directory with the gWikiMatch files (only .tsv files are read)', required=True)
    parser.add_argument('-o', '--output', help='Directory with output images', required=True)
    parser.add_argument('-d', '--maxdegree', help='The maximum number of degrees per per vertex', required=True, type=int)

    args = vars(parser.parse_args())

    output = args["output"]
    os.makedirs( output, exist_ok=True)

    return ( args["input"], output,  args["maxdegree"])


def get_article( url):
    """
    Gets the article name from the url
    :param url:
    :return:

    """

    article = url[url.rindex( "/") + 1:]
    return article.replace("_", " ")


def read_file(file):
    """
    Read the file
    :param file:
    :return: a list of tuples in the form (from, to, isPositive)
    """
    info = []
    for line in functions.read_file(file).split("\n"):
        fields = line.split("\t")
        if len(fields) == 3:
            info.append((get_article(fields[1]), get_article(fields[2]), float(fields[0])))

    return info


def read_files(dir):
    """
    Read all files in the directory
    :param dir:
    :return: a dataframe
    """

    data = []
    for file in functions.read_all_files_from_directory(dir, "tsv"):
        data = data + read_file( file)

    return pandas.DataFrame(data, columns=["word1", "word2", "isPositive"])



def mw_label( mw):
    """
    Determines the label for the MW value of the row
    :param row:
    :return:
    """
    value = math.ceil( mw * 20)
    if value != 0:
        low = ((value - 1) / 20) + 0.01
        high = (value) / 20
        return f"{low:.2f} - {high:.2f}"
    else:
        return "0.00"


def visualize_MilneWitten( data, imagefile):
    """
    Creates a diagram of the MilneWitten values compared to the gWikiMatch data
    :param data:
    :return:
    """
    colors = ["rgb(116,0,48)","rgb(237,92,139)","rgb(38,139,210)","rgb(108,113,196)","rgb(211,54,130)","rgb(220,50,47)","rgb(203,75,22)","rgb(181,137,0)"]

    df = data.copy()
    df['Value'] = df.apply(lambda row: mw_label( row.MilneWitten), axis=1)

    labels = []
    yes = []
    no = []
    for i in range(0, 21):
        label = mw_label(i / 20)
        yes.append( 0)
        no.append( 0)
        labels.append( label)

    grp = df.groupby(['Value', 'isPositive']).count()
    for index, row in grp.iterrows():
        label = index[0]
        isPos = int( index[1])
        count = row[0]
        idx = labels.index(label)
        if isPos == 1:
            yes[idx] = count
        else:
            no[idx] = count

    plot = go.Figure(data=[go.Bar(
            name="Valid links",
            x = labels,
            y = yes,
            marker_color=colors[0]
    ),
        go.Bar(
            name="Invalid links",
            x = labels,
            y = no,
            marker_color = colors[1]
    )
    ])

    ## Workaround to remove error message from pdf
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig.write_image(imagefile, format="pdf")
    time.sleep(2)

    plot.update_layout(
        {"xaxis": {
            "title": "Range for the Milne&Witten score"
        },
        "yaxis": {
            "title": "Number of articles"
        }
    })

    plot.write_image(imagefile)


def distance_label( value):
    """
    Translated value for the label
    :param value:
    :return:
    """

    if value < 1 or value > 10:
        return "No path"
    else:
        return str(value)

def visualize_Distance( data, imagefile):
    """
    Creates a diagram of the distance values compared to the gWikiMatch data
    :param data:
    :return:
    """
    colors = ["rgb(116,0,48)","rgb(237,92,139)","rgb(38,139,210)","rgb(108,113,196)","rgb(211,54,130)","rgb(220,50,47)","rgb(203,75,22)","rgb(181,137,0)"]

    df = data.copy()

    df['Value'] = df.apply(lambda row: distance_label( row.distance), axis=1)
    labels = []
    yes = []
    no = []
    for i in range(-1, 11):
        label = distance_label(i)
        yes.append( 0)
        no.append( 0)
        labels.append( label)

    grp = df.groupby(['Value', 'isPositive']).count()
    for index, row in grp.iterrows():
        label = index[0]
        isPos = int( index[1])
        count = row[0]
        idx = labels.index(label)
        if isPos == 1:
            yes[idx] = count
        else:
            no[idx] = count

    plot = go.Figure(data=[go.Bar(
            name="Valid links",
            x = labels,
            y = yes,
            marker_color=colors[0]
        ),
        go.Bar(
            name="Invalid links",
            x = labels,
            y = no,
            marker_color = colors[1]
        )])

    plot.update_layout(
        {"xaxis": {
            "title": "Length of shortest path"
        },
        "yaxis": {
            "title": "Number of articles"
        }
    })


    ## Workaround to remove error message from pdf
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    fig.write_image(imagefile, format="pdf")
    time.sleep(2)

    plot.write_image(imagefile)





## Main part
if __name__ == '__main__':
    (input, outputdir, max_degree) = read_arguments()

    data = read_files(input)
    depths = [200]
    depths.extend(range(400, max_degree, 400))
    depths.append(max_degree)
    depths.reverse()

    for depth in depths:
        datafile = os.path.join( outputdir, f"milnewitten_{depth}.csv")
        if not os.path.exists( datafile):
            graph = WikiGraph(language="en", sql_dir=wikipedia_dumpdir, cache_dir="cache", max_degree=max_degree)
            graph.remove_nodes_degree_greater_than( depth )

            data['MilneWitten'] = data.apply(lambda row: graph.get_Milne_Witten(row.word1, row.word2), axis=1)
            graph.remove_cache_files()
            data.to_csv(datafile, sep='\t')
        else:
            data = pd.read_csv( datafile, sep="\t")
            graph = None

        visualize_MilneWitten(data, os.path.join( outputdir, f"milnewitten_{depth}.pdf"))
        if not graph is None:
            del graph


    # for depth in depths:
    #     datafile = os.path.join( outputdir, f"distance_{depth}.csv")
    #     if not os.path.exists( datafile):
    #         graph = WikiGraph(language="en", sql_dir=wikipedia_dumpdir, cache_dir="cache", max_degree=max_degree)
    #         graph.remove_nodes_degree_greater_than( depth )
    #
    #         data['distance'] = data.apply( lambda row: graph.get_distance( row.word1, row.word2), axis=1)
    #         graph.remove_cache_files()
    #         data.to_csv(datafile, sep='\t')
    #     else:
    #         data = pd.read_csv( datafile, sep="\t")
    #         graph = None
    #
    #     visualize_Distance( data, os.path.join( outputdir, f"distance_{depth}.pdf"))
    #
    #     if not graph is None:
    #         del graph

    print("ready")

