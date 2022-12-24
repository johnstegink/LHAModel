# Script to create embeddings from a document, similar to the LHA algorithm (Nikola I. Nikolov and Richard H.R. Hahnloser)

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

    parser = argparse.ArgumentParser(description='Evaluate the score.')
    parser.add_argument('-m', '--metricsdir', help='File with the metrics', required=True)
    parser.add_argument('-k', '--topk', help='The value of K (1-25) (as in topK) to use', choices=range(1,25), type=int, metavar="[1-25]" , required=True)
    parser.add_argument('-i', '--iteration', help='The sequence number of the iteration', type=int, required=True)
    parser.add_argument('-o', '--output', help='Directory where the image file should be saved', required=True)
    args = vars(parser.parse_args())

    return (args["metricsdir"], args["output"], args["topk"], args["iteration"])

def create_graph(df, topk, imagefile):
    graphcolors = {
        "main": "rgb(255, 126, 132)",
        "sub1":  "rgb(129, 0, 0)",
        "sub2":  "rgb(93, 138, 216)",
        "background": "rgb(240, 238, 238)"
    }

    colors=[graphcolors["main"], graphcolors["sub1"], graphcolors["sub2"]]

    plot = make_subplots()
    colorindex = 0
    fields = ["F1", "precision", "recall"]

    dftopk = df[(df.topk == topk)]
    for field in fields:
        layer   = go.Scatter(
                      x=dftopk["sim"],
                      y=dftopk[ field],
                      name=f"{field}",
                      marker=dict(
                          color=colors[colorindex]
                    )
                  )
        plot.add_trace( layer)
        colorindex += 1

    plot.update_layout(
        plot_bgcolor=graphcolors["background"],
        xaxis= {
        "title": "Value of θ"
        },
        yaxis={
            "title": "Score (between 0.0 and 1.0)"
        },
        yaxis_range=[0.0,1.0]

    )

    ## Workaround to remove error message from pdf
    workaround = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    workaround.write_image(imagefile, format="pdf")
    time.sleep(2)

    plot.write_image(imagefile)



# Main part of the script
if __name__ == '__main__':
    (metricsfile, outputdir, topk, iteration) = read_arguments()

    df = pandas.read_csv(metricsfile, delimiter="\t")
    name = os.path.splitext(os.path.basename(metricsfile))[0]
    outputfile = os.path.join( outputdir, f"iter_{iteration}_{name}_{topk}.pdf")
    print(outputfile)
    create_graph(df, topk, outputfile)
