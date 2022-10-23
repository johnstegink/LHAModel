import argparse
import pickle

wikipedia_dumpdir = "../../Corpora/WikipediaDump"

import os
import re
from Graph.WikiGraph import WikiGraph



def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Create a graph from the wikipedia dumps with all links as vertices')
    parser.add_argument('-l', '--language', help='Wiki language', required=True, default="en", choices=["nl", "en"])
    parser.add_argument('-d', '--maxdegree', help='The maximum number of degrees per per vertex', required=True, type=int)
    args = vars(parser.parse_args())

    return ( args["language"], args["maxdegree"])



## Main part
if __name__ == '__main__':
    (language, max_degree) = read_arguments()

    graph = WikiGraph(language=language, sql_dir=wikipedia_dumpdir, cache_dir="cache", max_degree=max_degree)
    print( graph.get_distance( "Albert speer", "computer"))
    print( graph.get_distance( "kyaniet", "geranium"))
    print( graph.get_distance( "Jolien van Vliet", "Malachiet"))
    print( graph.get_distance( "Piet Hein Donner", "Malachiet"))


    # (pages, pagelinks) = read_sql_from_dump( wikipedia_dumpdir, language)
    # counter = 0
    #
    # print("words...")
    # (words, revs) = read_all_page_ids( pages)
    # print("graphdict...")
    # graphdict = read_all_page_links( pagelinks, words, revs)
    # print("create files...")
    # create_cypherls( revs, graphdict, language, output)

    print("ready")

