import argparse
from Graph.WikiGraph import WikiGraph
import gc

wikipedia_dumpdir = "../../Corpora/WikipediaDump"


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


## Main part
if __name__ == '__main__':
    (language, max_degree) = read_arguments()

    graph = WikiGraph(language=language, sql_dir=wikipedia_dumpdir, cache_dir="cache", max_degree=max_degree)

    # print( graph.get_distance( "Albert speer", "computer"))
    # print( graph.get_distance( "kyaniet", "geranium"))
    # print( graph.get_distance( "Jolien van Vliet", "Malachiet"))
    # print( graph.get_distance( "Piet Hein Donner", "Malachiet"))
    #
    # graph.save_cache_files()
    # del graph
    # gc.collect()
    #
    # graph = WikiGraph(language=language, sql_dir=wikipedia_dumpdir, cache_dir="cache", max_degree=max_degree)
    # print( graph.get_Milne_Witten( "Albert speer", "computer"))
    # print( graph.get_Milne_Witten( "kyaniet", "geranium"))
    # print( graph.get_Milne_Witten( "Jolien van Vliet", "Malachiet"))
    # print( graph.get_Milne_Witten( "Piet Hein Donner", "Malachiet"))
    # graph.save_cache_files()

    # create_DOT(graph, "Bonaire", 10, "c")
    print( graph.get_distance("Hippopotamus", "Frederick Griffith"))
    graph.remove_nodes_degree_greater_than( 200)


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

