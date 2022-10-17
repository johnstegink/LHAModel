import pickle

wikipedia_dumpdir = "../../Corpora/WikipediaDump"

import os
import re
from igraph import Graph



def read_sql_from_dump(dump_dir):
    """
    Read the page.sql and the pagelink.sql from the dumpdirectory
    :return: (pages.sql, pagelinks.sql
    """

    files = os.listdir(dump_dir)
    dump_re = re.compile(".*?wiki-(\d{8})-page.sql")
    dump_files = list(filter(dump_re.match, files))

    dump_files = list(filter(dump_re.match, files))
    if len(dump_files) > 0:
        pages_file = dump_files[0]
        links_file = pages_file.replace("-page.sql", "-pagelinks.sql")  # To make sure we use the same date
        if links_file in files:
            return (os.path.join(dump_dir, pages_file), os.path.join(dump_dir, links_file))

    print(f"Dump file does not contain the right files, use the page.sql file for this language and the accompanying pagelinks.sql file")



def replace_comma_in_string( str, replacement):
    """
    Replace the comma in a string and return the string without quotes
    :param str:
    :param replacement:
    :return:
    """
    alt = str.replace("\\'", "##quote##")
    parts = alt.split("'")
    for i in range(1, len(parts), 2):
        parts[i] = parts[i].replace(",", replacement)

    return "".join(parts).replace("##quote##", "'")


def read_values_from_sql( sqlfile):
    """
    Read all values in the VALUES part of an SQL file in a list
    :param sqlfile:
    :return:
    """

    with open( sqlfile) as file:
        for line in file:
            if line.startswith("INSERT INTO"):  # Only lines with inserts
                for part in line.replace(");\n", "").split("),("):
                    tobesplitted = part
                    if tobesplitted.startswith("INSERT INTO"):  # First part, start at the right part
                        tobesplitted = tobesplitted[tobesplitted.index("(") + 1:]

                    tobesplitted = replace_comma_in_string(tobesplitted, "###")
                    yield tobesplitted.split(",")



def read_all_page_ids( pages):
    """
    Use the SQL file to read all pageid and corresponding titles,
    read them from the cache if possible
    :param pages:
    :return:
    """

    chache_name = os.path.join( "cache", os.path.basename( pages) + ".cache")
    if  os.path.isfile( chache_name):
        with open( chache_name, "rb") as pickle_file:
            words = pickle.load( pickle_file)

    else:
        words = {}
        for record in read_values_from_sql( pages):
            if record[1] == "0":     # Main namespace
                words[record[0]] = record[2]

        with open( chache_name, "wb") as pickle_file:
            pickle.dump( words, pickle_file)

    return words


def read_all_page_links( pagelinks):
    """
    Use the SQL file to read all pagelinks and return a dictionary of dictonaries with links,
    read them from the cache if possible
    :param pages:
    :return:
    """

    chache_name = os.path.join( "cache", os.path.basename( pagelinks) + ".cache")
    if  os.path.isfile( chache_name):
        with open( chache_name, "rb") as pickle_file:
            linksdict = pickle.load( pickle_file)

    else:
        linksdict = {}
        counter = 0
        for record in read_values_from_sql(pagelinks):
            if len(record) > 3 and record[1] == "0" and record[3] == "0" and record[0] in words and record[2] in words:  # Main namespace

                src = words[record[0]].replace("_", " ")  # read the source
                dest = record[2].replace("_", " ")        # destination
                if not src in linksdict:                  # check if a src key exists
                    linksdict[src] = []
                linksdict[src].append( dest)                  # default weight of 0

            counter += 1
            if counter > 100000000:
                break

        with open( chache_name, "wb") as pickle_file:
            pickle.dump( linksdict, pickle_file)


    return linksdict


## Main part
if __name__ == '__main__':
    (pages, pagelinks) = read_sql_from_dump( wikipedia_dumpdir)
    counter = 0

    words = read_all_page_ids( pages)
    graphdict = read_all_page_links( pagelinks)
    graph = Graph.ListDict( graphdict)

    Graph.write_picklez( graph,os.path.join("cache", "graph.pkz"))
    a = Graph.distances(graph, v='Albert Speer', to='Anthony Fokker')


    print("klaar")

