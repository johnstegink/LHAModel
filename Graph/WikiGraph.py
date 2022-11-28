# Class that creates and uses a graph of wikipedia links
import datetime
import math
import os
import pickle
import re
import networkit as nk
import array as arr

import functions


class WikiGraph:

    def __init__(self, language, sql_dir, max_degree, cache_dir):
        """
        Set variables and read or create the graph
        :param language: "en" or "nl"
        :param sql_dir: directory containing the files with the pages.sql and pagelinks.sql
        :param cache_dir:
        :param max_degree: The maximum degree a node can have, otherwise it will be ignored
        :param cache_dir: name of the directory with the cache files

        """
        self.language = language
        self.sql_dir = sql_dir
        self.cache_dir = cache_dir
        self.max_degree = max_degree
        self.nk_graph = None
        self.py_graph = None
        self.distance = None
        self.milne_witten = None
        self.distance_counter = 0
        self.removed = set()

        (self.python_cache, self.nk_graph_cache, self.py_graph_cache, self.distance_cache, self.milne_witten_cache) = self.__cache_files()
        if not os.path.isfile( self.python_cache) or not os.path.isfile(self.nk_graph_cache) or not os.path.isfile(self.py_graph_cache):
            self.__fill_cache(self.max_degree)

        self.words = self.__read_from_cache(self.python_cache)
        self.word_indexes = { word.lower():index for (index, word) in enumerate(self.words)}


    def __cache_files(self):
        """
        Returns a tuple with the names of the cache files
        :return: (python structures, graph)
        """

        return ( os.path.join(self.cache_dir, f"{self.language}_py.cache"),
                 os.path.join(self.cache_dir, f"{self.language}_nk_graph.cache"),
                 os.path.join(self.cache_dir, f"{self.language}_py_graph.cache"),
                 os.path.join(self.cache_dir, f"{self.language}_distance.cache"),
                 os.path.join(self.cache_dir, f"{self.language}_milnewitten.cache")
                 )


    def __fill_cache(self, max_degree):
        """
        Fill the cache
        :param max_degree: The maximum degree a node can have, otherwise it will be ignored
        :return: (words, graph) list of words and a graph
        """

        (pages_sql, pagelinks_sql) = self.__read_sql_from_dump(self.sql_dir, self.language)
        if os.path.isfile(self.python_cache + ".tmp"):
            with open(self.python_cache + ".tmp", "rb") as pickle_file:
                (words, pageids, degrees) = pickle.load( pickle_file )
            word_indexes = {word: index for (index, word) in enumerate(words)}
        else:
            (words, pageids) = self.__read_all_page_ids( pages_sql)
            word_indexes = {word: index for (index, word) in enumerate(words)}
            degrees = self.__read_degrees(pagelinks_sql, words, word_indexes, pageids)
            # Write to cache
            with open(self.python_cache + ".tmp", "wb") as pickle_file:
                pickle.dump((words, pageids, degrees), pickle_file)

        (nk_graph, py_graph) = self.__create_graph( pagelinks_sql, words, word_indexes, pageids, degrees, max_degree)

        print( "Saving graph...")
        nk.writeGraph(G=nk_graph, path=self.nk_graph_cache, fileformat=nk.Format.NetworkitBinary)
        self.__save_in_pickle(words, self.python_cache)
        self.__save_in_pickle(py_graph, self.py_graph_cache)
        self.__save_in_pickle({}, self.distance_cache)
        self.__save_in_pickle({}, self.milne_witten_cache)

        # Free some memory
        del word_indexes
        del words
        del pageids
        del nk_graph
        del py_graph


    def __save_in_pickle(self, data, file):
        """
        Saves the data in a pickle file
        :param data:
        :param file:
        :return: None
        """

        with open(file , "wb") as pickle_file:
            pickle.dump(data, pickle_file)

    def __replace_comma_in_string(self, str, replacement):
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



    def __read_values_from_sql(self, sqlfile):
        """
        Read all values in the VALUES part of an SQL file in a list
        :param sqlfile:
        :return:
        """

        with open(sqlfile) as file:
            for line in file:
                if line.startswith("INSERT INTO"):  # Only lines with inserts
                    for part in line.replace(");\n", "").split("),("):
                        tobesplitted = part
                        if tobesplitted.startswith("INSERT INTO"):  # First part, start at the right part
                            tobesplitted = tobesplitted[tobesplitted.index("(") + 1:]

                        tobesplitted = self.__replace_comma_in_string(tobesplitted, "###")
                        yield tobesplitted.split(",")



    def __clean_word(self, word):
        """
        Clean the word by replacing special characters
        :param word:
        :return:
        """
        return word.replace("_", " ")


    def __read_all_page_ids(self, pages_sql):
        """
        Use the SQL file to read all pageid and corresponding titles,
        read them from the cache if possible
        :param pages_sql:
        :return: list of words, and a dictionary containing the pageid as key and the word as value
        """

        page_ids = {}
        words = []

        for record in self.__read_values_from_sql(pages_sql):
            if record[1] == "0":  # Main namespace
                pageid = int(record[0])
                word = self.__clean_word( record[2])
                words.append( word)
                page_ids[pageid] = word


        return( words, page_ids)



    def __read_degrees(self, sql_file, words, word_indexes, pageids):
        """
        Use the SQL file to read all pagelinks and return a dictionary of dictonaries with links,
        :param sql_file: sql file
        :param words: list of words
        :param word_indexes: dictionary word -> index
        :param pageids: dictionary pageid -> word
        :return: a dictionary with the degrees
        """

        counter = 0
        degrees = {}
        for record in self.__read_values_from_sql(sql_file):
            # Check the record
            if len(record) > 3 and record[1] == "0" and record[3] == "0":
                src_id = int( record[0])
                dest_wrd = self.__clean_word( record[2])

                # Check for existence
                if src_id in pageids and dest_wrd in word_indexes:
                    src_wrd = pageids[src_id]
                    src_index = word_indexes[ src_wrd]
                    dest_index = word_indexes[dest_wrd]
                    if not src_index in degrees:
                        degrees[src_index] = 0
                    if not dest_index in degrees:
                        degrees[dest_index] = 0

                    degrees[src_index] += 1
                    degrees[dest_index] += 1

                if counter % 1000000 == 0:
                    now = datetime.datetime.now()
                    print( f"{now.hour:02}:{now.minute:02}:{now.second:02} Count {int( counter / 1000000)}")
                counter += 1

        return degrees



    def __create_graph(self, sql_file, words, word_indexes, pageids, degrees, max_degree):
        """
        Use the SQL file to read all pagelinks and return a dictionary of dictonaries with links
        creates a graph as a nk.Graph and as a list of sets were the set contains all destination indexes
        :param sql_file: sql file
        :param words: list of words
        :param word_indexes: dictionary word -> index
        :param pageids: dictionary pageid -> word
        :return:
        """


        nk_graph = nk.Graph(n=len(words),weighted=False,directed=False)
        py_graph = []
        py_graph = [arr.array("i", []) for i in range(0,len(word_indexes))]  # Create a list of sets with for every word, it contains the destinations
        counter = 0
        for record in self.__read_values_from_sql(sql_file):
            # Check the record
            if len(record) > 3 and record[1] == "0" and record[3] == "0":
                src_id = int( record[0])
                dest_wrd = self.__clean_word( record[2])

                # Check for existence
                if src_id in pageids and dest_wrd in word_indexes:
                    src_wrd = pageids[src_id]
                    src_index = word_indexes[ src_wrd]
                    dest_index = word_indexes[dest_wrd]

                    # Only add if there are not too many degrees
                    if degrees[src_index] <= max_degree  and degrees[dest_index] <= max_degree:
                        nk_graph.addEdge(src_index, dest_index, 1.0, False)

                    # Add the destination to the source graph
                    py_graph[src_index].append(dest_index)

                if counter % 1000000 == 0:
                    now = datetime.datetime.now()
                    print( f"{now.hour:02}:{now.minute:02}:{now.second:02} Graph {int( counter / 1000000)}")
                counter += 1

        return (nk_graph, py_graph)


    def __read_from_cache(self, file):
        """
        Read the datastructure from the cache
        :param file: the cache file to use
        :return: the data from the file, or None if the file does not exist
        """

        if os.path.isfile( file):
            if file == self.nk_graph_cache:
                return nk.readGraph(path=file, fileformat=nk.Format.NetworkitBinary)
            else:
                with open(file, "rb") as pickle_file:
                    return pickle.load( pickle_file)
        else:
            return None


    def __read_sql_from_dump(self, dump_dir, language):
        """
        Read the page.sql and the pagelink.sql from the dumpdirectory
        :param dump_dir: directory with the sql dumps
        :param language: language code
        :return: (pages.sql, pagelinks.sql)
        """

        files = os.listdir(dump_dir)
        dump_re = re.compile(f"{language}wiki-(\d{{8}})-page.sql")
        dump_files = list(filter(dump_re.match, files))

        dump_files = list(filter(dump_re.match, files))
        if len(dump_files) > 0:
            pages_file = dump_files[0]
            links_file = pages_file.replac  # To make sure we use the same date
            if links_file in files:
                return (os.path.join(dump_dir, pages_file), os.path.join(dump_dir, links_file))

        print(f"Dump file does not contain the right files, use the page.sql file for this language and the accompanying pagelinks.sql file")


    def __get_graphid_of_word(self, word):
        """
        Returns the graph id of a word
        :param word:
        :return: the graph id, -1 if not found
        """

        wrd = word.lower()
        if wrd in self.word_indexes:
            return self.word_indexes[wrd]
        else:
            return -1


    def get_distance(self, word1, word2):
        """
        Determine the distance in clicks from
        :param word1:
        :param word2:
        :return: the distance
        """

        if self.distance is None:
            self.distance = self.__read_from_cache(self.distance_cache)
            if self.distance is None:
                self.distance = {}

        # Try to read it from the cache
        cache_key = f"{word1}#{word2}"
        if( cache_key in self.distance):
            return self.distance[cache_key]

        if self.distance_counter is None:
            self.distance_counter = 0

        src = self.__get_graphid_of_word( word1)
        target = self.__get_graphid_of_word( word2)

        if self.nk_graph is None:
            self.nk_graph = self.__read_from_cache(self.nk_graph_cache)

        if src >= 0 and target >= 0 and src not in self.removed and target not in self.removed:
            biBFS = nk.distance.BidirectionalBFS(G=self.nk_graph, source=src, target=target)
            biBFS.run()
            print( [self.words[i] for i in biBFS.getPath()])
            dist = int( biBFS.getDistance())
            if dist > 1000:
                dist = -1
        else:
            dist = -2

        # Save the data for the next time
        self.distance[cache_key] = dist

        # Save the cache every now and then
        self.distance_counter += 1
        if self.distance_counter % 10 == 0:
            self.save_cache_files()

        print(f"{self.distance_counter}: {word1} -> {word2} [{dist}]");

        return dist

    def get_Milne_Witten(self, word1, word2):
        """
        Retrieve teh Milne&Witten score, is much faster than te distance
        :param word1:
        :param word2:
        :return: score
        """

        if self.milne_witten is None:
            self.milne_witten = self.__read_from_cache(self.milne_witten_cache)
            if self.milne_witten is None:
                self.milne_witten = {}

        # Try to read it from the cache
        cache_key = f"{word1}#{word2}"
        if( cache_key in self.milne_witten):
            return self.milne_witten[cache_key]

        src = self.__get_graphid_of_word( word1)
        target = self.__get_graphid_of_word( word2)

        if self.py_graph is None:
            self.py_graph = self.__read_from_cache( self.py_graph_cache)

        if src >= 0  and target >= 0:
            A = self.py_graph[src]
            B = self.py_graph[target]
            maximum = max( len(A), len(B))
            minimum = min( len(A), len(B))
            intersect = len( set(A).intersection( B))
            wikilen = len( self.py_graph)

            if intersect != 0:
                mw = 1 - (math.log(maximum) - math.log( intersect)) / (math.log(wikilen) - math.log(minimum))
            else:
                mw = 0

        else:
            mw = 0

        # Save the data for the next time
        self.milne_witten[cache_key] = mw

        return mw


    def save_cache_files(self):
        """
        Save the cache files for milne_witten and the distance
        :return: None
        """

        if not self.milne_witten is None:
            self.__save_in_pickle(self.milne_witten, self.milne_witten_cache)

        if not self.distance is None:
            self.__save_in_pickle(self.distance, self.distance_cache)

    def remove_cache_files(self):
        """
        Remove the cache files for milne_witten and the distance
        :return: None
        """

        if os.path.isfile( self.milne_witten_cache):
            os.remove( self.milne_witten_cache)
        if os.path.isfile( self.distance_cache):
            os.remove( self.distance_cache)


    def neighbors(self, word):
        """
        Returns a list of all neighbours of a node
        :param word:
        :return:
        """
        node = self.__get_graphid_of_word( word)
        if self.nk_graph is None:
            self.nk_graph = self.__read_from_cache(self.nk_graph_cache)

        neighbors = []
        for nb in self.nk_graph.iterNeighbors(node):
            neighbors.append( self.words[nb])

        return neighbors



    def remove_nodes_degree_greater_than(self, max_degree):
        """
        Remove all nodes with a degree greater than the value
        :param max_degree: maximum degree
        :return:
        """
        if self.nk_graph is None:
            self.nk_graph = self.__read_from_cache(self.nk_graph_cache)

        nodes = [node for node in self.nk_graph.iterNodes() if self.nk_graph.degree( node) >= max_degree]
        for node in nodes:
            self.nk_graph.removeNode( node)
            self.removed.add( node)
