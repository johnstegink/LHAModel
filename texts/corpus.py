# Class to read a corpus in the Common Format and iterates through documents
import os.path
import random

import sklearn.model_selection

import functions
from Distances.DocumentRelations import DocumentRelations
from texts.document import Document
from texts.similarities import Similarities
from pathlib import Path

class Corpus:

    def __init__(self, directory):
        """
        Read the corpus
        :param directory:
        """

        self.directory = directory
        (name, language_code) = functions.read_corpus_info(directory)
        self.name = name
        self.language_code = language_code

        # Create a directory of filenames (without extensions) and their corresponding path
        self.files = {Path(file).stem:file for file in functions.read_all_files_from_directory( self.directory, "xml")}
        self.ids = list(self.files.keys())

        self.language = functions.translate_language_code(language_code)
        self.documentCache = {}

    def get_ids(self):
        """
        Returns a list of all ids of the corpus
        :return:
        """
        return self.ids

    def get_language(self):
        """
        Get the full name of the language
        :return:
        """
        return self.language

    def get_name(self):
        """
        Get the name of the corpus
        :return:
        """
        return self.name

    def get_language_code(self):
        """
        Get the language code of the corpus
        :return:
        """
        return self.language_code



    def get_number_of_documents(self):
        """
        Returns the number of documents
        :return:
        """
        return len(self.ids)


    def get_id_of_index(self, index):
        """
        Translate the document id to an index
        :param index:
        :return:
        """

        if index >= len(self.ids):
            raise(f"Index {index} too large, max: {len(self.ids)}")

        return self.ids[index]


    def get_index_of_id(self, id):
        """
        Translate the index to a docid
        :param id:
        :return:
        """

        return self.ids.index( id)


    def __iter__(self):
        """
        Initialize the iterator
        :return:
        """
        self.id_index = 0
        return self



    def __next__(self):
        """
        Next document from the corpus
        :return:
        """
        if self.id_index < len(self.ids):
            document = self.getDocument( self.get_id_of_index( self.id_index))

            self.id_index += 1  # Ready for the next document
            return document

        else:  # Done
            raise StopIteration


    def getDocument(self, id):
        """
        Read the document by id
        :param id: 
        :return: 
        """""

        if id in self.documentCache:
            return self.documentCache[id]
        else:
            if not id in self.files:
                raise Exception( f"Unknown id {id}")

            document =  Document(filename=self.files[id], language=self.language, index=self.get_index_of_id( id))
            self.documentCache[id] = document
            return document


    def has_document(self, id):
        """
        Checks whether the document with the given Id is in the corpus
        :param id:
        :return:
        """
        return id in self.files


    def read_similarities(self, single=False):
        """
        Return a new similarities object with all similarities in this corpus
        :param single: If true the links are made only one way
        :return:
        """

        sim = Similarities()
        for src in self.ids:
            document = self.getDocument(src)
            for (dest, similarity) in document.get_links():
                sim.add( src, dest, similarity)
                if not single:
                    sim.add( dest, src, similarity)  # Optional

        return sim


    def add_dissimilarities(self, sims):
        """
        Add a dissimelarity for every similarity
        :param sims: The similarities that the dissimilarities will be added to
        :return: None
        """

        all = sims.get_all_similarities()
        existsing = { f"{sim.get_src()}#{sim.get_dest()}" for sim in all }
        for sim in all:
            # find the next destination that is not linked
            dest = None
            src = sim.get_src()
            while dest is None:
                dest = random.choice( self.ids)
                if f"{src}#{dest}" in existsing:  # We had this already
                    dest = None

            if dest is None:
                raise f"Cannot find item that is not linked for {src}"

            else:
                existsing.add(f"{src}#{dest}")
                sims.add( src, dest, 0)


    def read_document_pairs(self, shuffled, training_set=True, test_set=True):
        """
        returns the document pairs of the corpus that have manually been annotated
        :param training_set: if true the training set will be included
        :param test_set: if true the test set will be included
        :return: Documentrelations object
        """

        dv = DocumentRelations([])
        self.create_training_test_pairs()
        pairs = []
        if training_set:
            pairs += functions.read_article_pairs( self.directory, "pairs_train.tsv")

        if test_set:
            pairs += functions.read_article_pairs( self.directory, "pairs_test.tsv")

        for pair in pairs:
            dv.add( pair[0], pair[1], pair[2])

        return dv


    def create_training_test_pairs(self):
        """
        Create a training set and a test set if the files are older
        than the pairs file
        :return:
        """

        pairs_file = os.path.join(self.directory, "pairs.tsv")
        train_file = os.path.join(self.directory, "pairs_train.tsv")
        test_file = os.path.join(self.directory, "pairs_test.tsv")

        # Create new files, only if neccesairy
        if not self.file_exists_or_is_newer( train_file, pairs_file) or not self.file_exists_or_is_newer( test_file, pairs_file):
            pairs = functions.read_article_pairs( self.directory)
            X = []
            Y = []
            for pair in pairs:
                X.append( (pair[0], pair[1]))
                Y.append( pair[2])

            X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split( X, Y, test_size=0.15, )
            self.write_pairs( X_train, Y_train, "pairs_train.tsv")
            self.write_pairs( X_test, Y_test, "pairs_test.tsv")


    def file_exists_or_is_newer(self, file, cmpfile):
        """
        Checks wether a file exists or is newer than the cmpfile
        :param file:
        :param cmpfile:
        :return:
        """

        if not os.path.isfile( file):
            return False

        return os.path.getmtime( file) > os.path.getmtime( cmpfile)

    def write_pairs(self, file, X, Y):
        """
        Write the pairs aftere splitting into to a file
        :param file:
        :param X:
        :param Y:
        :return:
        """

        lines = ""
        for i in range(len(X)):
            lines += f"{X[0]}\t{X[1]}\t{Y}\n"

        with open( file, "w") as f:
            f.write( lines)

