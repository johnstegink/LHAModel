# Class to read a corpus in the Common Format and iterates through documents
import random

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


    def read_document_pairs(self, shuffled):
        """
        returns the document pairs of the corpus that have manually been annotated
        :param shuffled: If true the set will be shuffled
        :return: Documentrelations object
        """

        dv = DocumentRelations([])
        pairs = functions.read_article_pairs( self.directory)
        if shuffled: random.shuffle( pairs)

        for pair in pairs:
            dv.add( pair[0], pair[1], pair[2])

        return dv