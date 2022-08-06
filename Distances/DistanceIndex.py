# Class that contains functionality for computing the distance

from documentencoders.Sent2VecEncoder import Sent2VecEncoder
from annoy import AnnoyIndex
import functions
import os
from tqdm import *

NUMBEROFTREES = 50
ANNOYFILE = "index.ann"
VECTORSIZE = 600

class DistanceIndex:
    def __init__(self, corpus, language, vector_size, outputdir):
        self.corpus = corpus
        self.outputdir = outputdir
        self.vector_size = vector_size
        self.language = language
        self.index = self.__load(self.index_file)


    def __createindex(self):
        return AnnoyIndex(VECTORSIZE, "angular")

    def __build(self, index_file):
        """
        Builds the index file
        :param index_file:
        :return: the index
        """
        functions.show_message("Loading encoder")
        encoder = Sent2VecEncoder(self.language, self.vector_size)
        functions.show_message("Create index")
        index = self.__createindex()
        with tqdm(total=self.corpus.get_number_of_documents(), desc="Total progress") as progress:
            for document in self.corpus:
                vector = encoder.embed_text(document.get_fulltext())
                index.add_item(document.get_index(), vector)
                progress.update()

        functions.show_message("Save index")
        del encoder
        index.build(NUMBEROFTREES)
        index.save( self.index_file)
        functions.show_message("Index saved")

        return index


    def __load(self, index_file):
        """
        Tries to load the index file from disk, if the file is not there it will be created
        :param index_file:
        :return: the annoy index
        """

        if not os.path.isfile( self.index_file):
            return self.__build(index_file)
        else:
            functions.show_message("Loading index")
            index = self.__createindex()
            index.load( index_file)
            functions.show_message("Index loaded")
            return index


    def __find_links(self, documentid):
        document = self.corpus.get_document( documentid)