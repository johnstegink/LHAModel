# Class for the word2vec encoder
import os

import numpy as np
import sent2vec

import functions
from texts.clean import Cleaner
from gensim.models.keyedvectors import KeyedVectors

from documentencoders.documentencoder import Documentencoder_base


class AvgWord2VecEncoder(Documentencoder_base):

    WORD2VECMODELPATH_EN = "Pretrained/Word2Vec/GoogleNews-vectors-negative300.bin"
    WORD2VECMODELPATH_NL = "Pretrained/Word2Vec/nl-combined-320.txt"
    WORD2VECVECTORSIZE = 300

    def __init__(self, language_code):
        super(AvgWord2VecEncoder, self).__init__(language_code)
        modelpath = os.path.join(os.getcwd(), AvgWord2VecEncoder.WORD2VECMODELPATH_NL if language_code == "nl" else AvgWord2VecEncoder.WORD2VECMODELPATH_EN)
        self.word2vec = KeyedVectors.load_word2vec_format(modelpath, binary=True)
        self.cleaner = Cleaner(language_code=self.language_code)

    def get_vector_size(self):
        """
        The resulting vector size
        :return:
        """
        return AvgWord2VecEncoder.WORD2VECVECTORSIZE      # This is because the Pretrained model has a fixed vector size


    def embed_text(self, text):
        """
        Create sentence to vec embedding
        :param text: can either be a string or a list of strings (sentences)
        :return:
        """

        joined = " ".join( text) if type(text) == list else text
        clean = self.cleaner.clean_text(txt=joined, remove_stop=True, remove_digits=True, lower=True)
        vectors = self.__create_vectors(clean)

        return np.mean( vectors, 0).tolist()


    def __create_vectors(self, text):
        """
        Create a list of word vectors
        :param text:
        :return:
        """
        vectors = []
        for word in text:
            if self.word2vec.has_index_for( word):
                word_embedding = self.word2vec.get_vector(word)
                vectors.append( word_embedding)

        return vectors
