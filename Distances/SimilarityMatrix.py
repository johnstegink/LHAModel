# Class that contains functionality for computing similarity matrix between vectors
import numpy as np
import pandas as pd
import sklearn
from scipy.spatial import distance

from Distances.DocumentRelations import DocumentRelations
from annoy import AnnoyIndex
from scipy.spatial import distance as spdistance


class SimilarityMatrix:
    def __init__(self, documentvectors):
        """
        Initialization
        :param documentvectors: dictionary with a description of the vector as a key and the vector as the value
        """
        self.documentvectors = documentvectors
        self.vectors = None
        self.index_to_id = {}
        self.id_to_index = {}
        self.__build()

    def __build(self):
        """
        Build the vectors list based the documentvectors and fill the index_to id and id_toindex
        :return:
        """

        i = 0
        vectors = []
        for (id, vector) in self.documentvectors:
            self.index_to_id[i] = id
            self.id_to_index[id] = i
            vectors.append( vector)
            i += 1
        self.vectors = np.array( vectors)

        distances = distance.cdist(vectors, vectors)
        similarities = 1 - distances  # Convert the distances to similarities
        return similarities

