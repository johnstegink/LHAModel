# Class that contains functionality for computing similarity matrix between vectors
import numpy as np
import pandas as pd
import sklearn
from scipy.spatial import distance

from Distances.DocumentRelations import DocumentRelations
from annoy import AnnoyIndex
from scipy.spatial import distance as spdistance


class SimilarityMatrix:
    def __init__(self, vectors1, vectors2):
        """
        Initialization
        :param vectors1: Dictionary<id, vector>
        :param vectors2: Dictionary<id, vector>
        """

        self.vectors = [vectors1, vectors2] # a list of  tuples( id, vectors)
        self.index_to_id = [] # a list of index to id translations which also is a list
        self.id_to_index = [] # a list of id to index translations which also is a list
        self.similarities = None # Contains the matrix, after the build

        self.__build()



    def __build(self):
        """
        Build the vectors list based the vectors and fill the index_to id and id_toindex
        :return:
        """

        np_vectors = []  # The array of the vectors of the two documents
        for vector_info in self.vectors:
            indexes = []
            ids = []
            vectors = []
            for (id, vector) in vector_info:
                indexes.append( len(ids))
                ids.append(id)
                vectors.append( vector)

            self.id_to_index.append( indexes)
            self.index_to_id.append( ids)
            np_vectors.append( np.array( vectors))

        distances = distance.cdist(np_vectors[0], np_vectors[1], 'cosine')
        self.similarities = 1 - distances  # Convert the distances to similarities


    def get_values(self):
        """
        Returns a dictionary with the src as a key and a list of tuples( dest, similarity) as value
        :return:
        """

        data = {}
        for axis in [0,1]:     # axis
            opposite_axis = (axis + 1) % 2

            for i in range(self.similarities.shape[axis]):
                srcid = self.index_to_id[axis][i]
                data[srcid] = []
                for j in range( self.similarities.shape[opposite_axis]):
                    dstid = self.index_to_id[opposite_axis][j]
                    similarity = self.similarities[i][j]
                    data[srcid].append( (dstid, similarity))

        return data