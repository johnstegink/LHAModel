# Class implements a dataset for corpus files. It uses a cache for the distance matrix
import pickle

import numpy
import torch
import torch.utils.data
import torch.nn.functional as torchF
import math

from texts.corpus import Corpus
from Distances.DocumentVectors import  DocumentVectors
from scipy.spatial import distance
import os
import numpy as np
from statistics import mean

class SectionDataset(torch.utils.data.IterableDataset):

    def __init__(self, N, device, corpus_dir, documentvectors_dir, nntype, transformation, cache_file):
        """
        :param N: the maximum number of sections i.e. the size of the vector will be NxN
        :param device: the device to put the tensors on
        :param corpus_dir: the directory with the corpus containing the document pairs
        :param nntype: is a choice from the list ["plain", "masked", "lstm"]
        :param documentvectors_dir: the directory containing the documentvectors
        :param transformation: can either be 'truncate' or 'avg'
                               - truncate: elements having index > (N-1) are discarded
                               - last:     element (N-1) is the average of all elements having index > (N -1)
        :param cache_file: The cachefile containing a cached version of the similarity graph
        """

        super(SectionDataset).__init__()


        self.N = N
        if not nntype in ["plain", "masked"]:
            raise f"Unknown nntype: {transformation}, only 'plain' or 'masked' are allowed"

        self.NNType = nntype
        self.device = device
        self.withmask = nntype == "masked"
        if transformation.lower() == 'truncate':
            self.transformation = self.__truncate_transformation
        elif transformation.lower() == 'avg':
            self.transformation = self.__average_transformation
        else:
            raise f"Unknown transformation: {transformation}, only 'truncate' or 'avg' are allowed"



        # if not os.path.isfile( cache_file):
        #     self.__fill_cache( cache_file, Corpus(directory=corpus_dir),  DocumentVectors.read(documentvectors_dir))

        self.__fill_cache(cache_file, Corpus(directory=corpus_dir), DocumentVectors.read(documentvectors_dir))

        self.data = self.__read_from_cache( cache_file)

    def __fill_cache(self, file, corpus, document_vectors):
        """
        Fill the cache if it does not exist
        :param file The cache file to be created
        :param corpus The corpus
        :param file The documentvectors
        """

        rows = []
        for pair in corpus.read_document_pairs( True):
            src = pair.get_src()
            dest = pair.get_dest()

            # if not ( src == "Q166620" and  dest == "Q1572329"): continue

            if document_vectors.documentvector_exists( src) and document_vectors.documentvector_exists( dest):
                srcname = corpus.getDocument(src).get_title()
                destname = corpus.getDocument(dest).get_title()

                src_vector = document_vectors.get_documentvector( pair.get_src())
                dest_vector = document_vectors.get_documentvector( pair.get_dest())

                src_vectors = [vector for (index, vector) in src_vector.get_sections()] # if not all(v == 0 for v in vector)]
                dest_vectors = [vector for (index, vector) in dest_vector.get_sections()] # if not all(v == 0 for v in vector)]

                # everything zero is not equal
                distances = distance.cdist(src_vectors, dest_vectors, 'cosine')
                sim_matrix = 1 - distances

                vector = self.__matrix_to_vector( sim_matrix)

                # everything zero is not equal
                if all(v == 0 for v in vector):
                    sim_matrix = numpy.zeros( src_vectors.shape[0], dest_vectors.shape[0])

                nans = [True for val in vector if math.isnan(val)]
                if len( nans) == 0:
                    if self.withmask:
                        mask = np.zeros(vector.shape[0], dtype=float)
                        mask[vector != 0.0] = 1
                        vector += list(mask)
                        rows.append((list(vector) + list(mask), pair.get_similarity(), f"{srcname} -> {destname}", f"{src} -- {dest}"))
                    else:
                        rows.append((list(vector), pair.get_similarity(), f"{srcname} -> {destname}", f"{src} -- {dest}"))



        self.__save_in_pickle(file,rows)


    def __read_from_cache(self, file):
        """
        Read thy numpy arrays for data and labels from the file
        :param file:
        :return: (data, labels)
        """

        with open(file, "rb") as pickle_file:
            return pickle.load(pickle_file)


    def __save_in_pickle(self, file, data):
        """
        Saves the data and the labels in a pickle file
        :param data:
        :param file:
        :return: None
        """

        with open(file, "wb") as pickle_file:
            pickle.dump(data, pickle_file)

    def __iter__(self):
        return iter(self.data)

    def __matrix_to_vector(self, sim_matrix):
        """
        Converts the similarty matrix into a vector of size (matrix_rows) * (N)
        :param sim_matrix:
        :return:
        """
        matrix = np.zeros((self.N, self.N), float)
        for row in range(0, min( sim_matrix.shape[0], self.N)):
            for column in range(0, min(sim_matrix.shape[1], self.N)):
                matrix[row,column] = sim_matrix[row,column]

        # Flatten into a vector
        return matrix.flatten()

    def __truncate_transformation(self, matrix_row):
        """
        Returns a vector of the matrix_row, that is larger than N
        :param matrix_row:
        :return:
        """

        return matrix_row[0:(self.N - 1)]

    def __average_transformation(self, matrix_row):
        """
        Returns a vector of the matrix_row, that is larger than N
        :param matrix_row:
        :return:
        """

        average = mean( matrix_row[(self.N - 1):])
        return matrix_row[0:(self.N - 1)] + [average]


# N = 12
# test_ds = SectionDataset( N=N, corpus_dir="/Volumes/Extern/Studie/studie/corpora/wikisim_nl", documentvectors_dir="/Volumes/Extern/Studie/studie/vectors/wikisim_nl.xml", transformation="avg", cache_file=f"/Volumes/Extern/Studie/studie/scratch/wikisim_nl/dataset_{N}.pkl")
# for (data, label) in test_ds:
#     print( f"{len(data)} :  {label}   :  {data[0]}  :  {data[N -1]}")