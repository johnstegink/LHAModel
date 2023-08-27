# Class implements a dataloader for corpus files. It uses a cache for the distance matrix
import pickle
import torch
import torch.nn.functional as torchF
import math

from texts.corpus import Corpus
from Distances.DocumentVectors import  DocumentVectors
from scipy.spatial import distance
import os
import numpy as np
from statistics import mean

class SectionDataset:

    def __init__(self, N, device, corpus_dir, dataset, documentvectors_dir, nntype, transformation, cache_file):
        """
        Set variables and read or create the graph
        :param N: the maximum number of sections i.e. the size of the vector will be NxN
        :param device: the device to put the tensors on
        :param corpus_dir: the directory with the corpus containing the document pairs
        :param dataset: can either be 'train' or 'validation'
        :param nntype: is a choice from the list ["plain", "masked", "lstm", "stat"]
        :param documentvectors_dir: the directory containing the documentvectors
        :param transformation: can either be 'truncate' or 'avg'
                               - truncate: elements having index > (N-1) are discarded
                               - last:     element (N-1) is the average of all elements having index > (N -1)
        :param cache_file: The cachefile containing a cached version of the similarity graph
        """



        self.N = N
        self.NNType = nntype
        self.device = device
        self.withmask = nntype == "masked"
        if transformation.lower() == 'truncate':
            self.transformation = self.__truncate_transformation
        elif transformation.lower() == 'avg':
            self.transformation = self.__average_transformation
        else:
            raise f"Unknown transformation: {transformation}, only 'truncate' or 'avg' are allowed"

        if not os.path.isfile( cache_file):
            self.__fill_cache( cache_file, Corpus(directory=corpus_dir),  DocumentVectors.read(documentvectors_dir))

        (all_data, all_labels, all_titles, all_pairs) = self.__read_from_cache( cache_file)

        validation_start = len( all_data) -int( len(all_data) / 10)
        if dataset.lower() == 'train':
            start_index = 0
            end_index = validation_start
        elif dataset.lower() == 'validation':
            start_index = validation_start + 1
            end_index = len(all_data) - 1
        else:
            raise f"Unknown dataset {dataset} only 'train' or 'validation' are allowed"

        self.data = []
        self.labels = []
        self.titles = []
        self.pairs = []
        for i in range( start_index, end_index + 1):
            self.data.append(torch.tensor( all_data[i], dtype=torch.float32, device=self.device))
            self.labels.append(torch.tensor( all_labels[i], dtype=torch.float32, device=self.device))
            self.titles.append( all_titles[i])
            self.pairs.append( all_pairs[i])

    def __fill_cache(self, file, corpus, document_vectors):
        """
        Fill the cache if it does not exist
        :param file The cache file to be created
        :param corpus The corpus
        :param file The documentvectors
        """

        labels = []
        rows = []
        titles = []
        pairs = []
        for pair in corpus.read_document_pairs():
            src = pair.get_src()
            dest = pair.get_dest()

            if document_vectors.documentvector_exists( src) and document_vectors.documentvector_exists( dest):
                srcname = corpus.getDocument(src).get_title()
                destname = corpus.getDocument(dest).get_title()

                titles.append(f"'{srcname} -> {destname}")
                pairs.append(f"{src} -- {dest}")

                src_vector = document_vectors.get_documentvector( pair.get_src())
                dest_vector = document_vectors.get_documentvector( pair.get_dest())

                # remove vectors that are all 0
                src_vectors = [vector for (index, vector) in src_vector.get_sections() if not all(v == 0 for v in vector)]
                dest_vectors = [vector for (index, vector) in dest_vector.get_sections() if not all(v == 0 for v in vector)]
                distances = distance.cdist(src_vectors, dest_vectors, 'cosine')
                sim_matrix = 1 - distances

                vector = self.__matrix_to_vector( sim_matrix)
                if self.withmask:
                    mask = np.zeros(vector.shape[0], dtype=float)
                    mask[vector != 0.0] = 1
                    vector += list(mask)
                    rows.append(list(vector) + list(mask))
                else:
                    rows.append( list( vector))


                labels.append( pair.get_similarity() )


        self.__save_in_pickle(rows, labels, titles, pairs, file)


    def __read_from_cache(self, file):
        """
        Read thy numpy arrays for data and labels from the file
        :param file:
        :return: (data, labels)
        """

        with open(file, "rb") as pickle_file:
            (data, labels, titles, pairs) = pickle.load(pickle_file)

        return data, labels, titles, pairs


    def __save_in_pickle(self, data, labels, titels, pairs, file):
        """
        Saves the data and the labels in a pickle file
        :param data:
        :param file:
        :return: None
        """

        with open(file , "wb") as pickle_file:
            pickle.dump( (data, labels, titels, pairs), pickle_file)


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


    def __len__(self):
        """
        Standard functions that returns the number of documentpairs
        :return:
        """

        return len(self.data)


    def __getitem__(self, idx):
        """
        Get the i-th item from the dataset
        :param idx:
        :return:
        """

        return( self.data[idx], self.labels[idx])

    def get_title(self, idx):
        """
        Get the (wikipedia)titels as a string
        :param idx:
        :return:
        """
        return self.titles[idx]


    def get_pair(self, idx):
        """
        Get the ids that form the pair as a string
        :param idx:
        :return:
        """
        return self.pairs[idx]


# N = 12
# test_ds = SectionDataset( N=N, corpus_dir="/Volumes/Extern/Studie/studie/corpora/wikisim_nl", documentvectors_dir="/Volumes/Extern/Studie/studie/vectors/wikisim_nl.xml", transformation="avg", cache_file=f"/Volumes/Extern/Studie/studie/scratch/wikisim_nl/dataset_{N}.pkl")
# for (data, label) in test_ds:
#     print( f"{len(data)} :  {label}   :  {data[0]}  :  {data[N -1]}")