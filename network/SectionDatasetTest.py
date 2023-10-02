# Class implements a dataset for corpus files with test information. It uses a cache so iterations can be compared
import pickle
import random

import torch
import torch.utils.data

from texts.corpus import Corpus
from Distances.DocumentSectionRelations import DocumentSectionRelations
import os
from statistics import mean

class SectionDatasetTest(torch.utils.data.IterableDataset):

    def __init__(self, device, dataset, set_size, cache_file):
        """
        :param device: the device to put the tensors on
        :param dataset: can either be 'train' or 'validation'
        :param size of the set
        :param cache_file: The cachefile containing a cached version of the similarity graph
        """

        super(SectionDatasetTest).__init__()

        self.device = device
        # if not os.path.isfile( cache_file):
        #     self.__fill_cache( cache_file,  set_size)

        self.__fill_cache( cache_file,  set_size)

        all_data = self.__read_from_cache( cache_file)

        validation_start = len( all_data) - int( len(all_data) / 10)
        if dataset.lower() == 'train':
            self.data = all_data[0:validation_start - 1]
        elif dataset.lower() == 'validation':
            self.data = all_data[validation_start:]
        else:
            raise f"Unknown dataset {dataset} only 'train' or 'validation' are allowed"


    def __fill_cache(self, file, set_size):
        """
        Fill the cache if it does not exist
        :param file: The file
        :param set_size: The size of the set to be generated
        """

        rows = []
        for count in range(0, set_size):
            equal = (random.random() < 0.5)

            if equal:
                vector = [random.uniform(0.7, 1.0), random.uniform(0.8, 1.0), random.uniform(0.7, 0.9), random.uniform(0.7, 0.9)]
                title = "Equal"
            else:
                vector = [random.uniform(0.0, 0.3), random.uniform(0.2, 0.4), random.uniform(0.1, 0.3), random.uniform(0.1, 0.3)]
                title = "Not equal"

            # Add a tuple X, Y, title, pair
            rows.append( (list(vector), 1.0 if equal else 0.0, title, title))

        data = ([], [], [], [])
        for row in rows:
            data[0].append( row[0])
            data[1].append(row[1])
            data[2].append(row[2])
            data[3].append(row[3])

        self.__save_in_pickle(rows, file)



    def __read_from_cache(self, file):
        """
        Read data from the cache
        :param file:
        :return: (data, labels, titles, pairs)
        """

        with open(file, "rb") as pickle_file:
            return pickle.load(pickle_file)


    def __save_in_pickle(self, data, file):
        """
        Saves the data and the labels in a pickle file
        :param data:
        :param file:
        :return: None
        """

        with open(file , "wb") as pickle_file:
            pickle.dump( data, pickle_file)


    def __iter__(self):
        ...
        return iter(range(0, len(self.data)))

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

        return( self.data[idx])

    def get_title(self, idx):
        """
        Get the (wikipedia)titels as a string
        :param idx:
        :return:
        """
        return self.data[idx][2]


    def get_pair(self, idx):
        """
        Get the ids that form the pair as a string
        :param idx:
        :return:
        """
        return self.data[idx][3]

