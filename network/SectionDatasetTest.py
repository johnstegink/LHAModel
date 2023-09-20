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
        if not os.path.isfile( cache_file):
            self.__fill_cache( cache_file, Corpus(directory=corpus_dir),  relationsfile)

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


    def __fill_cache(self, file, corpus, set_size):
        """
        Fill the cache if it does not exist
        :param corpus: The corpus
        :param relationsfile: the xml file containing the relations between the documents
        """

        labels = []
        rows = []
        titles = []
        pairs = []

        for count in range(0, set_size):
            equal = (random.random() < 0.5)

            if equal:
                vector = [random.uniform(0.5, 1.0), random.uniform(0.8, 1.0), random.uniform(0.5, 0.9), random.uniform(0.5, 0.9)]
                title = "Equal"
            else:
                vector = [random.uniform(0.0, 0.5), random.uniform(0.2, 0.4), random.uniform(0.1, 0.5), random.uniform(0.1, 0.5)]
                title = "Not equal"

            rows.append( list(vector))
            titles.append( title)
            pairs.append( title)
            labels.append( 1.0 if equal else 0.0)

        self.__save_in_pickle(rows, labels, titles, pairs, file)



    def __read_from_cache(self, file):
        """
        Read data from the cache
        :param file:
        :return: (data, labels, titles, pairs)
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

