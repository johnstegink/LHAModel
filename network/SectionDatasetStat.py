# Class implements a dataset for corpus files with statistical information. It uses a cache
import pickle
import torch

from Distances.DocumentRelations import DocumentRelations
from texts.corpus import Corpus
from Distances.DocumentSectionRelations import DocumentSectionRelations
from scipy.spatial import distance
import os
import numpy as np
from statistics import mean

class SectionDatasetStat:

    def __init__(self, device, corpus_dir, dataset, relationsfile, top_threshold, cache_file):
        """
        :param device: the device to put the tensors on
        :param corpus_dir: the directory with the corpus containing the document pairs
        :param dataset: can either be 'train' or 'validation'
        :param top_threshold: The threshold to be considered the top of the document
        :param relationsfile: the xml file containing the relations between the documents
        :param cache_file: The cachefile containing a cached version of the similarity graph
        """

        self.device = device
        self.top_threshold = top_threshold
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


    def __fill_cache(self, file, corpus, relationsfile):
        """
        Fill the cache if it does not exist
        :param corpus: The corpus
        :param relationsfile: the xml file containing the relations between the documents
        """

        labels = []
        rows = []
        titles = []
        pairs = []

        dsr = DocumentSectionRelations.read( relationsfile)
        for pair in corpus.read_document_pairs():
            src = pair.get_src()
            dest = pair.get_dest()

            sections = dsr.get_section_relations( pair.get_src(), pair.get_dest())
            if len(sections) != 0:
                src_doc = corpus.getDocument(src)
                dest_doc = corpus.getDocument(dest)
                nrof_src_sections  = float(src_doc.get_nrof_sections())
                nrof_dest_sections = float(dest_doc.get_nrof_sections())
                total_nrof_src_sections = nrof_src_sections + nrof_dest_sections
                avg_count = float(len(sections)) / total_nrof_src_sections;
                avg_score = mean( [sect.get_similarity() for sect in sections])
                top_src_score = sum( [1.0 for sect in sections if self.is_top_section( sect.get_src_place(), nrof_src_sections )]  ) / total_nrof_src_sections
                top_dest_score = sum( [1.0 for sect in sections if self.is_top_section( sect.get_dest_place(), nrof_dest_sections )] ) / total_nrof_src_sections
                vector = [avg_count, avg_score, top_src_score, top_dest_score]
            else:
                vector = [0.0, 0.0, 0.0, 0.0]

            rows.append( list(vector))
            # titles.append( f"{src_doc.get_title()} --> {dest_doc.get_title()}");
            titles.append( f"{src} --> {dest} ")
            pairs.append( f"{src} --> {dest} ")
            labels.append(pair.get_similarity())

        self.__save_in_pickle(rows, labels, titles, pairs, file)


    def is_top_section(self, place, total_nrof_sections):
        """
        Returns true if the section belongs to the top of the document when the threshold is considered
        :param place:
        :param total_nrof_sections:
        :return:
        """

        # if not enough sections exists then the value true
        if int(1.0 / self.top_threshold) < int(total_nrof_sections):
            return True

        return (float(place) / float(total_nrof_sections)) < self.top_threshold



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

