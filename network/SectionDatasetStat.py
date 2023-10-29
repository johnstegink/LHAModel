# Class implements a dataset for corpus files with statistical information. It uses a cache
import math
import pickle
import torch
import random
import torch.utils.data
from tqdm import tqdm

from Distances.DocumentRelations import DocumentRelations
from texts.corpus import Corpus
from Distances.DocumentSectionRelations import DocumentSectionRelations
from scipy.spatial import distance
import os
import numpy as np
from statistics import mean

class SectionDatasetStat(torch.utils.data.IterableDataset):

    def __init__(self, device, corpus_dir, relations_file, top_threshold, cache_file):
        """
        :param device: the device to put the tensors on
        :param corpus_dir: the directory with the corpus containing the document pairs
        :param top_threshold: The threshold to be considered the top of the document
        :param relations_file: the xml file containing the relations between the documents
        :param cache_file: The cachefile containing a cached version of the similarity graph
        """

        super(SectionDatasetStat).__init__()

        self.device = device
        self.top_threshold = top_threshold
        if not os.path.isfile( cache_file):
            self.__fill_cache(cache_file, Corpus(directory=corpus_dir), relations_file)

        self.data = self.__read_from_cache( cache_file)



    def __fill_cache(self, file, corpus, relationsfile):
        """
        Fill the cache if it does not exist
        :param corpus: The corpus
        :param relationsfile: the xml file containing the relations between the documents
        """

        rows = []

        print("Reading document relations")
        dsr = DocumentSectionRelations.read( relationsfile)
        print("Reading document pairs")
        pairs = corpus.read_document_pairs( True)
        with tqdm(total=len( pairs), desc="Creating stat vectors") as progress:
            for pair in pairs:
                src = pair.get_src()
                dest = pair.get_dest()

                # Search for source or destination
                sections = dsr.get_section_relations( pair.get_src(), pair.get_dest())
                if len(sections) != 0:
                    sections = dsr.get_section_relations(pair.get_dest(), pair.get_src())

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

                # Filter the nans
                nans = [True for val in vector if math.isnan(val)]
                if len( nans) == 0:
                    rows.append( (list(vector), pair.get_similarity(), f"{src} --> {dest} ", f"{src} --> {dest} "))
                progress.update()

        random.shuffle( rows)
        self.__save_in_pickle(rows, file)


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
        return iter( self.data)

