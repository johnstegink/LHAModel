# Class containing a Pytorch wrapper around corpus documents

from torch.utils.data import Dataset
import torch
import torch.nn as nn

class CorpusDataset( Dataset):
    def __init__(self, preprocessor, similarities):
        """
        Creates a dataset for the preprocessor
        :param preprocessor:
        :param similarities: The similarities to be served by this dataset
        """
        self.preprocessor = preprocessor

        self.similarities = similarities

    def __getitem__(self, index):
        """
        Returns the item with the given index from the dataset
        :param index:
        :return: (src, dest, similarity)  where src and dest are numpy arrays
        """

        sim = self.similarities[index]

        # Retrieve the vectors for the source and destination
        src = self.preprocessor.create_or_load_embedding( sim.get_src())
        dest = self.preprocessor.create_or_load_embedding( sim.get_dest())

        return (  torch.from_numpy(src),
                  torch.from_numpy(dest),
                  torch.tensor( [[float(sim.get_similarity())]])
               )


    def __len__(self):
        """
        Returns the length of the dataset
        :return:
        """

        return len( self.similarities)

        