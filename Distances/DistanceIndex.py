# Class that contains functionality for computing the distance
import numpy as np
import sklearn
from tqdm import *

from Distances.DocumentRelations import DocumentRelations
from annoy import AnnoyIndex
from scipy.spatial import distance as spdistance


NUMBEROFTREES = 50

class DistanceIndex:
    def __init__(self, documentvectors):
        self.documentvectors = documentvectors
        self.index = AnnoyIndex( documentvectors.get_vector_size(), "angular")
        self.index_to_id = {}
        self.id_to_index = {}
        self.search_k_multiply = 2

    def build(self):
        """
        Build the index based on the documentvectors
        :return:
        """

        i = 0
        for dv in self.documentvectors:
            id = dv.get_id()
            self.index_to_id[i] = id
            self.id_to_index[id] = i
            self.index.add_item(i, dv.get_vector())
            i += 1

        self.index.build(NUMBEROFTREES)



    def calculate_relations(self, minimal_similarity, nearest_lim=2, second_index=None):
        """
        Determine the relations between the documents given the minimal distance
        :param minimal_similarity: value between 0 and 1
        :param second_index: The name of the index to compare to, if ommitted the index is compared to itself
        :param nearest_lim: Limit
        :return: a object with document relations
        """

        dr = DocumentRelations([])
        index_to_compare_to = second_index if not second_index is None else self

        for dv in self.documentvectors:
            src_id = dv.get_id()
            vector = np.array( dv.get_vector())
            src_index = self.id_to_index[src_id]

            (dest_indexes, distances) = index_to_compare_to.index.get_nns_by_vector(vector,n=nearest_lim + 1, search_k=(nearest_lim + 1)* self.search_k_multiply, include_distances=True)
            similarities = 1.0 / (1.0 + np.array( distances))

            added = 0
            for (dest_index, similarity) in zip( dest_indexes, similarities):
                if similarity > 0  and (second_index is None or dest_index != src_index):
                    if( float(similarity) >= minimal_similarity) and added < nearest_lim:
                        added += 1
                        dr.add(src=src_id, dest=index_to_compare_to.index_to_id[dest_index], similarity=float(similarity))

        return dr

    def calculate_relations_slow(self, minimal_similarity, second_index=None):
        """
        Determine the relations between the documents given the minimal distance, this is without using the ANN
        :param minimal_similarity: value between 0 and 1
        :param second_index: The name of the index to compare to, if ommitted the index is compared to itself
        :param nearest_lim: Limit
        :return: a object with document relations
        """

        vec1 = self.documentvectors.get_numpy_dict()
        vec2 = second_index.documentvectors.get_numpy_dict() if not second_index is None else self

        dr = DocumentRelations([])
        with tqdm(total=len(vec1.keys()), desc="Distance calculation") as progress:
            for src_id in vec1.keys():
                sims = []
                for dest_id in vec2.keys():
                    sim = 1. - spdistance.cosine( vec1[src_id], vec2[dest_id])
                    if( sim >= minimal_similarity):
                        sims.append( (dest_id, sim))

                sims.sort( key=lambda x: x[1], reverse=True)

                for (dest_id, sim) in sims[0:1]:
                    dr.add(src=src_id, dest=dest_id, similarity=sim)

                progress.update()

        return dr

    def calculate_relations_less_slow(self, minimal_similarity, second_index=None, maximum_number_of_results=100, corpus_pairs=None):
        """
        Determine the relations between the documents given the minimal distance, this is without using the ANN,
        but by using sklearn to compare to matrices
        :param minimal_similarity: value between 0 and 1
        :param second_index: The name of the index to compare to, if ommitted the index is compared to itself
        :param maximum_number_of_results:
        :param corpus_pairs: Create relations for the pairs in the corpus only
        :return: a object with document relations
        """

        (docids1, matrix1) = self.documentvectors.get_index_and_matrix()
        (docids2, matrix2) = (second_index if not second_index is None else self) .documentvectors.get_index_and_matrix()

        orgsims =  sklearn.metrics.pairwise.cosine_similarity(matrix1, matrix2, dense_output=False)
        # distances = 1 - orgsims
        # sims = 1.0 / (1.0 + distances)
        sims = orgsims
        last_sim_row = sims.shape[1]

        dr = DocumentRelations([])
        for i1 in range(0, len(docids1)):
            sorted = np.argsort( sims[i1]).tolist()
            sorted.reverse() # We want to order descending
            for iSorted in range(0, min( last_sim_row, maximum_number_of_results)):
                i2 = sorted[iSorted]
                if sims[i1, i2] >= minimal_similarity:
                    docid1 = docids1[i1]
                    docid2 = docids2[i2]
                    if corpus_pairs is None or corpus_pairs.pair_is_available( docid1, docid2):  # Only if we want this pair
                        if not(second_index is None) or docid1 != docid2:   # When camparing to the same index, skip identical documents
                            dr.add(src=docid1, dest=docid2, similarity=sims[i1, i2])
                else:
                    break

        return dr



    def cosine_sim(self, id1, id2):
        """
        Calculate the cosine similarity
        :param id1:
        :param id2:
        :return:
        """
        vector1 = self.documentvectors.get_documentvector(id1)
        vector2 = self.documentvectors.get_documentvector(id2)

        if not vector1 is None and not vector2 is None:
            return 1. / (1. + spdistance.cosine(vector1.get_vector(), vector2.get_vector()))
        else:
            return  None
