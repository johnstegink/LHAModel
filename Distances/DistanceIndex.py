# Class that contains functionality for computing the distance

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
        self.search_k_multiply = 20

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


    def calculate_relations(self, minimal_distance, nearest_lim=2):
        """
        Determine the relations between the documents given the minimal distance
        :param minimal_distance: value between 0 and 1
        :return: a object with document relations
        """

        dr = DocumentRelations()
        for dv in self.documentvectors:
            src_id = dv.get_id()
            src_index = self.id_to_index[src_id]
            (dest_indexes, distances) = self.index.get_nns_by_item(i=src_index,n=nearest_lim * self.search_k_multiply, include_distances=True)
            for i in range(0, len(dest_indexes)):
                distance = distances[i]
                dest_index = dest_indexes[i]
                if distance > 0  and dest_index != src_index :
                    cosine_distance = self.cosine_sim(src_id, self.index_to_id[dest_index])
                    if( cosine_distance >= minimal_distance):
                        dr.add(src=src_id, dest=self.index_to_id[dest_index], distance=cosine_distance)

        return dr.top( nearest_lim)

    def cosine_sim(self, id1, id2):
        """
        Calculate the cosine similarity
        :param id1:
        :param id2:
        :return:
        """
        vector1 = self.documentvectors.get_documentvector(id1).get_vector()
        vector2 = self.documentvectors.get_documentvector(id2).get_vector()

        return 1. / (1. + spdistance.cosine(vector1, vector2))
