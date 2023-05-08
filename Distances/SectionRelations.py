# Class to read and write section relations


from Distances.SectionRelation import SectionRelation
from lxml import etree as ET
import html
import functions

class SectionRelations:
    def __init__(self, id, similarity):
        """
        Relations between sections of src and dest document
        :param dest: id of destination document
        :param id: the similarity between the documents
        """
        self.relations = SectionRelation(id, similarity)   # Dictionary of relations
        self.id = id
        self.similarity = similarity

    def add(self, dest, similarity):
        """
        Add a document relation
        :param src: source id of section
        :param dest: destination id of section
        :param similarity: the distance (between 0 and 1)
        :return:
        """

        self.relations.add_destination( dest, similarity)

    def get_id(self):
        """
        The id of the destination
        :return:
        """

        return self.id


    def get_similarity(self):
        """
        Returns the similarity
        :return:
        """
        return self.similarity


    def get_relations(self):
        """
        Returns a list of all destinations
        :return:
        """
        return self.relations

