# Class to read and write documentvectors

from Distances.DocumentVector import DocumentVector
from lxml import etree as ET
import numpy as np
import functions
import re

class DocumentVectors:
    def __init__(self):
        self.vectors = {}

    def add(self, documentid, vector):
        """
        Add a document vector
        :param documentid:
        :param vector: list of floats
        :return:
        """
        documentvector = DocumentVector( documentid, vector)
        self.vectors[documentid] = documentvector


    def get_vector_size(self):
        """
        Read the vector size and check the file if all vectors have the same size
        :return:
        """
        size = 0
        for vector in self.vectors.values():
            if size == 0:
                size = vector.get_vector_size()

            if vector.get_vector_size() != size:
                raise(f"Size of vector is not the right size {vector.get_vector_size()} instead of {size}")

        return size



    def save(self, file):
        """
        Save the vectors in the given Xml file
        :param file:the output file
        :return:
        """

        root = ET.fromstring("<documents></documents>")
        for vector in self.vectors.values():
            document = ET.SubElement(root, "document")
            ET.SubElement(document, "id").text = vector.get_id()
            # Comma seperated vector
            ET.SubElement(document, "vector").text = ",".join([str(value) for value in vector.get_vector()])

        # Write the file
        functions.write_file( file, functions.xml_as_string(root))


    @staticmethod
    def read(file, id_filter=None):
        """
        Returns a new DocumentVectors object filled with the info in the Xml file
        :param file: xml file, that was created with a save
        :param id_filter: regular expression that filters the ids
        :return: DocumentVectors object
        """
        dv = DocumentVectors()
        root = ET.parse(file).getroot()
        for document in root:
            id = document.find("id").text
            if id_filter is None or id_filter.match( id):
                vector = [float(value) for value in document.find("vector").text.split(",")]
                dv.add( id, vector)

        return dv


    def get_numpy_dict(self):
        """
        Get the vectors in a numpy dict with the ID as key and the vector as a numpy array
        :return:
        """

        np_dict = {}
        for vector in self.vectors.values():
            np_dict[vector.documentid] = np.array(vector.vector, dtype=np.float32)

        return np_dict;



    def __iter__(self):
        """
        Initialize the iterator
        :return:
        """
        self.ids = list( self.vectors.keys())
        self.id_index = 0
        return self



    def __next__(self):
        """
        Next documentvector
        :return:
        """
        if self.id_index < len(self.ids):
            vector = self.vectors[self.ids[self.id_index]]
            self.id_index += 1  # Ready for the next vector
            return vector

        else:  # Done
            raise StopIteration


    def get_documentvector(self, id):
        """
        Get the document vector of the document with the given id
        :param id:
        :return:
        """
        return self.vectors[id]