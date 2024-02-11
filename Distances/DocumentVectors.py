# Class to read and write documentvectors
import gc
import pickle

from Distances.DocumentVector import DocumentVector
from lxml import etree as ET
import numpy as np
import functions
import re

class DocumentVectors:
    def __init__(self, vectors):
        self.vectors = vectors

    def add(self, documentid, vector):
        """
        Add a document vector
        :param documentid:
        :param vector: list of floats
        :return:
        """
        documentvector = DocumentVector( documentid, vector)
        self.vectors[documentid] = documentvector

    def add_section(self, documentid, vector):
        """
        Add a section to the document
        :param documentid:
        :param vector:
        :return:
        """
        self.vectors[documentid].add_section( vector)


    def get_vector_size(self):
        """
        Read the vector size and check the file if all document vectors have the same size
        :return:
        """
        size = 0
        for vector in self.vectors.values():
            if size == 0:
                size = vector.get_vector_size()

            if vector.get_vector_size() != size:
                raise(f"Size of vector is not the right size {vector.get_vector_size()} instead of {size}")

        return size



    def save(self, filename):
        """
        Save the vectors in the given Xml file
        :param file:the output file
        :return:
        """

        file = open(filename, mode="w", encoding="utf-8-sig")
        file.write("<documents>\n")

        for vector in self.vectors.values():
            vectorValue = vector.get_vector()
            if hasattr(vectorValue, '__iter__'):
                document = ET.fromstring("<document></document>")
                ET.SubElement(document, "id").text = vector.get_id()
                # Comma seperated vector
                ET.SubElement(document, "vector").text = ",".join([str(value) for value in vectorValue])

                # Add the sections with the index in an attribute
                sections = ET.SubElement(document, "sections")
                for (section_index, section_vector) in vector.get_sections():
                    ET.SubElement(sections, "section", attrib={"index": str(section_index)}).text = ",".join([str(value) for value in section_vector])

                file.write( functions.xml_as_string(document))
                del document


        # Write the end of the file
        file.write("</documents>\n")
        file.close()

        # Write the file
        functions.write_pickle(filename, self.vectors)


    @staticmethod
    def read(file, id_filter=None):
        """
        Returns a new DocumentVectors object filled with the info in the Xml file
        :param file: xml file, that was created with a save
        :param id_filter: regular expression that filters the ids
        :return: DocumentVectors object
        """

        dv = functions.read_from_pickle(file)
        if dv is None:
            dv = DocumentVectors({})
            root = ET.parse(file).getroot()
            for document in root:
                id = document.find("id").text
                if id_filter is None or id_filter.match( id):
                    if not document.find("vector").text is None:
                        vector = [float(value) for value in document.find("vector").text.split(",")]
                        dv.add( id, vector)
                        for section in (document.find("sections").findall("section")):
                            if not section.text is None:
                                sectionvector = [float(value) for value in section.text.split(",")]
                                dv.add_section( id, sectionvector)
            functions.write_pickle( file, dv)

            del root

        else:
            if type(dv).__name__ == 'dict': dv = DocumentVectors(dv)

        return dv


    def get_numpy_dict(self):
        """
        Get the vectors in a numpy dict with the ID as key and the vector as a numpy array
        :return:
        """

        np_dict = {}
        for vector in self.vectors.values():
            np_dict[vector.documentid] = np.array(vector.vector, dtype=np.float32)

        return np_dict

    def get_index_and_matrix(self):
        """
        Returns a matrix with all values of all vectors with the rows being the vectors
        The indexes are the documentids in the rows
        :return: (indexes, matrix)
        """
        indexes = []
        matrix = []

        for vector in self.vectors.values():
            indexes.append( vector.documentid)
            matrix.append(vector.vector)

        return (indexes, matrix)



    def get_index_and_matrix_for_list(self, ids):
        """
        Returns a matrix with all values of all vectors with the given ids
        The indexes are the documentids in the rows
        :param src: src id
        :param dest_list: list of destination ids
        :return: (indexes, matrix)
        """

        indexes = []
        matrix = []

        for vector in self.vectors.values():
            if vector.get_id() in ids:
                indexes.append( vector.documentid)
                matrix.append(vector.vector)

        return (indexes, matrix)



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

        if id in self.vectors:
            return self.vectors[id]
        else:
            return None

    def documentvector_exists(self, id):
        """
        Determines wheather a document vector with this Id exists
        :param id:
        :return:
        """

        return id in self.vectors
