# Class to read and write document similarities

from texts.similarity import Similarity
from lxml import etree as ET
import functions

class Similarities:
    def __init__(self):
        self.similarities = {}

    def add(self, src, dest, similarity):
        """
        Add a document relation
        :param src: source id
        :param dest: destination id
        :param similarity: (0, 1 or 2)
        :param overwrite the value if it already added
        :return:
        """

        if not src in self.similarities:
            self.similarities[src] = {}

        self.similarities[src][dest] = Similarity(src, dest, similarity)


    def save(self, file):
        """
        Save the similarities in the given Xml file
        :param file:the output file
        :return:
        """

        root = ET.fromstring("<similarities></similarities>")
        for src in self.similarities.keys():
            for dest in self.similarities[src].keys():
                similarity = self.similarities[src][dest]
                document = ET.SubElement(root, "relation")
                ET.SubElement(document, "src").text = similarity.get_src()
                ET.SubElement(document, "dest").text = similarity.get_dest()
                ET.SubElement(document, "similarity").text = str(similarity.get_similarity())

        # Write the file
        functions.write_file( file, functions.xml_as_string(root))


    @staticmethod
    def read(file):
        """
        Returns a new Similarities object filled with the info in the Xml file
        :param file: xml file, that was created with a save
        :return: Similarities object
        """
        sim = Similarities()
        root = ET.parse(file).getroot()
        for document in root:
            src = document.find("src").text
            dest = document.find("dest").text
            similarity = int(document.find("similarity").text)
            sim.add( src, dest, similarity)

        return sim

    def get_similiarties(self, src):
        """
        Returns a list of similarities of the source
        :param src:
        :return: List of similarities
        """

        if src in self.similarities:
            return list( self.similarities[src].values())
        else:
            return []   # No similarities


    def __iter__(self):
        """
        Initialize the iterator
        :return:
        """
        self.id_index = 0
        self.keys = list(self.similarities.keys())
        return self

    def __next__(self):
        """
        Next similarity
        :return:
        """
        if self.id_index < len(self.keys):
            similarity = self.similarities[self.keys[self.id_index]]
            self.id_index += 1  # Ready for the next similarity
            return similarity

        else:  # Done
            raise StopIteration

    def count(self):
        """
        Count the number of similarities
        :return:
        """
        return len(self.similarities.keys())