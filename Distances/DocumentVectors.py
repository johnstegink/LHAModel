# Class to read and write documentvectors

from Distances.DocumentVector import DocumentVector
from lxml import etree as ET
import functions

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
    def read(file):
        """
        Returns a new DocumentVectors object filled with the info in the Xml file
        :param file: xml file, that was created with a save
        :return: DocumentVectors object
        """
        dv = DocumentVectors()
        root = ET.parse(file).getroot()
        for document in root:
            id = document.find("id").text
            vector = [float(value) for value in document.find("vector").text.split(",")]
            dv.add( id, vector)

        return dv
