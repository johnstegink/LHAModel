# Class to read and write document relations

from Distances.DocumentRelation import DocumentRelation
from lxml import etree as ET
import functions

class DocumentRelations:
    def __init__(self):
        self.relations = []

    def add(self, src, dest, distance):
        """
        Add a document relation
        :param src: source id
        :param dest: destination id
        :param distance: the distance (between 0 and 1)
        :return:
        """
        relation = DocumentRelation( src, dest, distance)
        self.relations.append( relation)

    def save(self, file):
        """
        Save the relations in the given Xml file
        :param file:the output file
        :return:
        """

        root = ET.fromstring("<relations></relations>")
        for relation in self.relations:
            document = ET.SubElement(root, "relation")
            ET.SubElement(document, "src").text = relation.get_src()
            ET.SubElement(document, "dest").text = relation.get_dest()
            ET.SubElement(document, "distance").text = str(relation.get_distance())

        # Write the file
        functions.write_file( file, functions.xml_as_string(root))


    @staticmethod
    def read(file):
        """
        Returns a new DocumentRelations object filled with the info in the Xml file
        :param file: xml file, that was created with a save
        :return: DocumentVectors object
        """
        dr = DocumentRelations()
        root = ET.parse(file).getroot()
        for document in root:
            src = document.find("src").text
            dest = document.find("dest").text
            distance = float(document.find("distance").text)
            dr.add( src, dest, distance)

        return dr

    def __iter__(self):
        """
        Initialize the iterator
        :return:
        """
        self.id_index = 0
        return self

    def __next__(self):
        """
        Next relation
        :return:
        """
        if self.id_index < len(self.relations):
            relation = self.relations[self.id_index]
            self.id_index += 1  # Ready for the next relation
            return relation

        else:  # Done
            raise StopIteration


    def count(self):
        """
        Count the number of relations
        :return:
        """

        return len(self.relations)


    def top(self, limit):
        """
        Returns a new Relations object with the top x relations, ordered by reversed distance
        :param limit:
        :return:
        """

        top = sorted( self.relations, key=(lambda rel: rel.get_distance() * -1))
        nw = DocumentRelations()
        for i in range(0, min(limit, len(top))):
            nw.add( top[i].get_src(), top[i].get_dest(), top[i].get_distance())

        return nw