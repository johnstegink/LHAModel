# Class to read and write document relations

from Distances.DocumentRelation import DocumentRelation
from lxml import etree as ET
import html
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


    def save_html(self, src_corpus, output, dest_corpus = None):
        """
        Save the relations as a html file
        :param src_corpus: The originating corpus
        :param output:
        :param dest_corpus: The destination corpus
        :return:
        """

        # Set the destination corpus
        dst_corpus = dest_corpus if not dest_corpus is None else src_corpus

        with open( output, "w", encoding="utf-8") as htmlfile:
            htmlfile.write(f"<html>\n<head>\n<meta charset='UTF-8'>\n<title>Relations</title>\n</head>")
            htmlfile.write(f"<body>\n")
            htmlfile.write(f"<table>\n")
            htmlfile.write(f"<tr>\n<th>{html.escape(src_corpus.get_name())}</th><th>{html.escape(dst_corpus.get_name())}</th><th>Similarity</th></tr>")
            for relation in self.relations:
                src = src_corpus.getDocument(relation.get_src())
                dest = dst_corpus.getDocument(relation.get_dest())

                htmlfile.write("<tr>\n")
                htmlfile.write(f"<td>{src.create_html_link(target='link1')}</td>\n")
                htmlfile.write(f"<td>{dest.create_html_link(target='link2')}</td>\n")
                htmlfile.write(f"<td>{relation.get_distance():0.2}</td>\n")
                htmlfile.write("</tr>\n")
            htmlfile.write("</table>\n</body>\n</html>\n")


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
