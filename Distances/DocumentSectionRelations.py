# The Xml file has the form
#   <sectionrelations>
#      <srcdoc id="">
#         <destdoc id="" similarity="">
#           <section src="" dest="" similarity="">
#           <section src="" dest="" similarity="">
#           ...
#         </destdoc>
#         ...
#      <srcdoc id="" similarity="">
#         ...
#      </srcdoc>
#   </sectionrelations>

from Distances.SectionRelations import SectionRelations
from lxml import etree as ET
import html
import functions

class DocumentSectionRelations:
    def __init__(self):
        self.relations = {}   # Dictionary of relations, with the src as a key, and a list of relations as value

    def add(self, src, dest, similarity):
        """
        Add a document relation
        :param src: id of source document
        :param dest: id of destination document
        :param similarity: the similarity (between 0 and 1)
        :return: the sectionrelations object
        """

        sectionrelations = SectionRelations( dest, similarity)
        if not src in self.relations:
            self.relations[src] = [sectionrelations]
        else:
            self.relations[src].append( sectionrelations)

        return sectionrelations


    def save(self, file):
        """
        Save the relations in the given Xml file
        :param file:the output file
        :return:
        """

        root = ET.fromstring("<sectionrelations></sectionrelations>")

        for src_doc in self.relations.keys:
            src_doc_node = ET.SubElement(root, "srcdoc", attrib={"id": src_doc})
            for dest_doc in self.relations[src_doc]:
                dest_doc_node = ET.SubElement(src_doc_node, "destdoc", attrib={"id": dest_doc.get_id(), "similarity" : dest_doc.get_similarity()})
                for relation in dest_doc_node.get_relations():
                    ET.SubElement( dest_doc_node, "section", attrib={"src": relation.get_src(), "dest": relation.get_dest(), "similarity": relation.get_similarity()})

        # Write the file
        functions.write_file( file, functions.xml_as_string(root))


    @staticmethod
    def read(file):
        """
        Returns a new DocumentRelations object filled with the info in the Xml file, together with the parameters that generated the file
        :param file: xml file, that was created with a save
        :return: Tuple (DocumentVectors object, attributes dictionary)
        """
