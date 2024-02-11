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
import gc
import os.path
import pickle

from tqdm import tqdm

from Distances.SectionRelations import SectionRelations
from lxml import etree as ET
import html
import functions

class DocumentSectionRelations:
    def __init__(self, relations):
        self.relations = relations   # Dictionary of relations, with the src as a key, and a list of relations as value

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

    def get_section_relations(self, src, dest):
        """
        returns all similar sections of the given source and destination
        :param src: Id of source
        :param dest: Id of destination
        :return:
        """

        if src in self.relations:
            for relation in self.relations[src]:
                if relation.get_dest() == dest:
                    return relation.get_relations()

        return []       # No relations


    def save(self, filename):
        """
        Save the relations in the given Xml file
        :param filename:the output file
        :return:
        """

        file = open(filename, mode="w", encoding="utf-8-sig")
        file.write("<sectionrelations>\n")

        for src_doc in self.relations.keys():
            src_doc_node = ET.fromstring("<srcdoc></srcdoc>")
            src_doc_node.attrib["id"] = src_doc
            for dest_doc in self.relations[src_doc]:
                dest_doc_node = ET.SubElement(src_doc_node, "destdoc", attrib={"id": dest_doc.get_dest(), "similarity" : str(dest_doc.get_similarity())})
                for sect_relation in dest_doc.get_relations():
                    ET.SubElement( dest_doc_node, "section", attrib={"src": sect_relation.get_src(), "dest": sect_relation.get_dest(), "similarity": str( sect_relation.get_similarity())})

            file.write( functions.xml_as_string(src_doc_node))
            del src_doc_node

        # Write the end of the file
        file.write("</sectionrelations>\n")
        file.close()

        functions.write_pickle(filename, self.relations)


    @staticmethod
    def read(file):
        """
        Returns a new DocumentSectionsRelations object filled with the info in the Xml file
        :param file: xml file, that was created with a save
        :return: DocumentSectionsRelations object
        """

        drs = functions.read_from_pickle( file)
        if drs is None:
            drs = DocumentSectionRelations({})
            print("Counting...")
            nr_of_relations = functions.count_elements( file, "srcdoc")
            with tqdm(total=nr_of_relations, desc="Reading relations") as progress:
                for src_doc in functions.iterate_xml( file):
                    for dest_doc in src_doc:
                        dest = drs.add( src_doc.attrib["id"], dest_doc.attrib["id"], float(dest_doc.attrib["similarity"]) )
                        for section in dest_doc:
                            dest.add_section( section.attrib["src"], section.attrib["dest"], float( section.attrib["similarity"]))
                    progress.update()

            functions.write_pickle( file, drs)
        else:
            if type(drs).__name__ == 'dict': drs = DocumentSectionRelations(drs)

        return drs




