# Class to read and write document relations
import gc

from Distances.DocumentRelation import DocumentRelation
from lxml import etree as ET
import html
import functions

class DocumentRelations:
    def __init__(self, relations):
        self.relations = relations
        self.is_dirty = True

    def add(self, src, dest, similarity):
        """
        Add a document relation
        :param src: source id
        :param dest: destination id
        :param similarity: the similarity (between 0 and 1)
        :return:
        """
        relation = DocumentRelation(src, dest, similarity)
        self.relations.append( relation)
        self.is_dirty = True


    def save(self, file, parameters):
        """
        Save the relations in the given Xml file, the params are added ass attributes to the root node
        :param file:the output file
        :param parameters: dictionary of parameters
        :return:
        """

        root = ET.fromstring("<relations></relations>")
        for (name, value) in parameters.items():
            root.set(name, value)

        for relation in self.relations:
            document = ET.SubElement(root, "relation")

            ET.SubElement(document, "src").text = relation.get_src()
            ET.SubElement(document, "dest").text = relation.get_dest()
            ET.SubElement(document, "similarity").text = str(relation.get_similarity())

        # Write the file
        functions.write_file( file, functions.xml_as_string(root))
        functions.write_pickle( file, self.relations)

    def save_html(self, src_corpus, output, dest_corpus = None, startof_id_filter = None):
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
            rels = list(self.relations)
            rels.sort( key=lambda rel: rel.get_src())
            for relation in rels:
                if startof_id_filter is None or relation.get_src().startswith( startof_id_filter) or relation.get_dest().startswith( startof_id_filter):
                    src = src_corpus.getDocument(relation.get_src())
                    dest = dst_corpus.getDocument(relation.get_dest())

                    htmlfile.write("<tr ")
                    if relation.get_src() == relation.get_dest():
                        htmlfile.write(" style='color: orange; font-weight: bold;'")
                    htmlfile.write('>')
                    htmlfile.write(f"<td>{src.create_html_link(target='link1', language_code='simple')}</td>\n")
                    htmlfile.write(f"<td>{dest.create_html_link(target='link2')}</td>\n")
                    htmlfile.write(f"<td>{relation.get_similarity():0.2}</td>\n")
                    htmlfile.write("</tr>\n")
            htmlfile.write("</table>\n</body>\n</html>\n")


    @staticmethod
    def read(file):
        """
        Returns a new DocumentRelations object filled with the info in the Xml file, together with the parameters that generated the file
        :param file: xml file, that was created with a save
        :return: Tuple (DocumentVectors object, attributes dictionary)
        """

        dr = functions.read_from_pickle( file, "dr")
        attr = functions.read_from_pickle( file, "attr")
        if dr is None or attr is None:
            dr = DocumentRelations([])
            root = ET.parse(file).getroot()
            for document in root:
                src = document.find("src").text
                dest = document.find("dest").text
                similarity = float(document.find("similarity").text)
                dr.add( src, dest, similarity)

            # copy the attributes
            attr = {}
            for name in root.attrib:
                attr[name] = str(root.attrib[name])

            functions.write_pickle( file, { "dr":dr, "attr": attr})
        else:
            if type(dr).__name__ == 'list': dr = DocumentRelations(dr)

        return (dr, attr)

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



    def get_relations_of(self, id):
        """
        Get all relations with the given src id
        :param id:
        :return:
        """

        if self.is_dirty:
            # Fill a dictionary with the src as key and all relations as values
            self.per_src = {}
            for rel in self.relations:
                src = rel.get_src()
                if not src in self.per_src:
                    self.per_src[src] = []
                self.per_src[src].append( rel)
            self.is_dirty = False

        return self.per_src[id] if id in self.per_src else []


    def get_similarity(self, src, dest):
        """
        Returns the similarity for the relation, 0 if no such relation exists
        :param src: id of the source article
        :param dest: id of the destination article
        :return: similarity, 0 if it does not exist
        """
        for rel in self.relations:
            if rel.get_src() == src  and  rel.get_dest() == dest:
                return rel.get_similarity()

        return 0

    def __len__(self):
        """
        The number of relations
        :return:
        """
        return len( self.relations)