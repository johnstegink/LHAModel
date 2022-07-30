# Class that contains a document from the corpus
import functions
from lxml import etree as ET
from texts.section import Section

class Document:

    def __init__(self, filename, language):
        """
        Read the documents from the file
        :param filename:
        :param language: full name of the language
        """
        self.filename = filename
        self.xml = ET.parse(filename)
        self.element = self.xml.getroot()
        self.sections = self.element.findall("section")
        self.language = language


    def get_id(self):
        """
        Returns the documentID
        :return:
        """

        return self.element.attrib["id"]


    def get_fulltext(self):
        """
        Get full text of the document
        :return:
        """
        return self.element.text_content()


    def __iter__(self):
        """
        Initialize the iterator through the sections
        :return:
        """
        self.section_index = 0
        return self



    def __next__(self):
        """
        Next section from the document
        :return:
        """

        if self.section_index < len( self.sections):
            section_elem = self.sections[self.section_index]
            self.file_index += 1  # Ready for the next section
            return Section( element=section_elem, language=self.language)

        else:  # Done
            raise StopIteration