# Class that contains a document from the corpus
import functions
from lxml import etree as ET
from texts.section import Section

class Document:

    def __init__(self, filename, language, index):
        """
        Read the documents from the file
        :param filename:
        :param language: full name of the language
        :param index: numeric index (starting at 0)
        """
        self.filename = filename
        self.xml = ET.parse(filename)
        self.element = self.xml.getroot()
        self.sections = self.element.findall("section")
        self.language = language
        self.index = index

    def get_id(self):
        """
        Returns the documentID
        :return:
        """

        return self.element.attrib["id"]

    def get_index(self):
        """
        Returns the index
        :return:
        """
        return self.index


    def get_title(self):
        title = self.element.find("title").text
        return title if not title is None else ""


    def get_fulltext(self):
        """
        Get full text of the document, by getting the title and texts of the sections
        :return:
        """

        texts = []
        texts.append( self.get_title())
        for section in self:
            texts.append( section.get_fulltext())

        return " ".join( texts)


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
            self.section_index += 1  # Ready for the next section
            return Section( element=section_elem, language=self.language)

        else:  # Done
            raise StopIteration


    def get_links(self):
        """
        Returns a list of tuples with the links
        :return: [(id, similarity)] a list of tuples with the documentid of the destination and the similarity (0,1 or 2)
        """

        info = []
        links = self.element.find("links")
        if not links is None:
            for link in links.findall("link"):
                dest = link.attrib["id"]
                similarity =  int(link.attrib["class"]) if "class" in link.attrib else 2
                info.append( (dest, similarity))

        return info