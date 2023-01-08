# Class that contains a section from a document
import functions
from lxml import etree as ET
from nltk.tokenize import sent_tokenize


class Section:

    def __init__(self, element, language):
        """
        Gets information about the section
        :param element: xml element containing the section
        :param language: full name of the language
        """
        self.element = element
        self.language = language



    def get_id(self):
        """
        Returns the documentID
        :return:
        """

        return self.element.attrib["id"]

    def get_title(self):
        if self.element.find("title") is None:
            print( functions.xml_as_string( self.element))

        title = self.element.find("title").text
        return title if not title is None else ""

    def get_text(self):
        if self.element.find("text") is None:
            print( functions.xml_as_string( self.element))

        text = self.element.find("text").text
        return text if not text is None else ""


    def get_fulltext(self):
        """
        Get full text of the document
        :return:
        """

        return self.get_title() + " " + self.get_text()


    def get_sentences(self):
        """
        Returns a list of sentences
        :return:
        """

        return sent_tokenize( self.get_fulltext(), self.language)

