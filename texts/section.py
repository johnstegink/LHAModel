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


    def get_fulltext(self):
        """
        Get full text of the document
        :return:
        """
        return self.element.text_content()


    def get_sentences(self):
        """
        Returns a list of sentences
        :return:
        """

        return sent_tokenize( self.get_fulltext(), self.language)