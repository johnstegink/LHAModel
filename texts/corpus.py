# Class to read a corpus in the Common Format and iterates through documents
import functions
from texts.document import Document

class Corpus:

    def __init__(self, name, directory, language_code):
        """
        Read the corpus
        :param name:
        :param directory:
        :param language_code:
        """
        self.name = name
        self.directory = directory
        self.files = functions.read_all_files_from_directory( self.directory, "xml")
        self.language_code = language_code
        self.language = functions.translate_language_code(language_code)


    def get_number_of_documents(self):
        """
        Returns the number of documents
        :return:
        """
        return len(self.files)


    def __iter__(self):
        """
        Initialize the iterator
        :return:
        """
        self.file_index = 0
        return self



    def __next__(self):
        """
        Next document from the corpus
        :return:
        """
        if self.file_index < len( self.files):
            document = Document( filename=self.files[self.file_index], language=self.language)
            self.file_index += 1  # Ready for the next document
            return document

        else:  # Done
            raise StopIteration