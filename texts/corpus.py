# Class to read a corpus in the Common Format and iterates through documents
import functions
from texts.document import Document
from texts.similarities import Similarities
from pathlib import Path

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

        # Create a directory of filenames (without extensions) and their corresponding path
        self.files = {Path(file).stem:file for file in functions.read_all_files_from_directory( self.directory, "xml")}
        self.ids = list(self.files.keys())

        self.language_code = language_code
        self.language = functions.translate_language_code(language_code)


    def get_number_of_documents(self):
        """
        Returns the number of documents
        :return:
        """
        return len(self.ids)


    def get_id_of_index(self, index):
        """
        Translate the document id to an index
        :param index:
        :return:
        """

        if index >= len(self.ids):
            raise(f"Index {index} too large, max: {len(self.ids)}")

        return self.ids[index]


    def get_index_of_id(self, id):
        """
        Translate the index to a docid
        :param id:
        :return:
        """

        return self.ids.index( id)


    def __iter__(self):
        """
        Initialize the iterator
        :return:
        """
        self.id_index = 0
        return self



    def __next__(self):
        """
        Next document from the corpus
        :return:
        """
        if self.id_index < len(self.ids):
            document = self.getDocument( self.get_id_of_index( self.id_index))

            self.id_index += 1  # Ready for the next document
            return document

        else:  # Done
            raise StopIteration


    def getDocument(self, id):
        if not id in self.files:
            raise( f"Unknown id {id}")

        document =  Document(filename=self.files[id], language=self.language, index=self.get_index_of_id( id))
        return document


    def read_similarities(self):
        """
        Return a new similarities object with all similarities in this corpus
        :return:
        """

        sim = Similarities()
        for src in self.ids:
            document = self.getDocument(src)
            for (dest, similarity) in document.get_links():
                sim.add( src, dest, similarity, True)
                sim.add( dest, src, similarity, False)  # Optional

        return sim
