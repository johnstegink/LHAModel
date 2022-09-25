# Class to read a corpus in the Common Format and iterates through documents
import functions
from texts.document import Document
from texts.similarities import Similarities
from pathlib import Path

class Corpus:

    def __init__(self, directory):
        """
        Read the corpus
        :param directory:
        """

        self.directory = directory
        (name, language_code) = functions.read_corpus_info(directory)
        self.name = name
        self.language_code = language_code

        # Create a directory of filenames (without extensions) and their corresponding path
        self.files = {Path(file).stem:file for file in functions.read_all_files_from_directory( self.directory, "xml")}
        self.ids = list(self.files.keys())

        self.language = functions.translate_language_code(language_code)


    def get_ids(self):
        """
        Returns a list of all ids of the corpus
        :return:
        """
        return self.ids

    def get_language(self):
        """
        Get the full name of the language
        :return:
        """
        return self.language

    def get_name(self):
        """
        Get the name of the corpus
        :return:
        """
        return self.name

    def get_language_code(self):
        """
        Get the language code of the corpus
        :return:
        """
        return self.language_code



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
        """
        Read the document by id
        :param id: 
        :return: 
        """""
        if not id in self.files:
            raise Exception( f"Unknown id {id}")

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
