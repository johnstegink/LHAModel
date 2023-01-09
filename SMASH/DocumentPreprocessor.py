# class to preprosess the document into a list of paragraphs, containing a list of sentences, containing a list of words
import re

from nltk.tokenize import word_tokenize

class DocumentPreprocessor:
    def __init__(self, corpus, similarities):
        """
        Fill the class properties
        :param corpus: The corpus containing the documents
        :param similarities:  The similarities from which to get the IDs
        """
        self.corpus = corpus

        self.documentids = set()  # Contains al documentids
        for sim in similarities.get_all_similarities():
            self.documentids.add( sim.get_src())
            self.documentids.add( sim.get_dest())

        self.documents = {}

    def __read_documents(self):
        """
        Read all documents
        :return:
        """

        nonalpha = re.compile(r"[^\w]|[0-9]")

        for id in self.documentids:
            sections = []
            for section in self.corpus.getDocument(id):
                sentences = []
                for sentence in section.get_sentences():
                    # Remove words with only no alpha characters
                    words =  [word for word in word_tokenize( sentence) if nonalpha.sub("", word) != ""]
                    if len(words) > 0:
                        sentences.append( words)

                if len( sentences) > 0:
                    sections.append( sentences)

            yield (id, sections)

    def __update_dict(self, dict, len):
        """
        Update the dictionary of set the value to 1 if the length already exists
        :param dict:
        :param len:
        :return:
        """
        if len in dict:
            dict[len] += 1
        else:
            dict[len] = 1

    def get_statistics(self):
        """
        Returns the statistics for this set, containing dictionaries with the number of elements as key, and the number of occurrences a values
        :return: (sections, sentences, words)
        """

        sections = {}
        sentences = {}
        words = {}

        for (id, doc) in self.__read_documents():
            self.__update_dict( sections, len(doc))
            for section in doc:
                self.__update_dict( sentences, len(section))
                for sentence in section:
                    self.__update_dict( words, len(sentence))

        return (sections, sentences, words)
