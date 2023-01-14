# class to preprosess the document into a list of paragraphs, containing a list of sentences, containing a list of words
import re
import os

import numpy
import torch
from nltk.tokenize import word_tokenize
from elmoformanylangs import Embedder

class DocumentPreprocessor:
    def __init__(self, corpus, similarities, elmomodel):
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

        self.elmomodel = elmomodel
        self.embedder = Embedder(self.elmomodel)  # Load the model


    def get_all_documentids(self):
        """
        Returns a list of all documentids
        :return:
        """

        return list( self.documentids)



    def __read_documents(self):
        """
        Read all documents
        :return:
        """

        for id in self.documentids:
            doc = self.corpus.getDocument(id)

            yield (id, self.__read_document( doc))


    def __read_document(self, doc):
        """
        Read the document, split into sections, lines and words
        :return:
        [
            [
                ["First", "sentence", "of", "the", "first", "section"],
                ["Second", "sentence", "of", "the", "first", "section"],
                ...
            ],
            [
                ["First", "sentence", "of", "the", "second", "section"],
                ["Second", "sentence", "of", "the", "second", "section"],
                ...
            ]
        ]
        """

        nonalpha = re.compile(r"[^\w]|[0-9]")
        sections = []
        for section in doc:
            sentences = []
            for sentence in section.get_sentences():
                # Remove words with only no alpha characters
                words =  [word for word in word_tokenize( sentence) if nonalpha.sub("", word) != ""]
                if len(words) > 0:
                    sentences.append( words)

            if len( sentences) > 0:
                sections.append( sentences)

        return sections


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


    def __embed(self, doc):
        """
        Embed the document, as split in lines, sections etc
        :param doc:
        :return:
        """

        arrs = []
        for section in doc:
            arrs.append( numpy.array( self.embedder.sents2elmo( section)))

        together =  torch.tensor( numpy.array( arrs))
        return torch.tensor( together)


    def len_with_maximum(self, list, max):
        """
        Returns the length of the list having a maximum value
        :param list:
        :param max:
        :return:
        """
        return min( len(list), max)

    def __pad(self, input, section_cnt=2, sent_cnt=2, word_cnt=50, dim=1024):
        """
        Create a numpy matrix for this input
        :param input:
        :param section_cnt:
        :param sent_cnt:
        :param word_cnt:
        :param dim:
        :return:
        """
        sent_cnt += 1
        zeros = numpy.zeros((section_cnt, sent_cnt, word_cnt, dim))

        for section_index in range( self.len_with_maximum( input, section_cnt)):
            sent_embeddings = self.embedder.sents2elmo( input[section_index])
            for sent_index in range( self.len_with_maximum(sent_embeddings, sent_cnt)):
                line_embeddings = sent_embeddings[sent_index]
                for line_index in range( self.len_with_maximum( line_embeddings, word_cnt)):
                    zeros[section_index][sent_index][line_index] = line_embeddings[line_index]

        return torch.from_numpy( zeros)


    def CreateOrLoadEmbeddings(self, id, embeddingsdir, max_sections, max_sentences, max_words, dim=1024 ):
        """
        Create or load the numpy embeddings for the given document as a tensor
        :param id:
        :param embeddingsdir:
        :param max_sections:
        :param max_sentences:
        :param max_words:
        :param dim:
        :return:
        """

        # First create the embeddings directory if it is available
        dir = os.path.join( embeddingsdir, f"embeddings_{max_sections}_{max_sentences}_{max_words}_{dim}")
        os.makedirs( dir, exist_ok=True)

        file = os.path.join( embeddingsdir, f"{id}.pt")
        if os.path.exists( file):
            return torch.load(file)
        else:
            doc = self.corpus.getDocument( id)
            text = self.__read_document( doc)
            tensor = self.__pad( text, section_cnt=max_sections, sent_cnt=max_sentences, word_cnt=max_words, dim=dim)
            torch.save( tensor, file)

            return tensor
