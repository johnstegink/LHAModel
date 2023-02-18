# class to preprosess the document into a list of paragraphs, containing a list of sentences, containing a list of words
import re
import os

import numpy
import torch
from nltk.tokenize import word_tokenize
from SMASH.CorpusDataset import CorpusDataset
from SMASH.BertEmbedder import BertEmbedder

class DocumentPreprocessor:
    def __init__(self, corpus, similarities, modelname, embeddingsdir, max_sections, max_sentences, max_tokens, device, debug = False, dim=768):
        """
        Fill the class properties
        :param corpus: The corpus containing the documents
        :param similarities:  The similarities from which to get the IDs
        """
        self.corpus = corpus
        self.similarities = similarities
        self.embeddingsdir = embeddingsdir
        self.max_sections = max_sections
        self.max_sentences = max_sentences
        self.max_tokens = max_tokens
        self.device = device
        self.dim = dim

        self.max_size = 50 if debug else None

        self.documentids = set()  # Contains al documentids
        for sim in similarities.get_all_similarities():
            self.documentids.add( sim.get_src())
            self.documentids.add( sim.get_dest())

        (self.train, self.test, self.validation) = self.__split_list( similarities)

        self.embedder = BertEmbedder(modelname, device)  # Load the model


    def get_device(self):
        """
        Return the device the model is running on
        :return:
        """
        return self.device



    def __split_list(self, similarities):
        """
        Split the similarities into a train, test and validation set (60,20,20)
        :param similarities:
        :return:
        """

        sims = similarities.get_all_similarities()
        if not self.max_size is None:
            sims = sims[0:self.max_size]
        test = int(len(sims) * 0.6)
        val = int(len(sims) * 0.8)

        return sims[:test], sims[test:val], sims[val:]



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

        for docid in self.documentids:
            doc = self.corpus.getDocument(docid)

            yield docid, self.__read_document( doc)


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

        nonalpha = re.compile(r"\W|[0-9]")
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


    def __update_dict(self, the_dict, length):
        """
        Update the dictionary of set the value to 1 if the length already exists
        :param the_dict:
        :param length:
        :return:
        """
        if length in the_dict:
            the_dict[length] += 1
        else:
            the_dict[length] = 1



    def get_statistics(self):
        """
        Returns the statistics for this set, containing dictionaries with the number of elements as key, and the number of occurrences a values
        :return: (sections, sentences, words)
        """

        sections = {}
        sentences = {}
        words = {}

        # maxdocs = 100
        # docs = 0
        for (docid, doc) in self.__read_documents():
            # docs += 1
            # if docs > maxdocs:
            #     break

            self.__update_dict( sections, len(doc))
            for section in doc:
                self.__update_dict( sentences, len(section))
                for sentence in section:
                    self.__update_dict( words, len(sentence))

        return sections, sentences, words


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


    def len_with_maximum(self, lst, maximum):
        """
        Returns the length of the list having a maximum value
        :param lst:
        :param maximum:
        :return:
        """
        return min(len(lst), maximum)


    def __bert_embeddings(self, text_data, max_sections, max_sentences, max_tokens, dim=768):
        result = torch.zeros((max_sections, max_sentences, dim))

        for section_index in range(self.len_with_maximum(text_data, max_sections)):
            section = text_data[section_index]

            # Create a list of sentences
            sentences = []
            for sent_index in range( self.len_with_maximum(section, max_sentences)):
                sentences.append( " ".join( section[sent_index]) + ".") # Construct a normal sentence

            # Embed the sentences at once
            sentence_embeddings = self.embedder.embed_sentences(sentences, max_tokens)

            # Put the embeddings in the result
            for sent_index in range(len( sentence_embeddings)):
                result[section_index][sent_index] = sentence_embeddings[sent_index]

        return result



    def __pad(self, the_input, section_cnt=2, sent_cnt=2, word_cnt=50, dim=1024):
        """
        Create a numpy matrix for this input
        :param the_input:
        :param section_cnt:
        :param sent_cnt:
        :param word_cnt:
        :param dim:
        :ret urn:
        """
        zeros = numpy.zeros((section_cnt, sent_cnt, word_cnt, dim))

        for section_index in range(self.len_with_maximum(the_input, section_cnt)):
            sent_embeddings = self.embedder.sents2elmo(self.__truncate_sentences(the_input[section_index], sent_cnt, word_cnt))
            for sent_index in range( self.len_with_maximum(sent_embeddings, sent_cnt)):
                line_embeddings = sent_embeddings[sent_index]
                for word_index in range( self.len_with_maximum( line_embeddings, word_cnt)):
                    zeros[section_index][sent_index][word_index] = line_embeddings[word_index]

        return zeros


    def __truncate_sentences(self, section, sent_cnt, word_cnt):
        """
        Truncate the number of sentences and number of words in the sentence
        :param section:
        :param sent_cnt:
        :param word_cnt:
        :return:
        """
        nw = []
        for sent_index in range(self.len_with_maximum(section, sent_cnt)):
            sentence = section[sent_index]
            if len( sentence) > word_cnt:
                sentence = sentence[:word_cnt]
            nw.append( sentence)

        return nw

    def create_or_load_embedding(self, docid):
        """
        Create or load the pytorch embeddings for the given document
        :param docid:
        :return:
        """

        # First create the embeddings directory if it is available
        directory = os.path.join( self.embeddingsdir, f"embeddings_{self.max_sections}_{self.max_sentences}_{self.max_tokens}_{self.dim}")
        os.makedirs( directory, exist_ok=True)

        file = os.path.join(directory, f"{docid}.pt")
        if os.path.exists( file):
            return torch.load( file)
        else:
            doc = self.corpus.getDocument(docid)
            text = self.__read_document( doc)
            tensor = self.__bert_embeddings( text, max_sections=self.max_sections, max_sentences=self.max_sentences, max_tokens=self.max_tokens, dim=self.dim)

            torch.save(tensor, file)

            return tensor

    def create_training_dataset(self):
        """
        Create the training set
        :return:
        """
        return CorpusDataset( self, self.train, self.device)

    def create_test_dataset(self):
        """
        Create the test set
        :return:
        """
        return CorpusDataset( self, self.test, self.device)

    def create_validation_dataset(self):
        """
        Create the validation set
        :return:
        """
        return CorpusDataset(self, self.validation, self.device)