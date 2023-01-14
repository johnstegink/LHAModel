# Class to generate or read embeddings for a document
# copied from https://github.com/SageAgastya/SmashRNN/blob/master/SmashRnn.py

class Embeddings:
    def __init(self, document, max_sections, max_sentences, max_words):
        self.document = document
        self.max_sessions = max_sections
        self.max_sentences = max_sentences
        self.max_words = max_words

