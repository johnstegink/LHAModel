# Base class for a document encoder

class Documentencoder_base:

    def __init__(self, vector_dict, vector_size):
        """
        Initialisation of the embedder
        :param vector_dict:
        :param vector_size:
        """
        self.vector_dict = vector_dict
        self.vector_size = vector_size


    def clean_text(self, text):
        raise("Not implemented")

    def embed_text(self, text):
        raise("Not implemented")