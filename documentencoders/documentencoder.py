# Base class for a document encoder

class Documentencoder_base:

    def __init__(self, language_code, vector_size):
        """
        Initialisation of the embedder
        :param vector_dict:
        :param vector_size:
        :param language_code
        """
        self.vector_size = vector_size
        self.language_code = language_code



    def clean_text(self, text):
        raise("Not implemented")



    def get_vector_size(self):
        raise("Not implemented")


    def embed_text(self, text):
        raise("Not implemented")