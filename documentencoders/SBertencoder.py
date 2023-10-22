# Class for the Sentence Bert encoder
import os

from sentence_transformers import SentenceTransformer
from texts.clean import Cleaner

from documentencoders.documentencoder import Documentencoder_base


class SBertEcoder(Documentencoder_base):

    SBERTODELPATHS = {
        "en": "sentence-transformers/all-mpnet-base-v2",
        "nl": "sentence-transformers/distiluse-base-multilingual-cased-v1"
    }
    SBERTVECTORSIZE = 768
    # SBERTODELPATH = "sentence-transformers/all-MiniLM-L12-v2"
    # SBERTVECTORSIZE = 384

    def __init__(self, language_code):
        super(SBertEcoder, self).__init__(language_code)

        self.model = SentenceTransformer(SBertEcoder.SBERTODELPATH[language_code])
        self.cleaner = Cleaner(language_code=self.language_code)

    def get_vector_size(self):
        """
        The resulting vector size
        :return:
        """
        return SBertEcoder.SBERTVECTORSIZE      # This is because the Pretrained model has a fixed vector size


    def embed_text(self, text):
        """
        Create sentence to vec embedding
        :param text: can either be a string or a list of strings (sentences)
        :return:
        """

        joined = " ".join( text) if type(text) == list else text
        clean = self.cleaner.clean_text(txt=joined, remove_stop=True, remove_digits=True, lower=False)
        vector = self.model.encode( text)

        return vector            # return the vector
