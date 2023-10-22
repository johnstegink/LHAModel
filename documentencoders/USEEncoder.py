# Class for the Universal Sentence Encoder

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from absl import logging

from sentence_transformers import SentenceTransformer
from texts.clean import Cleaner

from documentencoders.documentencoder import Documentencoder_base


class USEEcoder(Documentencoder_base):
    module_urls = {
     "en" :"https://tfhub.dev/google/universal-sentence-encoder/4",
     "nl": "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
    }

    USEVECTORSIZE = 512

    def __init__(self, language_code):
        super(USEEcoder, self).__init__(language_code)

        self.model =  hub.load(self.module_urls[language_code])
        self.cleaner = Cleaner(language_code=self.language_code)

    def get_vector_size(self):
        """
        The resulting vector size
        :return:
        """
        return USEEcoder.USEVECTORSIZE      # This is because the Pretrained model has a fixed vector size


    def embed_text(self, text):
        """
        Create sentence to vec embedding
        :param text: can either be a string or a list of strings (sentences)
        :return:
        """
        # Reduce logging output.
        logging.set_verbosity(logging.ERROR)
        joined = " ".join( text) if type(text) == list else text
        vector = self.model( joined)

        return vector            # return the vector
