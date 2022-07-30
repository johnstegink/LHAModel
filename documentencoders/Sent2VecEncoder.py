# Class for the sent2Vec encoder
import sent2vec as sent2vec

from documentencoders.documentencoder import Documentencoder_base


class Sent2VecEncoder(Documentencoder_base):

    sent2vecModelPath = "..

    def __init__(self, vector_dict, vector_size):
        super.__init__(self, vector_dict, vector_size)
        self.sent2vec_model = sent2vec.Sent2vecModel()
        self.sent2vec_model.load_model()

    def embed_text(self, text):
        """
        Create sentence to vec embedding
        :param text:
        :return:
        """

        self.sent2vec_model.load_model()
        return self.sent2vec_model.embed_sentence( text)