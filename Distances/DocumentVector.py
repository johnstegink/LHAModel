# Class to store a document vector

class DocumentVector:
    def __init__(self, documentid, vector):
        """
        Fill the class with the info
        :param documentid:
        :param vector: list of floats
        """
        self.documentid = documentid
        self.vector = vector

    def get_id(self):
        """
        Read the documentID
        :return:
        """
        return self.documentid

    def get_vector(self):
        """
        Read the vector as list of floats
        :return:
        """
        return self.vector