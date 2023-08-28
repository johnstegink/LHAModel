# Class to store a relation of sections between two documents

class SectionRelation:
    def __init__(self, src, dest, similarity):
        """
        Fill the class with the info
        :param src: id of the source section
        :param dest: id of the destination section
        :param similarity: the corresponding similarity
        """
        self.src = src
        self.dest = dest
        self.similarity = similarity

    def get_dest(self):
        """
        Read the destination id
        :return:
        """
        return self.dest

    def get_dest_place(self):
        """
        Returns the place of the destination section in the document
        :return:
        """

        return int( self.dest[1:])

    def get_src_place(self):
        """
        Returns the place of the source section in the document
        :return:
        """

        return int(self.dest[1:])

    def get_src(self):
        """
        Read the source id
        :return:
        """
        return self.src

    def get_similarity(self):
        """
        :return: a list of tuples (id, similarity) of all destinations

        """
        return self.similarity

