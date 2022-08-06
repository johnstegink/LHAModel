# Class to store a document relation

class DocumentRelation:
    def __init__(self, src, dest, distance):
        """
        Fill the class with the info
        :param src:
        :param dest:
        :param distance: float
        """
        self.src = src
        self.dest = dest
        self.distance = distance

    def get_src(self):
        """
        Read the source id
        :return:
        """
        return self.src

    def get_dest(self):
        """
        Read the destination id
        :return:
        """
        return self.dest


    def get_distance(self):
        """
        Read the distance
        :return:
        """
        return self.distance

