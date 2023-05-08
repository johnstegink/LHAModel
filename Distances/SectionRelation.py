# Class to store a relation of sections between two documents

class SectionRelation:
    def __init__(self, id, similarity):
        """
        Fill the class with the info
        :param id: the source section
        """
        self.id = id
        self.srsimilarityc = similarity
        self.destinations = []

    def get_id(self):
        """
        Read the source id
        :return:
        """
        return self.id

    def get_destinations(self):
        """
        :return: a list of tuples (id, similarity) of all destinations

        """
        return self.destinations


    def add_destination(self, destination, similarity):
        """
        Add this destination to the list of destinations
        :param destination:
        :param similarity:
        :return:
        """
        self.destinations.append(( destination, similarity))


