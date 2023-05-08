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
        self.sections = []

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

    def get_vector_size(self):
        """
        Determines the size of the vector
        :return:
        """

        return len( self.vector)


    def add_section(self,  vector):
        """
        Add a section to this document vector
        :param vector:
        :return:
        """
        self.sections.append( vector)


    def get_sections(self):
        """
        Read all sections
        :return: a list of tuples with the index of the section and the vector
        """
        sections = []
        for index in range( len(self.sections)):
            sections.append((index, self.sections[index]))

        return  sections
