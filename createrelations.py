# Script to create relations based on documentvectors

import argparse

import sys
import os
import functions
from Distances.DocumentVectors import  DocumentVectors
from Distances.DistanceIndex import  DistanceIndex



def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Create document relations based on the documentvectors that were created with "createvectors.py"')
    parser.add_argument('-i', '--documentvectorfile', help='The xml file containing the documentvectors', required=True)
    parser.add_argument('-d', '--distance', help='Minimum distance between the files (actual distance times 100)', required=True)
    parser.add_argument('-o', '--output', help='Output file for the xml file with the document relations', required=True)
    args = vars(parser.parse_args())

    # Create the output directory if it doesn't exist
    outputdir = os.path.dirname(args["output"])
    os.makedirs( outputdir, exist_ok=True)

    return (args["documentvectorfile"], int(args["distance"]), args["output"])


# Main part of the script
if __name__ == '__main__':
    (input, distance, output) = read_arguments()

    functions.show_message("Reading document vectors")
    dv = DocumentVectors.read(input)
    distance_index = DistanceIndex( dv)

    functions.show_message("Building index")
    distance_index.build()

    functions.show_message("Calculating distances")
    relations = distance_index.calculate_relations( (float(distance) / 100.0))
    relations.save( output)


    a = 0

