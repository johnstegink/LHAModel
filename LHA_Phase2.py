# Script to create relations the sections

import argparse
import heapq

import sys
import os

from tqdm import tqdm

import functions
from Distances.DocumentRelations import DocumentRelations
from Distances.DocumentVectors import  DocumentVectors
from Distances.SimilarityMatrix import  SimilarityMatrix
from Distances.DocumentSectionRelations import  DocumentSectionRelations
from texts.corpus import Corpus



def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Create document relations between the sections using the files created with "createvectors.py" and "createrelations.py"')
    parser.add_argument('-c', '--corpusdirectory', help='The corpus directory in the Common File Format', required=True)
    parser.add_argument('-i', '--documentvectorfile', help='The xml file containing the documentvectors', required=True)
    parser.add_argument('-r', '--relationsfiles', help='The xml file containing the relations between the documents', required=True)
    parser.add_argument('-s', '--similarity', help='Minimum similarity used to select the sections (actual similarity times 100)', required=True)
    parser.add_argument('-k', '--nearestneighbors', help='The maximum number of nearest neighbours to find (K)')
    parser.add_argument('-o', '--output', help='Output file for the xml file with the section relations', required=True)
    parser.add_argument('-d', '--html', help='Output file for readable html output (debug)', required=False)
    args = vars(parser.parse_args())

    # Create the output directory if it doesn't exist
    outputdir = os.path.dirname(args["output"])
    os.makedirs( outputdir, exist_ok=True)

    corpusdir = args["corpusdirectory"] if "corpusdirectory" in args else None
    if corpusdir is not None and len( functions.read_all_files_from_directory(corpusdir , "xml")) == 0:
        sys.stderr.write(f"Directory '{corpusdir}' doesn't contain any files\n")
        exit( 2)

    return (corpusdir, args["documentvectorfile"], args["relationsfiles"], int(args["similarity"]), int(args["nearestneighbors"]), args["output"], args["html"])



def section_to_dict( sections):
    """
    Translates the section from the document vector to a list of tuples with the name "00x" and the vector
    that can be used for the similarity matrix
    :param section: sections of a documentvector
    :return: dictionary<name, vector>
    """

    return [(f"{index + 1:0=3}",vector) for (index, vector) in sections]


def determine_NN( relations, src, dest,  min_sim, K):
    """
    Determine the Nearest Neighbours of the sections in src en dest as described in
    "Large-scale Hierarchical Alignment for Data-driven Text Rewriting", Nikola I. Nikolov et al. section 3.2

    :param relations: the sectionrelations object to add these vectors to
    :param dest: destination vectors
    :param src: source vectors
    :param min_sim: minimal similarity
    :param K: the number of nearest neighbours
    :return:
    """

    src_sections = section_to_dict(src.get_sections())
    dst_sections = section_to_dict(dest.get_sections())

    similarity = SimilarityMatrix( src_sections, dst_sections)
    data = similarity.get_values()
    for src in data.keys():
        nearest = heapq.nlargest( K, data[src], key=lambda x: x[0])  # Nearest N neighbours
        filtered = [val for val in nearest if val[1] >= min_sim] # filter on minimal similarity
        for relation in filtered:
            relations.add( relation[0], relation[1])



# Main part of the script
if __name__ == '__main__':
    (corpusdir, documentvectors, documentrelations, similarity, K, output, html) = read_arguments()

    functions.show_message("Reading document vectors")
    dv = DocumentVectors.read(documentvectors)

    functions.show_message("Reading document relations")
    (dr, dr_info) = DocumentRelations.read(documentrelations)

    functions.show_message("Reading corpus")
    corpus = Corpus(directory=corpusdir)
    functions.show_message(f"The corpus contains {corpus.get_number_of_documents()} documents")

    dsr = DocumentSectionRelations()
    with tqdm(total=dr.count(), desc="Section similarity progess") as progress:
        for relation in dr:
            src = relation.get_src()
            dest = relation.get_dest()
            sim = relation.get_similarity()
            src_vector = dv.get_documentvector(src)
            dest_vector = dv.get_documentvector(dest)

            relations = dsr.add(src, dest, similarity)
            NN = determine_NN( relations, src_vector, dest_vector, similarity, K)

            progress.update()

    dsr.save(output)
    functions.show_message("Done")

