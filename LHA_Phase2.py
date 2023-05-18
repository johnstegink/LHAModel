# Script to create relations the sections

import argparse
import heapq
import html

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
    parser.add_argument('-d', '--html', help='Output directory for readable html output (debug)', required=False)
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

    return [(section_index_to_id(index),vector) for (index, vector) in sections]


def section_index_to_id( index):
    """
    Returns the id as a string that belongs the the section with the given index
    :param index:
    :return:
    """

    return f"S{index + 1:0=3}"



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
    for id in data.keys():
        nearest = heapq.nlargest( K, data[id], key=lambda x: x[0])  # Nearest N neighbours
        filtered = [val for val in nearest if val[1] >= min_sim] # filter on minimal similarity
        for relation in filtered:
            relations.add_section( relation[0], id, relation[1])



def create_htmls( dsr, corpus,  outputdir):

    # Copy the javascript files
    functions.copy_dir( os.path.join("assets", "lha_phase2"), os.path.join(outputdir, "assets"))

    for src_id in dsr.relations.keys():
        src = corpus.getDocument(src_id)
        for dest_rel in dsr.relations[src_id]:
            dest = corpus.getDocument( dest_rel.get_dest())
            html_output_name = os.path.join( outputdir, f"{src_id}_{dest_rel.get_dest()}.html")
            with open(html_output_name, "w", encoding="utf-8") as htmlfile:
                htmlfile.write(f"<html>\n<head><meta charset='utf-8' />\n<script src='assets/jquery-3.7.0.min.js'></script><script src='assets/leader-line.min.js'></script>\n")
                htmlfile.write(f"<script src='assets/bootstrap.min.js'></script><script src='assets/preview.js'></script><link rel='stylesheet' href='assets/bootstrap.min.css'></link><link rel='stylesheet' href='assets/preview.css'></link>\n\n")
                write_relations( htmlfile, "src_", "dest_", dest_rel.get_relations())
                htmlfile.write(f"</head><body>\n")

                htmlfile.write(f"<div class='container'>\n")
                htmlfile.write(f"<div class='row'>\n")

                htmlfile.write("<div class='col-sm-5'>\n")
                write_document_html(htmlfile, "src_", src)
                htmlfile.write("</div>\n")
                htmlfile.write("<div class='col-sm-1'>&#160;</div>\n")
                htmlfile.write("<div class='col-sm-5'>\n")
                write_document_html(htmlfile, "dest_", dest)
                htmlfile.write("<div class='col-sm-1'>&#160;</div>\n")
                htmlfile.write("</div>\n")

                htmlfile.write(f"</div></div>")
                htmlfile.write(f"</body>\n</html>")

def write_relations( htmlfile, src_id_prefix, dest_id_prefix, relations):
    """
    Create the javascript that adds all relations
    :param htmlfile:
    :param src_id_prefix:
    :param dest_id_prefix:
    :param relations:
    :return:
    """
    htmlfile.write("<script type='text/javascript'>\n")
    for relation in relations:
        htmlfile.write(f"add_relation('{src_id_prefix}{relation.get_src()}', '{dest_id_prefix}{relation.get_dest()}', {relation.get_similarity()});")
    htmlfile.write("</script>\n")

def write_document_html(htmlfile, id_prefix, document):
    """
    Write the columns with the sections fot this document with the given ID prefix
    :param htmlfile:
    :param id_prefix:
    :param document:
    :return: nothing
    """

    section_index = 0
    htmlfile.write(f"<div class='document'>\n")
    htmlfile.write(f"<h2>{html.escape(document.get_title())}</h2>\n")
    for section in document:
        htmlfile.write(f"<div class='section' id='{id_prefix}{section_index_to_id(section_index)}'>\n")
        htmlfile.write(f"<h4>{html.escape(section.get_title())}</h4>\n")
        htmlfile.write(f"<div>{create_html_text(section.get_text())}</div>\n")
        htmlfile.write("</div>")
        section_index += 1
    htmlfile.write(f"</div>\n")


def create_html_text( text):
    """
    converts the text into paragraphs
    :param text:
    :return: the html
    """

    paras = ""
    for para in text.split("\n"):
        paras += "<p>" + html.escape(para) + "</p>"

    return paras


# Main part of the script
if __name__ == '__main__':
    (corpusdir, documentvectors, documentrelations, similarity, K, output, htmldir) = read_arguments()

    functions.show_message("Reading document vectors")
    dv = DocumentVectors.read(documentvectors)

    functions.show_message("Reading document relations")
    (dr, dr_info) = DocumentRelations.read(documentrelations)

    functions.show_message("Reading corpus")
    corpus = Corpus(directory=corpusdir)
    functions.show_message(f"The corpus contains {corpus.get_number_of_documents()} documents")

    dsr = DocumentSectionRelations()
    # with tqdm(total=dr.count(), desc="Section similarity progess") as progress:
    teller = 0
    for relation in dr:
        src = relation.get_src()
        dest = relation.get_dest()
        sim = relation.get_similarity()
        src_vector = dv.get_documentvector(src)
        dest_vector = dv.get_documentvector(dest)

        relations = dsr.add(src, dest, sim)
        NN = determine_NN( relations, src_vector, dest_vector, float(similarity) / 100.0 , K)
        teller += 1
        if teller > 100:
            break
#            progress.update()

    dsr.save(output)
    if not htmldir is None:
        create_htmls( dsr, corpus, htmldir)

    functions.show_message("Done")

