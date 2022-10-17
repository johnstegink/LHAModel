# Script to create relations based on documentvectors

import argparse

import sys
import os
import functions
from Distances.DocumentRelations import DocumentRelations
from texts.corpus import Corpus



def read_arguments():
    """
    Read the arguments from the commandline
    :return:
    """

    parser = argparse.ArgumentParser(description='Create HTML documents of the corpus')
    parser.add_argument('-c', '--corpusdirectory', help='The corpus directory in the Common File Format', required=True)
    parser.add_argument('-o', '--output', help='Output directory for the html files', required=True)
    parser.add_argument('-r', '--relationsxml', help='Xml file containing document relations', required=True)

    args = vars(parser.parse_args())

    # Create the output directory if it doesn't exist
    outputdir = args["output"]
    os.makedirs( outputdir, exist_ok=True)

    corpusdir = args["corpusdirectory"] if "corpusdirectory" in args else None
    if corpusdir is not None and len( functions.read_all_files_from_directory(corpusdir , "xml")) == 0:
        sys.stderr.write(f"Directory '{corpusdir}' doesn't contain any files\n")
        exit( 2)

    return (corpusdir, outputdir, args["relationsxml"])


def create_html( corpus, id, links, color, postfix):
    """
    Create HTML of the document beloging to the corpus
    :param corpus:
    :param id: the documentid
    :param links: list of ids
    :param color: bootstrap color
    :param postfix: postfix of the html file
    :return:
    """

    doc = corpus.getDocument(id)

    sectionhtml = ""
    for section in doc:
        sectionhtml = sectionhtml +\
               "<h3>" + section.get_title() + "</h3>" +\
               "<p>" + section.get_text() + "</p>"

    linkinfo = []
    for linkid in links:
        linkdoc = corpus.getDocument(linkid)
        linkinfo.append( (linkid, linkdoc.get_title()))

    linkinfo.sort( key=lambda x: x[1])
    linkshtml = "<ul>"
    for (linkid, linktitle) in linkinfo:
        linkshtml += f"<li><a href='{linkid}_{postfix}.html'>{linktitle}</a></li>"
    linkshtml += "</ul>"

    html = f"""<!DOCTYPE html>
<html lang="{corpus.get_language_code()}">
  <head>
    <title>{corpus.get_name()} : {doc.get_title()}</title>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.1/dist/css/bootstrap.min.css" rel="stylesheet">
      <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.1/dist/js/bootstrap.bundle.min.js"></script>
  <body>
    <div class="container">
        <div class="container-fluid p-3 bg-{color} text-white text-center">
          <h1>{doc.get_title()}</h1>
        </div>
        <div class="container mt-5">
          <div class="row">
            <div class="col-sm-9">
                {sectionhtml}
            </div>
            <div class="col-sm-3">
                {linkshtml}
            </div>
          </div>
        </div> 
    </div>
  </body>
</html>
    """

    return html


# Main part of the script
if __name__ == '__main__':
    (corpusdir, outdir, relationsxml) = read_arguments()

    functions.show_message("Reading corpus")
    corpus = Corpus(directory=corpusdir)
    functions.show_message(f"The corpus contains {corpus.get_number_of_documents()} documents")

    functions.show_message("Reading relations")
    (relations, attr) = DocumentRelations.read( relationsxml)


    functions.show_message("Generating corpus HTML")
    for doc in corpus:
        links = [ id for (id, sim) in doc.get_links()]
        html = create_html(corpus, doc.get_id(), links, "primary", "corpus")

        functions.write_file( os.path.join( outdir, doc.get_id() + "_corpus.html"), html)

    functions.show_message("Generating relations xml")
    for doc in corpus:
        rels = relations.get_relations_of( doc.get_id())
        links = [rel.get_dest() for rel in relations.get_relations_of( doc.get_id())]
        html = create_html(corpus, doc.get_id(), links, "warning", "relation")
        functions.write_file( os.path.join( outdir, doc.get_id() + "_relation.html"), html)

