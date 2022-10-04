#!/bin/zsh

COPRUSDIR=/Users/jstegink/thesis/corpora
CURRENT=`pwd`
VENVDIR=/Users/jstegink/Dropbox/John/Studie/OU/Afstuderen/Thesis/Code/LHA/py/bin
OUTDIR=/Users/jstegink/thesis/output/vectors

source $VENVDIR/activate
cd $CURRENT

# Wikidata
$VENVDIR/python createvectors.py -c $COPRUSDIR/wikidata/nl -o $OUTDIR/wikidata_nl.xml -a "word2vec"
$VENVDIR/python createvectors.py -c $COPRUSDIR/wikidata/en -o $OUTDIR/wikidata_en.xml -a "sent2vec"

# Wikimatch
$VENVDIR/python createvectors.py -c $COPRUSDIR/wikimatch/nl -o $OUTDIR/wikimatch_nl.xml -a "word2vec"
$VENVDIR/python createvectors.py -c $COPRUSDIR/wikimatch/en -o $OUTDIR/wikimatch_en.xml -a "sent2vec"

# S2ORC
$VENVDIR/python createvectors.py -c $COPRUSDIR/S2ORC -o $OUTDIR/S2ORC.xml -a "sent2vec"
