#!/bin/zsh

COPRUSDIR=/Users/jstegink/thesis/corpora
CURRENT=`pwd`
VENVDIR=/Users/jstegink/Dropbox/John/Studie/OU/Afstuderen/Thesis/Code/LHA/py/bin
OUTDIR=/Users/jstegink/thesis/output/vectors

source $VENVDIR/activate
cd $CURRENT

# Wikidata
$VENVDIR/python createvectors.py -c $COPRUSDIR/wikidata/nl -o $OUTDIR/wikidata/nl -a "word2vec"
$VENVDIR/python createvectors.py -c $COPRUSDIR/wikidata/en -o $OUTDIR/wikidata/en -a "sent2vec"

# Wikimatch
$VENVDIR/python createvectors.py -c $COPRUSDIR/wikimatch/nl -o $OUTDIR/wikimatch/nl -a "word2vec"
$VENVDIR/python createvectors.py -c $COPRUSDIR/wikimatch/en -o $OUTDIR/wikimatch/en -a "sent2vec"

# S2ORC
$VENVDIR/python createvectors.py -c $COPRUSDIR/S2ORC/history -o $OUTDIR/S2ORC/history -a "sent2vec"
