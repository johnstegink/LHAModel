#!/bin/zsh

COPRUSDIR=/Users/jstegink/thesis/corpora
CURRENT=`pwd`
VENVDIR=/Users/jstegink/Dropbox/John/Studie/OU/Afstuderen/Thesis/Code/LHA/py/bin
OUTDIR=/Users/jstegink/thesis/output/vectors

source $VENVDIR/activate
cd $CURRENT

# Wikidata
$VENVDIR/python createvectors.py -c $COPRUSDIR/wikimatch/nl -o $OUTDIR/wikimatch_nl.xml -a "word2vec"
$VENVDIR/python createvectors.py -c $COPRUSDIR/wikimatch/en -o $OUTDIR/wikimatch_en.xml -a "sent2vec"

# WiRe
$VENVDIR/python createvectors.py -c $COPRUSDIR/WikiSim/nl -o $OUTDIR/WikiSim_nl.xml -a "word2vec"
$VENVDIR/python createvectors.py -c $COPRUSDIR/WikiSim/en -o $OUTDIR/WikiSim_en.xml -a "sent2vec"

# WikiSim
$VENVDIR/python createvectors.py -c $COPRUSDIR/WiRe/nl -o $OUTDIR/WiRe_nl.xml -a "word2vec"
$VENVDIR/python createvectors.py -c $COPRUSDIR/WiRe/en -o $OUTDIR/WiRe_en.xml -a "sent2vec"

# Wikimatch
#$VENVDIR/python createvectors.py -c $COPRUSDIR/wikimatch/nl -o $OUTDIR/wikimatch_nl.xml -a "word2vec"
#$VENVDIR/python createvectors.py -c $COPRUSDIR/wikimatch/en -o $OUTDIR/wikimatch_en.xml -a "sent2vec"
#
## S2ORC
#$VENVDIR/python createvectors.py -c $COPRUSDIR/S2ORC -o $OUTDIR/S2ORC.xml -a "sent2vec"
