#!/bin/zsh

COPRUSDIR=/Volumes/Extern/Studie/studie/corpora
CURRENT=`pwd`
VENVDIR=/Users/jstegink/Dropbox/John/Studie/OU/Afstuderen/Thesis/Code/LHA/py/bin
OUTDIR=/Volumes/Extern/Studie/studie/vectors

source $VENVDIR/activate
cd $CURRENT



# Wikidata
#$VENVDIR/python createvectors.py -c $COPRUSDIR/wikimatch_nl -o $OUTDIR/wikimatch_nl.xml -a "word2vec"
#$VENVDIR/python createvectors.py -c $COPRUSDIR/gwikimatch_en -o $OUTDIR/gwikimatch_en_word2vec_300.xml -a "word2vec"

# WikiSim
#$VENVDIR/python createvectors.py -c $COPRUSDIR/WikiSim_nl -o $OUTDIR/WikiSim_nl.xml -a "word2vec"
$VENVDIR/python createvectors.py -c $COPRUSDIR/WikiSim_en -o $OUTDIR/WikiSim_en_word2vec_300.xml -a "word2vec"
$VENVDIR/python createvectors.py -c $COPRUSDIR/WikiSim_en -o $OUTDIR/WikiSim_en_use_512.xml -a "use"

# WiRe
#$VENVDIR/python createvectors.py -c $COPRUSDIR/WiRe_nl -o $OUTDIR/WiRe_nl.xml -a "word2vec"
$VENVDIR/python createvectors.py -c $COPRUSDIR/WiRe_en -o $OUTDIR/WiRe_en_word2vec_300.xml -a "word2vec"
$VENVDIR/python createvectors.py -c $COPRUSDIR/WiRe_en -o $OUTDIR/WiRe_en_use_512.xml -a "use"

# Wikimatch
#$VENVDIR/python createvectors.py -c $COPRUSDIR/gwikimatch_nl -o $OUTDIR/gwikimatch_nl.xml -a "word2vec"
#$VENVDIR/python createvectors.py -c $COPRUSDIR/gwikimatch_en -o $OUTDIR/gwikimatch_en.xml -a "sent2vec"
#
## S2ORC
#$VENVDIR/python createvectors.py -c $COPRUSDIR/S2ORC -o $OUTDIR/S2ORC.xml -a "sent2vec"
