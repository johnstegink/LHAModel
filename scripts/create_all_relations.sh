#!/bin/zsh -m

COPRUSDIR=/Users/jstegink/thesis/corpora
CURRENT=`pwd`
VENVDIR=/Users/jstegink/Dropbox/John/Studie/OU/Afstuderen/Thesis/Code/LHA/py/bin
VECTORDIR=/Users/jstegink/thesis/output/vectors
OUTDIR=/Users/jstegink/thesis/output/relations

SIM=80
MAXDOC=30

source $VENVDIR/activate
cd $CURRENT


# Wikimatch
PREFIX=wikimatch_nl
$VENVDIR/python createrelations.py -c $COPRUSDIR/wikimatch/nl -i $VECTORDIR/$PREFIX\.xml -s $SIM -o $OUTDIR/$PREFIX\.xml -m $MAXDOC
PREFIX=wikimatch_en
$VENVDIR/python createrelations.py -c $COPRUSDIR/wikimatch/en -i $VECTORDIR/$PREFIX\.xml -s $SIM -o $OUTDIR/$PREFIX\.xml -m $MAXDOC

# WikiSim
PREFIX=WikiSim_nl
$VENVDIR/python createrelations.py -c $COPRUSDIR/WikiSim/nl -i $VECTORDIR/$PREFIX\.xml -s $SIM -o $OUTDIR/$PREFIX\.xml -m $MAXDOC
PREFIX=WikiSim_en
$VENVDIR/python createrelations.py -c $COPRUSDIR/WikiSim/en -i $VECTORDIR/$PREFIX\.xml -s $SIM -o $OUTDIR/$PREFIX\.xml -m $MAXDOC

# WiRe
PREFIX=WiRe_nl
$VENVDIR/python createrelations.py -c $COPRUSDIR/WiRe/nl -i $VECTORDIR/$PREFIX\.xml -s $SIM -o $OUTDIR/$PREFIX\.xml -m $MAXDOC
PREFIX=WiRe_en
$VENVDIR/python createrelations.py -c $COPRUSDIR/WiRe/en -i $VECTORDIR/$PREFIX\.xml -s $SIM -o $OUTDIR/$PREFIX\.xml -m $MAXDOC

# S2ORC
#PREFIX=S2ORC
#$VENVDIR/python createrelations.py -c $COPRUSDIR/S2ORC/en -i $VECTORDIR/$PREFIX\.xml -s $SIM -o $OUTDIR/$PREFIX\.xml -m $MAXDOC & # -r $OUTDIR/$PREFIX\.html &

wait


