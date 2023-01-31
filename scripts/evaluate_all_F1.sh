#!/bin/zsh -m

CORPUSDIR=/Users/jstegink/thesis/corpora
RELATIONSDIR=/Volumes/Extern/Studie/studie/relations
METRICSDIR=/Volumes/Extern/Studie/studie/metrics
LATEXDIR=/Users/jstegink/Dropbox/John/Studie/OU/Afstuderen/Thesis/Document/Latex/images
ITERATION=2


rm $METRICSDIR/*

# Wikimatch
PREFIX=wikimatch_nl
$VENVDIR/python evaluate_F1.py -c $CORPUSDIR/wikimatch/nl -i $RELATIONSDIR/$PREFIX\.xml -o $METRICSDIR/$PREFIX\.tsv -l $LATEXDIR -t $ITERATION
PREFIX=wikimatch_en
$VENVDIR/python evaluate_F1.py -c $CORPUSDIR/wikimatch/en -i $RELATIONSDIR/$PREFIX\.xml -o $METRICSDIR/$PREFIX\.tsv -l $LATEXDIR -t $ITERATION

# WikiSim
PREFIX=WikiSim_nl
$VENVDIR/python evaluate_F1.py -c $CORPUSDIR/WikiSim/nl -i $RELATIONSDIR/$PREFIX\.xml -o $METRICSDIR/$PREFIX\.tsv -l $LATEXDIR -t $ITERATION
PREFIX=WikiSim_en
$VENVDIR/python evaluate_F1.py -c $CORPUSDIR/WikiSim/en -i $RELATIONSDIR/$PREFIX\.xml -o $METRICSDIR/$PREFIX\.tsv -l $LATEXDIR -t $ITERATION

# WiRe
PREFIX=WiRe_nl
$VENVDIR/python evaluate_F1.py -c $CORPUSDIR/WiRe/nl -i $RELATIONSDIR/$PREFIX\.xml -o $METRICSDIR/$PREFIX\.tsv -l $LATEXDIR -t $ITERATION
PREFIX=WiRe_en
$VENVDIR/python evaluate_F1.py -c $CORPUSDIR/WiRe/en -i $RELATIONSDIR/$PREFIX\.xml -o $METRICSDIR/$PREFIX\.tsv -l $LATEXDIR -t $ITERATION

