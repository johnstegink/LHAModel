#!/bin/zsh -m

CORPUSDIR=/Users/jstegink/thesis/corpora
CURRENT=`pwd`
RELATIONSDIR=/Volumes/Extern/Studie/studie/relations
VECTORDIR=/Volumes/Extern/Studie/studie/vectors
METRICSDIR=/Volumes/Extern/Studie/studie/metrics
HTMLDIR=/Volumes/Extern/Studie/studie/html
ITERATION=1

rm $METRICSDIR/*

  # Wikimatch
#  PREFIX=wikimatch_nl
#  $CURRENT/scripts/evaluate.sh $CORPUSDIR/wikimatch/nl $VECTORDIR/$PREFIX\.xml $RELATIONSDIR/$SIM/$PREFIX\.xml $METRICSDIR/$PREFIX\.tsv $HTMLDIR/$PREFIX/$SIM $SIM $ITERATION
  PREFIX=wikimatch_en
  $CURRENT/scripts/evaluate.sh $CORPUSDIR/wikimatch/en $VECTORDIR/$PREFIX\.xml $RELATIONSDIR/$SIM/$PREFIX\.xml $METRICSDIR/$PREFIX\.tsv $HTMLDIR/$PREFIX/$SIM $SIM $ITERATION

  # WikiSim
#  PREFIX=WikiSim_nl
#  $CURRENT/scripts/evaluate.sh $CORPUSDIR/WikiSim/nl $VECTORDIR/$PREFIX\.xml $RELATIONSDIR/$SIM/$PREFIX\.xml $METRICSDIR/$PREFIX\.tsv $HTMLDIR/$PREFIX/$SIM $SIM $ITERATION
#  PREFIX=WikiSim_en
#  $CURRENT/scripts/evaluate.sh $CORPUSDIR/WikiSim/en $VECTORDIR/$PREFIX\.xml $RELATIONSDIR/$SIM/$PREFIX\.xml $METRICSDIR/$PREFIX\.tsv $HTMLDIR/$PREFIX/$SIM $SIM $ITERATION

  # WiRe
#  PREFIX=WiRe_nl
#  $CURRENT/scripts/evaluate.sh $CORPUSDIR/WiRe/nl $VECTORDIR/$PREFIX\.xml $RELATIONSDIR/$SIM/$PREFIX\.xml $METRICSDIR/$PREFIX\.tsv $HTMLDIR/$PREFIX/$SIM $SIM $ITERATION
#  PREFIX=WiRe_en
#  $CURRENT/scripts/evaluate.sh $CORPUSDIR/WiRe/en $VECTORDIR/$PREFIX\.xml $RELATIONSDIR/$SIM/$PREFIX\.xml $METRICSDIR/$PREFIX\.tsv $HTMLDIR/$PREFIX/$SIM $SIM $ITERATION

  # S2ORC
#  PREFIX=S2ORC
#  $CURRENT/scripts/evaluate.sh $CORPUSDIR/S2ORC/history $VECTORDIR/$PREFIX\.xml $RELATIONSDIR/$SIM/$PREFIX\.xml $METRICSDIR/$PREFIX\.tsv $HTMLDIR/$PREFIX/$SIM $SIM &

  wait

