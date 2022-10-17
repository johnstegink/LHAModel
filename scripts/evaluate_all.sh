#!/bin/zsh -m

CORPUSDIR=/Users/jstegink/thesis/corpora
CURRENT=`pwd`
RELATIONSDIR=/Users/jstegink/thesis/output/relations
VECTORDIR=/Users/jstegink/thesis/output/vectors
METRICSDIR=/Users/jstegink/thesis/output/metrics
HTMLDIR=/Users/jstegink/thesis/output/html

rm $METRICSDIR/*

for SIM in {50..99..5}
do
  # Wikidata
  PREFIX=wikidata_nl
  $CURRENT/scripts/evaluate.sh $CORPUSDIR/wikidata/nl $VECTORDIR/$PREFIX\.xml $RELATIONSDIR/$SIM/$PREFIX\.xml $METRICSDIR/$PREFIX\.tsv $HTMLDIR/$PREFIX/$SIM $SIM
  PREFIX=wikidata_en
  $CURRENT/scripts/evaluate.sh $CORPUSDIR/wikidata/en $VECTORDIR/$PREFIX\.xml $RELATIONSDIR/$SIM/$PREFIX\.xml $METRICSDIR/$PREFIX\.tsv $HTMLDIR/$PREFIX/$SIM $SIM

  # Wikimatch
#  PREFIX=wikimatch_nl
#  $CURRENT/scripts/evaluate.sh $CORPUSDIR/wikimatch/nl $VECTORDIR/$PREFIX\.xml $RELATIONSDIR/$SIM/$PREFIX\.xml $METRICSDIR/$PREFIX\.tsv $HTMLDIR/$PREFIX/$SIM $SIM &
#  PREFIX=wikimatch_en
#  $CURRENT/scripts/evaluate.sh $CORPUSDIR/wikimatch/en $VECTORDIR/$PREFIX\.xml $RELATIONSDIR/$SIM/$PREFIX\.xml $METRICSDIR/$PREFIX\.tsv $HTMLDIR/$PREFIX/$SIM $SIM &

  # S2ORC
#  PREFIX=S2ORC
#  $CURRENT/scripts/evaluate.sh $CORPUSDIR/S2ORC/history $VECTORDIR/$PREFIX\.xml $RELATIONSDIR/$SIM/$PREFIX\.xml $METRICSDIR/$PREFIX\.tsv $HTMLDIR/$PREFIX/$SIM $SIM &

  wait
done

