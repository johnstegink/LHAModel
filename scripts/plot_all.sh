#!/bin/zsh -m

METRICSDIR=/Volumes/Extern/Studie/studie/metrics
ITERATION=1
LATEXDIR=/Users/jstegink/Dropbox/John/Studie/OU/Afstuderen/Thesis/Document/Latex/images



VENVDIR=/Users/jstegink/Dropbox/John/Studie/OU/Afstuderen/Thesis/Code/LHA/py/bin
for K in {1..15..2}
do
  # Wikimatch
  PREFIX=wikimatch_nl
  $VENVDIR/python plot.py -m $METRICSDIR/$PREFIX\.tsv -o $LATEXDIR -i $ITERATION  -k $K

  PREFIX=wikimatch_en
  $VENVDIR/python plot.py -m $METRICSDIR/$PREFIX\.tsv -o $LATEXDIR -i $ITERATION  -k $K

  # Wikidata
  PREFIX=wikidata_nl
  $VENVDIR/python plot.py -m $METRICSDIR/$PREFIX\.tsv -o $LATEXDIR -i $ITERATION  -k $K

  PREFIX=wikidata_en
  $VENVDIR/python plot.py -m $METRICSDIR/$PREFIX\.tsv -o $LATEXDIR -i $ITERATION  -k $K

done