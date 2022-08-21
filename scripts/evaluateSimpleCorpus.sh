#!/bin/zsh

COPRUSDIR=/Users/jstegink/thesis/corpora/testset
CURRENT=`pwd`
VENVDIR=/Users/jstegink/Dropbox/John/Studie/OU/Afstuderen/Thesis/Code/LHA/py/bin
OUTDIR=/Users/jstegink/thesis/output/simple

source $VENVDIR/activate
cd $CURRENT

#$VENVDIR/python createvectors.py -c $COPRUSDIR/simple -v 600 -o $OUTDIR/simple/documentvectors.xml
#$VENVDIR/python createvectors.py -c $COPRUSDIR/en -v 600 -o $OUTDIR/en/documentvectors.xml

$VENVDIR/python createrelationsfrom2.py -c1 $COPRUSDIR/simple -c2 $COPRUSDIR/en -i1 $OUTDIR/simple/documentvectors.xml -i2 $OUTDIR/en/documentvectors.xml -o $OUTDIR/relations.xml -d 60 -r $OUTDIR/relations.html


