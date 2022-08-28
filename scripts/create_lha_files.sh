#!/bin/zsh

COPRUSDIR=/Users/jstegink/thesis/corpora/testset
CURRENT=`pwd`
VENVDIR=/Users/jstegink/Dropbox/John/Studie/OU/Afstuderen/Thesis/Code/LHA/py/bin
OUTDIR=/Users/jstegink/thesis/output/simple

source $VENVDIR/activate
cd $CURRENT

$VENVDIR/python create_lha_files.py -c $COPRUSDIR/simple -o $OUTDIR/simple
$VENVDIR/python create_lha_files.py -c $COPRUSDIR/en -o $OUTDIR/en


