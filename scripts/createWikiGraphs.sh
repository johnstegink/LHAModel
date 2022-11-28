#!/bin/zsh -m

CURRENT=`pwd`
VENVDIR=/Users/jstegink/Dropbox/John/Studie/OU/Afstuderen/Thesis/Code/LHA/py/bin

source $VENVDIR/activate
cd $CURRENT || exit

MAXDEGREE=2000

#$VENVDIR/python createWikiGraph.py -l nl -d $MAXDEGREE
$VENVDIR/python createWikiGraph.py -l en -d $MAXDEGREE

wait


/Users/jstegink/Dropbox/John/Studie/OU/Afstuderen/Thesis/Code/Model