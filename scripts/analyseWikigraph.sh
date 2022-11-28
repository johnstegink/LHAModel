#!/bin/zsh -m

CURRENT=`pwd`
VENVDIR=/Users/jstegink/Dropbox/John/Studie/OU/Afstuderen/Thesis/Code/LHA/py/bin

IMGDIR=/Users/jstegink/Dropbox/John/Studie/OU/Afstuderen/Thesis/Document/grafieken/distance-gwikimatch
WIKIMATCHDIR=/Users/jstegink/Dropbox/John/Studie/OU/Afstuderen/Thesis/Corpora/code/WikiDataCorpus/GWikiMatch

source $VENVDIR/activate
cd $CURRENT || exit

MAXDEGREE=2000

#$VENVDIR/python createWikiGraph.py -l nl -d $MAXDEGREE
$VENVDIR/python analyseWikigraph.py -d $MAXDEGREE -i $WIKIMATCHDIR -o $IMGDIR

wait


/Users/jstegink/Dropbox/John/Studie/OU/Afstuderen/Thesis/Code/Model
