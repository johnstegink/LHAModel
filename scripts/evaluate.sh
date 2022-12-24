#!/bin/zsh -m

CURRENT=`pwd`
VENVDIR=/Users/jstegink/Dropbox/John/Studie/OU/Afstuderen/Thesis/Code/LHA/py/bin
MAXDOC=30


source $VENVDIR/activate
cd $CURRENT

CORPUS=$1
VECTORFILE=$2
RELATIONSFILE=$3
METRICSFILE=$4
HTMLDIR=$5
SIM=$6


if [[ ! ( -f $RELATIONSFILE ) ]] ; then
  $VENVDIR/python createrelations.py -c $CORPUS -i $VECTORFILE -s $SIM -o $RELATIONSFILE -m $MAXDOC
fi

#echo $HTMLDIR

# Create HTML files
#$VENVDIR/python corpus2html.py -c $CORPUS -r $RELATIONSFILE -o $HTMLDIR
$VENVDIR/python evaluate.py -c $CORPUS -i $RELATIONSFILE -o $METRICSFILE


wait

