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
SIM=$5


if [[ ! ( -f $RELATIONSFILE ) ]] ; then
  $VENVDIR/python createrelations.py -c $CORPUS -i $VECTORFILE -s $SIM -o $RELATIONSFILE -m $MAXDOC
fi

$VENVDIR/python evaluate.py -c $CORPUS -i $RELATIONSFILE -s positive -o $METRICSFILE

wait


