#!/bin/zsh
# tests a single test corpus
# Processes a single file usage: test_file -c <corpus> -l <language>

zmodload zsh/zutil
zparseopts -D -F c:=corpus_val l:=language_val || (echo "Usage: test_file -c <corpus> -l <language>"; exit 1)

BASEDIR=/Volumes/Extern/Studie/studie
CURRENT=`pwd`
VENVDIR=/Users/jstegink/Dropbox/John/Studie/OU/Afstuderen/Thesis/Code/LHA/py/bin

CORPUS=${corpus_val[-1]}
LANGUAGE=${language_val[-1]}

NN_TYPE="stat"
MODELMETHOD="sbert"

if [ "${LANGUAGE}" = "en" ]; then
  SIM=80
  MAXDOC=50
  NN=10
  NROFSECTIONS=12
  BATCHSIZE=50
  EPOCHS=10
  LEARNINGRATE=0.01
fi

if [ "${LANGUAGE}" = "nl" ]; then
  SIM=80
  MAXDOC=50
  NN=10
  NROFSECTIONS=12
  BATCHSIZE=100
  EPOCHS=200
  LEARNINGRATE=0.001
fi

for METHOD in "sbert" "sent2vec" "use" "word2vec" ; do
    MODELCORPUS="wire_${LANGUAGE}"
    CORPUSDIR="${BASEDIR}/corpora/${CORPUS}"
    MODELSDIR="${BASEDIR}/models"
    VECTORSDIR="${BASEDIR}/vectors/${CORPUS}_${METHOD}.xml"
    SCRATCHDIR="${BASEDIR}/scratch/${CORPUS}_${METHOD}_${SIM}_${MAXDOC}_${NN}"
    RELATIONSFILE="${BASEDIR}/relations/${CORPUS}_${METHOD}_${SIM}_${MAXDOC}_pairsonly.xml"

    source $VENVDIR/activate
    cd $CURRENT

    MODELSFILE="${MODELSDIR}/${MODELCORPUS}_${MODELMETHOD}_${SIM}_${MAXDOC}_${NN}_pairsonly_${NROFSECTIONS}_${BATCHSIZE}_${EPOCHS}_${LEARNINGRATE}.pt"
    $VENVDIR/python testModel.py -N $NROFSECTIONS -c $CORPUSDIR -mo $MODELSFILE -nn $NN_TYPE -v $VECTORSDIR -r $RELATIONSFILE -t truncate -s $SCRATCHDIR
done

