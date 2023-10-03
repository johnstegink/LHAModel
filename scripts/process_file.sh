#!/bin/zsh
# Processes a single file usage: process_file -c <corpus> -m <encoding method> [-s <Sim>] [-d <MaxDoc>] [-n NearestNeighbours] [-t <NrOfSecions>]

BASEDIR=/Volumes/Extern/Studie/studie
HTMLDIR="${BASEDIR}/html"
NROFSECTIONS=12
NEARESTNEIGHBORS=10
CURRENT=`pwd`
VENVDIR=/Users/jstegink/Dropbox/John/Studie/OU/Afstuderen/Thesis/Code/LHA/py/bin
RESULTSDIR=/Volumes/Extern/Studie/studie/results

source $VENVDIR/activate
cd $CURRENT

zmodload zsh/zutil
zparseopts -D -F c:=corpus_val m:=method_val s:=sim_val d:=maxdoc_val n:=nn_val t:=nrofsections_val || (echo "Usage: process_file -c <corpus> -m <encoding method> [-s <Sim>] [-d <MaxDoc>] [-n NearestNeighbours] [-t <NrOfSecions>]"; exit 1)

CORPUS=${corpus_val[-1]}
METHOD=${method_val[-1]}
SIM=${sim_val[-1]}
MAXDOC=${maxdoc_val[-1]}
NEARESTNEIGHBORS=${nn_val[-1]}
NROFSECTIONS=${nrofsections_val[-1]}


if [ "$CORPUS" = "" ]; then
  echo "Please provide a corpus"
  exit 1
fi

CORPUSDIR="${BASEDIR}/corpora/${CORPUS}"

if [ ! -d $CORPUSDIR ]; then
  echo "Corpus ${CORPUS} does not exist"
  exit 2
fi


if [ "$METHOD" != "word2vec" ]  && [ "$METHOD" != "sent2vec" ]  && [ "$METHOD" != "use" ] && [ "$METHOD" != "sbert" ]; then
  echo "Please use: word2vec, sent2vec, use or sbert for the encoding method"
  exit 3
fi

if [ "$SIM" = "" ]; then
  SIM=60
  echo "SIM was not specified, a value of ${SIM} is used"
fi

if [ "$MAXDOC" = "" ]; then
  MAXDOC=20
  echo "MaxDoc was not specified, a value of ${MAXDOC} is used"
fi

if [ "$NEARESTNEIGHBORS" = "" ]; then
  NEARESTNEIGHBORS=10
  echo "NearestNeighbours was not specified, a value of ${NEARESTNEIGHBORS} is used"
fi

if [ "$NROFSECTIONS" = "" ]; then
  NROFSECTIONS=12
  echo "NrOfSection was not specified, a value of ${NROFSECTIONS} is used"
fi

echo $SIM
echo $MAXDOC



# create the vectors if they do not exist
VECTORFILE="${BASEDIR}/vectors/${CORPUS}_${METHOD}.xml"
echo "$VENVDIR/python createvectors.py -c $CORPUSDIR -o "$VECTORFILE" -a $METHOD"
echo "..."

if [ ! -f "$VECTORFILE" ]; then
  $VENVDIR/python createvectors.py -c $CORPUSDIR -o "$VECTORFILE" -a $METHOD
else
  echo "  -- ${VECTORFILE} already exists"
fi

# create the relations if they do not exist
RELATIONSFILE="${BASEDIR}/relations/${CORPUS}_${METHOD}_${SIM}_${MAXDOC}.xml"
echo "$VENVDIR/python createrelations.py -c $CORPUSDIR -i $VECTORFILE -s $SIM -o $RELATIONSFILE -m $MAXDOC"
echo "..."

if [ ! -f "$RELATIONSFILE" ]; then
  $VENVDIR/python createrelations.py -c $CORPUSDIR -i $VECTORFILE -s $SIM -o $RELATIONSFILE -m $MAXDOC
else
  echo "  -- ${RELATIONSFILE} already exists"
fi

# create the section relations if they do not exist
SECTIONSFILE="${BASEDIR}/sections/${CORPUS}_${METHOD}_${SIM}_${MAXDOC}.xml"
echo "$VENVDIR/python LHA_Phase2.py -c $CORPUSDIR -i $VECTORFILE -r $RELATIONSFILE -s $SIM -o $SECTIONSFILE -k $NEARESTNEIGHBORS # -d $HTMLDIR"
echo "..."

if [ ! -f "$SECTIONSFILE" ]; then
  $VENVDIR/python LHA_Phase2.py -c $CORPUSDIR -i $VECTORFILE -r $RELATIONSFILE -s $SIM -o $SECTIONSFILE -k $NEARESTNEIGHBORS -d $HTMLDIR
else
  echo "  -- ${SECTIONSFILE} already exists"
fi

HEATMAPDIR="${BASEDIR}/heatmaps/${CORPUS}_${METHOD}_${SIM}_${MAXDOC}"
SCRATCHDIR="${BASEDIR}/scratch/${CORPUS}_${METHOD}_${SIM}_${MAXDOC}"

echo "-N $NROFSECTIONS -c $CORPUSDIR -nn stat -s $SCRATCHDIR -v $VECTORFILE -r $SECTIONSFILE -m $HEATMAPDIR -t "truncate" -o ${RESULTSDIR}"
echo "..."
$VENVDIR/python trainModel.py -N $NROFSECTIONS -c $CORPUSDIR -nn stat -s $SCRATCHDIR -v $VECTORFILE -r $SECTIONSFILE -m $HEATMAPDIR -t "truncate" -o $RESULTSDIR