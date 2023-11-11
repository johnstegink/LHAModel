#!/bin/zsh
# Processes a single file usage: process_file -c <corpus> -m <encoding method> [-s <Sim>] [-d <MaxDoc>] [-n NearestNeighbours] [-t <NrOfSecions>] [-r <NN_type>]

BASEDIR=/Volumes/Extern/Studie/studie/tijdelijk
HTMLDIR="${BASEDIR}/html"
NROFSECTIONS=12
NEARESTNEIGHBORS=10
CURRENT=`pwd`
VENVDIR=/Users/jstegink/Dropbox/John/Studie/OU/Afstuderen/Thesis/Code/LHA/py/bin
RESULTSDIR="${BASEDIR}/results"
FINAL_DIR="${BASEDIR}/final"
FINAL_FILE="together.xlsx"


mkdir -p $FINAL_DIR
mkdir -p $RESULTSDIR

source $VENVDIR/activate
cd $CURRENT

zmodload zsh/zutil
zparseopts -D -F c:=corpus_val m:=method_val s:=sim_val d:=maxdoc_val n:=nn_val t:=nrofsections_val r:=nn_type_val || (echo "Usage: process_file -c <corpus> -m <encoding method> [-s <Sim>] [-d <MaxDoc>] [-n NearestNeighbours] [-t <NrOfSecions>]"; exit 1)

CORPUS=${corpus_val[-1]}
METHOD=${method_val[-1]}
SIM=${sim_val[-1]}
MAXDOC=${maxdoc_val[-1]}
NEARESTNEIGHBORS=${nn_val[-1]}
NROFSECTIONS=${nrofsections_val[-1]}
NN_TYPE=${nn_type_val[-1]}

TRAPERR() {
  echo "Error during execution of script";
  exit $?;
}

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


if [ "${NN_TYPE}" != "stat" ]  && [ "${NN_TYPE}" != "plain" ]; then
  echo "Please use: stat, plain for the neural network type"
  exit 3
fi



# create the vectors if they do not exist
VECTORFILE="${BASEDIR}/vectors/${CORPUS}_${METHOD}.xml"
echo "$VENVDIR/python createvectors.py -c $CORPUSDIR -o "$VECTORFILE" -a $METHOD"
echo "..."

if [ ! -f "$VECTORFILE" ]; then
  $VENVDIR/python createvectors.py -c $CORPUSDIR -o "$VECTORFILE" -a $METHOD
else
  echo "  -- Vector file:  ${VECTORFILE} already exists"
fi


# create the relations if they do not exist
RELATIONSFILE="${BASEDIR}/relations/${CORPUS}_${METHOD}_${SIM}_${MAXDOC}_pairsonly.xml"
echo "$VENVDIR/python createrelations.py -c $CORPUSDIR -i $VECTORFILE -s $SIM -o $RELATIONSFILE -m $MAXDOC -p True"
echo "..."

if [ ! -f "$RELATIONSFILE" ]; then
  $VENVDIR/python createrelations.py -c $CORPUSDIR -i $VECTORFILE -s $SIM -o $RELATIONSFILE -m $MAXDOC -p True
else
  echo "  -- Relations file: ${RELATIONSFILE} already exists"
fi

# create the section relations if they do not exist
SECTIONSFILE="${BASEDIR}/sections/${CORPUS}_${METHOD}_${SIM}_${MAXDOC}_${NEARESTNEIGHBORS}_pairsonly.xml"
echo "$VENVDIR/python LHA_Phase2.py -c $CORPUSDIR -i $VECTORFILE -r $RELATIONSFILE -s $SIM -o $SECTIONSFILE -k $NEARESTNEIGHBORS  -d $HTMLDIR"
echo "..."

if [ ! -f "$SECTIONSFILE" ]; then
  $VENVDIR/python LHA_Phase2.py -c $CORPUSDIR -i $VECTORFILE -r $RELATIONSFILE -s $SIM -o $SECTIONSFILE -k $NEARESTNEIGHBORS -d $HTMLDIR
else
  echo "  -- Sections file: ${SECTIONSFILE} already exists"
fi

HEATMAPDIR="${BASEDIR}/heatmaps/${CORPUS}_${METHOD}_${SIM}_${MAXDOC}_${NEARESTNEIGHBORS}"
SCRATCHDIR="${BASEDIR}/scratch/${CORPUS}_${METHOD}_${SIM}_${MAXDOC}_${NEARESTNEIGHBORS}"

echo "-N $NROFSECTIONS -c $CORPUSDIR -nn $NN_TYPE -s $SCRATCHDIR -v $VECTORFILE -r $SECTIONSFILE -m $HEATMAPDIR -t truncate -o ${RESULTSDIR}"
echo "..."
$VENVDIR/python trainModel.py -N $NROFSECTIONS -c $CORPUSDIR -nn $NN_TYPE -s $SCRATCHDIR -v $VECTORFILE -r $SECTIONSFILE -m $HEATMAPDIR -t truncate -o $RESULTSDIR

echo "$VENVDIR/python results.py -d $RESULTSDIR -t excel -o ${FINAL_DIR}/${FINAL_FILE}"
echo "..."
#$VENVDIR/python results.py -d $RESULTSDIR -t excel -o "${FINAL_DIR}/${FINAL_FILE}"

exit 0