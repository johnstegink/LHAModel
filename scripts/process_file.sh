#!/bin/zsh
# Processes a single file usage: process_file <corpus> <encoding method> [<Sim>] [<MaxDoc>] [NearestNeighbours] [<NrOfSecions>]

BASEDIR=/Volumes/Extern/Studie/studie
HTMLDIR="${BASEDIR}/html"
NROFSECTIONS=12
NEARESTNEIGHBORS=10
CURRENT=`pwd`
VENVDIR=/Users/jstegink/Dropbox/John/Studie/OU/Afstuderen/Thesis/Code/LHA/py/bin
OUTDIR=/Volumes/Extern/Studie/studie/vectors

source $VENVDIR/activate
cd $CURRENT

# Read the arguments
CORPUS=$1
METHOD=$2

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

if [ "$3" = "" ]; then
  SIM=60
  echo "SIM was not specified, a value of ${SIM} is used"
else
  SIM=$3
fi

if [ "$4" = "" ]; then
  MAXDOC=20
  echo "MaxDoc was not specified, a value of ${MAXDOC} is used"
else
  MAXDOC=$4
fi

if [ "$5" = "" ]; then
  NEARESTNEIGHBORS=10
  echo "NearestNeighbours was not specified, a value of ${NEARESTNEIGHBORS} is used"
else
  NEARESTNEIGHBORS=$5
fi

if [ "$6" = "" ]; then
  NROFSECTIONS=12
  echo "NrOfSection was not specified, a value of ${NROFSECTIONS} is used"
else
  NROFSECTIONS=$6
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
echo "$VENVDIR/python LHA_Phase2.py -c $CORPUSDIR -i $VECTORFILE -r $RELATIONSFILE -s $SIM -o $SECTIONSFILE -k $NEARESTNEIGHBORS -d $HTMLDIR"
echo "..."

if [ ! -f "$SECTIONSFILE" ]; then
  $VENVDIR/python LHA_Phase2.py -c $CORPUSDIR -i $VECTORFILE -r $RELATIONSFILE -s $SIM -o $SECTIONSFILE -k $NEARESTNEIGHBORS -d $HTMLDIR
else
  echo "  -- ${SECTIONSFILE} already exists"
fi

HEATMAPDIR="${BASEDIR}/heatmaps/${CORPUS}_${METHOD}_${SIM}_${MAXDOC}"
SCRATCHDIR="${BASEDIR}/scratch/${CORPUS}_${METHOD}_${SIM}_${MAXDOC}"

echo "-N $NROFSECTIONS -c $CORPUSDIR -nn stat -s $SCRATCHDIR -v $VECTORFILE -r $SECTIONSFILE -m $HEATMAPDIR -t "truncate""
echo "..."
$VENVDIR/python trainModel.py -N $NROFSECTIONS -c $CORPUSDIR -nn stat -s $SCRATCHDIR -v $VECTORFILE -r $SECTIONSFILE -m $HEATMAPDIR -t "truncate"
