#!/bin/zsh
# Processes all files

SESSIONS_DIR="/Volumes/Extern/Studie/studie/sessions"

CURRENT=`pwd`
VENVDIR=/Users/jstegink/Dropbox/John/Studie/OU/Afstuderen/Thesis/Code/LHA/py/bin

source $VENVDIR/activate
cd $CURRENT

mkdir -p $SESSIONS_DIR

for language in en nl
do
  for corpus in wikisim wire gWikimatch
  do
    for method in word2vec sent2vec sbert use
    do
      for sim in 30 50 70 80
      do
        for maxdoc in 20 30 40 50
        do
          for nn in 10 20 30
          do
            for sections in 9 12
            do
              for nn_type in plain stat
              do
                combined="${language}_${corpus}_${method}_${sim}_${maxdoc}_${nn}_${sections}_${nn_type}"

                # make sure the same is not run twice
                SESSION_FILE="${SESSIONS_DIR}/${combined}.ses"
                if [ ! -f $SESSION_FILE ]; then
                  echo $combined

                  scripts/process_file.sh -c "${corpus}_${language}" -m $method -s $sim -d $maxdoc -n $nn -t $sections -r $nn_type
                  if [ $? = 0 ]; then
                    date "+%d-%m-%y %H:%M:%S" > $SESSION_FILE
                  else
                    echo "Error during excecution ${?}"
                  fi
                else
                  echo "${combined} has already been generated"
                fi
              done
            done
          done
        done
      done
    done
  done
done

py
