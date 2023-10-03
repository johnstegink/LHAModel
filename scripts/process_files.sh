#!/bin/zsh
# Processes all files

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
              echo "${corpus} ${language} ${method} ${sim} ${maxdoc} ${nn} ${sections}"
              scripts/process_file.sh -c "${corpus}_${language}" -m $method -s $sim -d $maxdoc -n $nn -t $sections
            done
          done
        done
      done
    done
  done
done