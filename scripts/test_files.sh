#!/bin/zsh
# test all corpora

scripts/test_file.sh -c wikisim_en -l en
scripts/test_file.sh -c wire_en -l en
scripts/test_file.sh -c wikisim_nl -l nl
scripts/test_file.sh -c wire_nl -l nl
