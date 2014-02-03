#!/bin/bash

# This is one _ugly_ hack, but I couldn't figure out how
# to cleanly pass command line options to the script if
# invoking using the "gimp --batch < script.py" syntax

# Basically I create a temp file, put the args into it,
# then the script gets the filename and reads back the
# args

export PIPELINE_ARGS_TMPFILE=`mktemp`

for arg in "$@"
do
	echo $arg >> $PIPELINE_ARGS_TMPFILE
done

gimp -i --batch-interpreter python-fu-eval --batch - < ../data_generation/pipeline/pipeline.py


