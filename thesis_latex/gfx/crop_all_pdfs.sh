#!/usr/bin/env bash
if [ -z "$1" ]
then
    echo "Using current directory."
    DIR='.'
else
    DIR="$1"
fi

find "$DIR" -type f -name "*.pdf" | parallel pdfcrop {} {}
