#!/usr/bin/env bash

# Remove Watermark from svgs and convert to pdf


text="Visual Paradigm Standard(Rasmus(University of Osnabrueck))"

gsed -i "s/$text//g" *.svg

for f in *.svg
do
    cairosvg -o "${f%%.svg}.pdf" "$f" &
done
wait
