#!/bin/bash
set -e

source generate_glove_model.sh

# Generate models for main corpora
declare -A all_corpora=(\
    ["webis_cmv"]="cmv-text_only--glove-format")


for i in ${!all_corpora[@]}
do
    # Generate glove model for the current corpus
    generate_glove_model \
        "../output/glove_format/${all_corpora[$i]}.txt" \
        "../output/glove" \
        $i

    # Generate glove models for all corpus splits
    for j in $(seq 0 4)
    do
        generate_glove_model \
            "../output/glove_format/splits/${all_corpora[$i]}__split$j.txt" \
            "../output/glove/splits" \
            "${i}__split$j"
    done
done
