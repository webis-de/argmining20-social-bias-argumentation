#!/bin/bash
set -e

source generate_glove_model.sh

# Generate models for main corpora
declare -A all_corpora=(\
    ["iac_combined"]="iac_posts_combined--glove-format" \
    ["iac_convinceme"]="iac_posts_convinceme--glove-format" \
    ["iac_createdebate"]="iac_posts_createdebate--glove-format" \
    ["iac_fourforums"]="iac_posts_fourforums--glove-format")


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
