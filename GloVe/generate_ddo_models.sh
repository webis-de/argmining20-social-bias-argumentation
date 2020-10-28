#!/bin/bash
set -e

source generate_glove_model.sh

# Generate models for main corpora
declare -A all_corpora=(\
    ["debate_org"]="debate_org-all_posts--glove-format" \
    ["debate_org-female"]="debate_org-gender-female-posts--glove-format" \
    ["debate_org-male"]="debate_org-gender-male-posts--glove-format" \
    ["debate_org-african-american"]="debate_org-ethnicity-african-american-posts--glove-format" \
    ["debate_org-european-american"]="debate_org-ethnicity-european-american-posts--glove-format" \
    ["debate_org-22-below"]="debate_org-age_groups-22-below-posts--glove-format" \
    ["debate_org-23-up"]="debate_org-age_groups-23-up-posts--glove-format")


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
