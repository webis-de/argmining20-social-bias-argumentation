#! /bin/bash

all_corpora=(\
    "debate_org-all_posts--glove-format.txt" \
    "iac_posts_combined--glove-format.txt" \
    "iac_posts_createdebate--glove-format.txt" \
    "iac_posts_convinceme--glove-format.txt" \
    "iac_posts_fourforums--glove-format.txt" \
    "debate_org-ethnicity-african-american-posts--glove-format.txt" \
    "debate_org-ethnicity-european-american-posts--glove-format.txt" \
    "debate_org-gender-female-posts--glove-format.txt" \
    "debate_org-gender-male-posts--glove-format.txt" \
    "debate_org-age_groups-22-below-posts--glove-format.txt" \
    "debate_org-age_groups-23-up-posts--glove-format.txt" \
    "cmv-text_only--glove-format.txt")

for i in ${!all_corpora[*]}
do
    echo ""
    echo -e "\e[1;31m================================================================================\e[0m"
    echo -e "\e[1;31mWorking on ${all_corpora[$i]}...\e[0m"

    python weat_cooccurrence_analysis.py \
        --data "output/glove_format/${all_corpora[$i]}" \
        --output "output/weat_cooccurrence_analysis" \
        --processing_cores 7 \
        --tests 1 2 3 4 5 6 7 8 9 10
done
