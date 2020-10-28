#! /bin/bash

# Full corpus
python prepare_glove_input_from_ddo.py \
    --input "output/debates_data-prepared.csv" \
    --output "output/glove_format" \
    --splits 5

# Sub-corpora based on groups of interest definitions
python prepare_glove_input_from_ddo.py \
    --input "output/debates_data-prepared.csv" \
    --output "output/glove_format" \
    --groups "data/groups_of_interest.json" \
    --splits 5
