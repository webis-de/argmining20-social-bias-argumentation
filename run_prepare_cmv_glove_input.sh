#!/bin/bash

python prepare_glove_input_from_cmv.py \
    --input "output/webis-cmv-20-texts_only.txt" \
    --output "output/glove_format" \
    --splits 5
