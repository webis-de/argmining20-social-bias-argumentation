#!/bin/bash

# ConvinceMe portal data
echo ""
echo "Preparing posts from Convinceme portal."
python prepare_glove_input_from_iac.py \
    --input "output/iacv2-convinceme-texts.txt" \
    --output "output/glove_format" \
    --filename_postfix "_convinceme" \
    --splits 5

# CreateDebate portal data
echo ""
echo "Preparing posts from CreateDebate portal."
python prepare_glove_input_from_iac.py \
    --input "output/iacv2-createdebate-texts.txt" \
    --output "output/glove_format" \
    --filename_postfix "_createdebate" \
    --splits 5

# 4Forums portal data
echo ""
echo "Preparing posts from 4Forums portal."
python prepare_glove_input_from_iac.py \
    --input "output/iacv2-fourforums-texts.txt" \
    --output "output/glove_format" \
    --filename_postfix "_fourforums" \
    --splits 5

# Combined portal data
echo ""
echo "Preparing posts from all portals combined."
python prepare_glove_input_from_iac.py \
    --input "output/iacv2-combined-texts.txt" \
    --output "output/glove_format" \
    --filename_postfix "_combined" \
    --splits 5
