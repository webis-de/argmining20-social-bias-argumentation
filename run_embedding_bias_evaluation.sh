#! /bin/bash

python embedding_bias_evaluation.py \
    --embedding_model "output/glove/debate_org-female-vectors.txt" \
    --output "output/embedding_model_evaluation" \
    --lowercase
