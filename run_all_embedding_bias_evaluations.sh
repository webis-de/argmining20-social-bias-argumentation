#! /bin/bash

# Evaluate baselines
python embedding_bias_evaluation.py \
    --embedding_model "word_vectors/glove.840B.300d.txt" \
    --output "output/embedding_model_evaluation"

python embedding_bias_evaluation.py \
    --embedding_model "word_vectors/numberbatch-en.txt" \
    --output "output/embedding_model_evaluation" \
    --lowercase


# Evaluate custom corpus models
for MODEL in ./output/glove/*-vectors.txt
do
    python embedding_bias_evaluation.py \
        --embedding_model $MODEL \
        --output "output/embedding_model_evaluation" \
        --lowercase
done

# Evaluate split-corpus models
for MODEL in ./output/glove/splits/*-vectors.txt
do
    python embedding_bias_evaluation.py \
        --embedding_model $MODEL \
        --output "output/embedding_model_evaluation" \
        --lowercase
done
