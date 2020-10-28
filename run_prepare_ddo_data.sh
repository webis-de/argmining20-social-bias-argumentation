#! /bin/bash

python prepare_ddo_data.py \
    --debates "data/debates.json" \
    --users "data/users.json" \
    --output "output" \
    --multicore \
    --processing_cores 7
