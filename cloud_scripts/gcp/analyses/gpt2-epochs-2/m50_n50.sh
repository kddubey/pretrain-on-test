#!/bin/bash
# Assume we're already in the analysis dir (from the repo root)
python run.py \
--analysis_name gpt2_epochs_2_m50_n50 \
--accuracies_home_dir accuracies_gpt2_epochs_2 \
--num_train 50 \
--num_test 50 \
--equation "p(num_correct, num_test) ~ method + (1|dataset/pair)" \
--id_vars num_test pair dataset
