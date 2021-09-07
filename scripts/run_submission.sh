#!/bin/bash
NUM_FOLDS=5

python main.py --Mode Create_NewSubmission --Backbone Gem4hz+8hz --Stage 2 \
            --N_folds $NUM_FOLDS --Val_batch_size 20 --Val_num_workers 4

python main.py --Mode Create_NewSubmission --Backbone Gem8hz+8hz --Stage 2 \
            --N_folds $NUM_FOLDS --Val_batch_size 20 --Val_num_workers 4

python main.py  --Mode Create_NewSubmission --Backbone Gem48hz+32hz --Stage 2 \
            --N_folds $NUM_FOLDS --Val_batch_size 20 --Val_num_workers 4

python main.py --Mode CombineNewCsv \