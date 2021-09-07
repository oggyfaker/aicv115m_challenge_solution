#!/bin/bash
NUM_FOLDS=5

# >>>> 4hz - Stage1 - Stage2 <<<<
for ((i=0; i<$NUM_FOLDS; i++)) do
  python main.py --Mode Train --Backbone Gem4hz --Stage 1 \
            --N_folds $NUM_FOLDS --Fold $i \
            --Train_batch_size 14 --Train_num_workers 3 \
            --Val_batch_size 10 --Val_num_workers 3

  python main.py --Mode Save_Base_Weights --Backbone Gem4hz --Stage 1 \
              --N_folds $NUM_FOLDS --Fold $i

  python main.py --Mode Train --Backbone Gem4hz+8hz --Stage 2 \
              --N_folds $NUM_FOLDS --Fold $i \
              --Train_batch_size 14 --Train_num_workers 3 \
              --Val_batch_size 10 --Val_num_workers 3
done




# >>>> 8hz - Stage1 - Stage2 <<<<
for ((i=0; i<$NUM_FOLDS; i++)) do
  python main.py --Mode Train --Backbone Gem8hz --Stage 1 \
              --N_folds $NUM_FOLDS --Fold $i \
              --Train_batch_size 14 --Train_num_workers 3 \
              --Val_batch_size 10 --Val_num_workers 3

  python main.py --Mode Save_Base_Weights --Backbone Gem8hz --Stage 1 \
              --N_folds $NUM_FOLDS --Fold $i

  python main.py --Mode Train --Backbone Gem8hz+8hz --Stage 2 \
              --N_folds $NUM_FOLDS --Fold $i \
              --Train_batch_size 14 --Train_num_workers 3 \
              --Val_batch_size 10 --Val_num_workers 3
done




# >>>> 48hz - Stage1 - Training and Saving <<<<
for ((i=0; i<$NUM_FOLDS; i++)) do
  python main.py --Mode Train --Backbone Gem48hz --Stage 1 \
                  --N_folds $NUM_FOLDS --Fold $i \
                  --Train_batch_size 7 --Train_num_workers 3 \
                  --Val_batch_size 5 --Val_num_workers 3

  python main.py --Mode Save_Base_Weights --Backbone Gem48hz --Stage 1 \
              --N_folds $NUM_FOLDS --Fold $i

  python main.py --Mode Train --Backbone Gem48hz+32hz --Stage 2 \
              --N_folds $NUM_FOLDS --Fold $i \
              --Train_batch_size 14 --Train_num_workers 3 \
              --Val_batch_size 10 --Val_num_workers 3
done