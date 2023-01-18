#!/bin/bash
# run few-shot experiment, fine-tune
# few_shot=(50 100 200 500)
# seeds=(123 234 345 456 567)
for seed in 123 234 345 456 567
do
  python train_bart.py --old_model_name facebook/mbart-large-cc25 --mode japanese_xlsum --tuning_mode finetune --do_train yes --fp16 yes --bsz 8  --epoch 10  --gradient_accumulation_step 3 --learning_rate 0.00005  --mid_dim 800 --seed 114514 --few_shot "$1" --few_shot_seed "$seed"
done
