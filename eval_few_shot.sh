#!/bin/bash
# evaluate few-shot experiment, prefix-tune and fine-tune
# few_shot=(50 100 200 500)
# seeds=(123 234 345 456 567)
echo "Evaluating prefix-tune, size=$1, seed=$2"
python train_bart.py --old_model_name facebook/mbart-large-cc25 --mode japanese_xlsum --do_train no --prefix_model_path /home/lr/shenjl/research/ref-code/PrefixTuning/models/japanese_xlsum_"$1"_seed"$2"/japanese_xlsumprefixtune_y_100_act_cat_b=24-e=10_d=0.0_l=0.0_lr=0.0005_w=0.0_s=114514_d=n_m=800/checkpointepoch=9 --mid_dim 800 --preseqlen 100 --rouge_lang japanese
echo "Evaluating fine-tune, size=$1, seed=$2"
python train_bart.py --old_model_name facebook/mbart-large-cc25 --mode japanese_xlsum --tuning_mode finetune --do_train no --finetune_model_path /home/lr/shenjl/research/ref-code/PrefixTuning/models/japanese_xlsum_"$1"_seed"$2"/japanese_xlsumfinetune_y_5_act_cat_b=24-e=10_d=0.0_l=0.0_lr=5e-05_w=0.0_s=114514_d=n_m=800/checkpoint-hello --mid_dim 800 --rouge_lang japanese
