#!/bin/bash
# evaluate multilingual head, finetune and prefix 100
# prefix
for lang in arabic burmese chinese_simplified english french hindi japanese korean russian spanish turkish
do
  echo ---------- "$lang" finetune ----------
  python train_bart.py --old_model_name facebook/mbart-large-cc25 --mode multilingual_xlsum_head --tuning_mode finetune --do_train no --finetune_model_path /home/lr/shenjl/research/ref-code/PrefixTuning/models/multilingual_xlsum_head/multilingual_xlsum_headfinetune_y_5_act_cat_b=24-e=10_d=0.0_l=0.0_lr=5e-05_w=0.0_s=114514_d=n_m=800/checkpoint-hello --mid_dim 800 --xlsum_lang "$lang" --rouge_lang "$lang"
  echo " "
  echo ---------- "$lang" prefix ----------
  python train_bart.py --old_model_name facebook/mbart-large-cc25 --mode multilingual_xlsum_head --preseqlen 100 --do_train no --prefix_model_path /home/lr/shenjl/research/ref-code/PrefixTuning/models/multilingual_xlsum_head/multilingual_xlsum_headprefixtune_y_100_act_cat_b=24-e=10_d=0.0_l=0.0_lr=0.0005_w=0.0_s=114514_d=n_m=800/checkpointepoch=9 --mid_dim 800 --xlsum_lang "$lang" --rouge_lang "$lang"
  echo " "
done
