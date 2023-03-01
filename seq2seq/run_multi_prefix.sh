#!/bin/bash
echo "---------- arabic multilingual prefix $1 ----------"
python train_bart.py --old_model_name facebook/mbart-large-cc25 --mode multilingual_xlsum_head --preseqlen "$1" --do_train yes --fp16 yes --bsz 8  --epoch 10  --gradient_accumulation_step 3 --learning_rate 5e-4 --mid_dim 800 --seed 114514 --xlsum_lang arabic
echo ----------
for lang in burmese chinese_simplified english french hindi japanese korean russian spanish turkish
do
  echo "---------- $lang multilingual prefix $1 ----------"
  python train_bart.py --old_model_name facebook/mbart-large-cc25 --prefix_model_path /home/lr/shenjl/research/ref-code/PrefixTuning/models/multilingual_xlsum_head/multilingual_xlsum_headprefixtune_y_"$1"_act_cat_b=24-e=10_d=0.0_l=0.0_lr=0.0005_w=0.0_s=114514_d=n_m=800/checkpointepoch=9 --mode multilingual_xlsum_head --preseqlen "$1" --do_train yes --fp16 yes --bsz 8  --epoch 10  --gradient_accumulation_step 3 --learning_rate 5e-4 --mid_dim 800 --seed 114514 --xlsum_lang "$lang"
  echo ----------
done
