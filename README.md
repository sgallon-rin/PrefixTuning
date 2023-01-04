# Prefix Tuning
## Files:
    .
    ├── gpt2                          # Code for GPT2 style autoregressive LM
    │   ├── train_e2e.py              # high-level scripts to train.
    │   ├── train_control.py          # code that implements prefix-tuning.
    │   ├── trainer_prefix.py         # trainer code for the training loop. 
    │   ├── run_language_modeling.py  # training code (contains data loading, model loading, and calls trainer)
    │   ├── gen.py                    # high-level scripts to decode. 
    │   └── run_generation.py         # decoding code. 
    │
    ├── seq2seq                       # Code for encoder-decoder architecture
    │   ├── train_bart.py             # high-level scripts to train.
    │   ├── prefixTuning.py           # code that implements prefix-tuning.
    │   ├── finetune.py               # training code (contains data loading, model loading, and calls trainer)   
    │   ├── lightning_base.py         # helper code
    │   ├── utils.py                  # helper code
    │   └── callbacks.py              # helper code
    └── ...


To run the code for GPT2 style autoregressive LM, the code is in ``gpt2/``. This corresponds to the table-to-text experiments in the paper. 

To run the code for encoder-decoder architecture like BART,  the code is in ``seq2seq``. This corresponds to the summarization experiments in the paper. 

The two primary scripts I used to run my codes are `` gpt2/train_e2e.py`` (for table-to-text) and ``seq2seq/train_bart.py``(for summarization).
they are set to default of good hyperparameters, and can be used to tune hyperparameter :) 

-----------------------------------------------------
## Setup:

``cd transformer; pip install -e .``

-----------------------------------------------------
## Train via prefix-tuning:

```shell
cd gpt2;

python train_e2e.py --optim_prefix yes --preseqlen 5 --epoch 5 --learning_rate 0.00005 --mode webnlg --bsz 5 --seed 101
```


```shell
cd seq2seq; 

python train_bart.py --mode xsum --preseqlen 200 --do_train yes --fp16 yes --bsz 16  --epoch 30  --gradient_accumulation_step 3 --learning_rate 0.00005  --mid_dim 800
```


Other baseline approaches 

```
cd gpt2;

python train_e2e.py --tuning_mode {finetune/adaptertune} --epoch 5 --learning_rate 0.00005 --mode webnlg --bsz 5 --seed 101
```

```
cd seq2seq;

python train_bart.py --tuning_mode finetune --epoch 5 --learning_rate 0.00005 --mode xsum --bsz 5 --seed 101
```
-----------------------------------------------------

## Decode:

```shell
cd gpt2;

python gen.py {data2text/webnlg/...} yes test {checkpoint_path} no
```


```shell
cd seq2seq; 

python train_bart.py --mode xsum --do_train no --prefix_model_path {checkpoint_path} --preseqlen {same as training} --mid_dim {same as training}
```

-----------------------------------------------------

For details of the methods and results, please refer to our paper. 

```bibtex
@misc{li2021prefixtuning,
      title={Prefix-Tuning: Optimizing Continuous Prompts for Generation}, 
      author={Xiang Lisa Li and Percy Liang},
      year={2021},
      eprint={2101.00190},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

https://worksheets.codalab.org/worksheets/0x16e0c8e7ab1f4b22aaccddc8b586541f

## Environment

https://github.com/XiangLi1999/PrefixTuning/issues/29

python version: 3.6.8

```shell
conda env create -f environment.yml -n prefix-tuning
conda activate prefix-tuning
cd transformers
pip install -e .
pip install pytorch-lightning==0.9.0
```

For mbart, use `transformers` 4.18.0 (as the old version in this repo cannot identify `facebook/mbart-large-50`).
See: https://huggingface.co/docs/transformers/migration
```
pip install transformers[sentencepiece]
```

pytorch version:
```
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```