#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        : 2023/2/13 15:05
# @Author      : sgallon
# @Email       : shcmsgallon@outlook.com
# @File        : sample_xlsum.py
# @Description :
# For preprocessed XLSUM dataset
# downsample all languages to the same amount as korean (fewest data) by picking the first n samples

import random
import os
import shutil

POLICIES = ["random", "head"]

LANGS = ["arabic", "english", "spanish", "french", "hindi",
         "japanese", "burmese", "russian", "turkish", "chinese_simplified"]
NUM_TRAIN = 4407
NUM_VAL = 550
NUM_TEST = 550

DATA_ROOT = "/home/lr/shenjl/research/data/data_prefix_tuning/"


def get_few_shot_data(data_dir, out_dir, train_size, val_size, test_size, policy="head", seed=123):
    assert data_dir != out_dir, "Input and output dir must be different!"
    print("Processing data dir: {}".format(data_dir))
    print("Output data dir: {}".format(out_dir))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    print("Sample train size: {}\nSample val size: {}\nSample test size: {}\nSample policy: {}\nRandom seed: {}"
          .format(train_size, val_size, test_size, policy, seed))
    # train
    print("Processing train ...")
    with open(os.path.join(data_dir, "train.source"), "r", encoding="utf-8") as f:
        train_source = f.readlines()
    with open(os.path.join(data_dir, "train.target"), "r", encoding="utf-8") as f:
        train_target = f.readlines()
    sample_train_source, sample_train_target = select(train_source, train_target, train_size, policy, seed)
    with open(os.path.join(out_dir, "train.source"), "w", encoding="utf-8") as f:
        f.writelines(sample_train_source)
    with open(os.path.join(out_dir, "train.target"), "w", encoding="utf-8") as f:
        f.writelines(sample_train_target)
    # val
    print("Processing val ...")
    with open(os.path.join(data_dir, "val.source"), "r", encoding="utf-8") as f:
        val_source = f.readlines()
    with open(os.path.join(data_dir, "val.target"), "r", encoding="utf-8") as f:
        val_target = f.readlines()
    sample_val_source, sample_val_target = select(val_source, val_target, val_size, policy, seed)
    with open(os.path.join(out_dir, "val.source"), "w", encoding="utf-8") as f:
        f.writelines(sample_val_source)
    with open(os.path.join(out_dir, "val.target"), "w", encoding="utf-8") as f:
        f.writelines(sample_val_target)
    # also sample test
    print("Processing test ...")
    with open(os.path.join(data_dir, "test.source"), "r", encoding="utf-8") as f:
        test_source = f.readlines()
    with open(os.path.join(data_dir, "test.target"), "r", encoding="utf-8") as f:
        test_target = f.readlines()
    sample_test_source, sample_test_target = select(test_source, test_target, test_size, policy, seed)
    with open(os.path.join(out_dir, "test.source"), "w", encoding="utf-8") as f:
        f.writelines(sample_test_source)
    with open(os.path.join(out_dir, "test.target"), "w", encoding="utf-8") as f:
        f.writelines(sample_test_target)
    print("Done! Sampled sizes: {}, {}, {}".format(len(sample_train_target), len(sample_val_target), len(sample_test_target)))


def select(source_list, target_list, size, policy, seed):
    le = len(source_list)
    assert le == len(target_list), "Error: source and target length mismatch!"
    if size > le:
        print("Warning: required size ({}) larger than source size ({})! Return original input.".format(size, le))
        return source_list, target_list
    if policy == "head":
        res_source = source_list[::size]
        res_target = target_list[::size]
    elif policy == "random":
        print("Random seed: {}".format(seed))
        random.seed(seed)
        idxs = random.sample(list(range(le)), k=size)
        res_source = [source_list[i] for i in idxs]
        res_target = [target_list[i] for i in idxs]
    else:
        raise (ValueError, "Invalid selection policy: {}; Valid options are {}".format(policy, POLICIES))
    return res_source, res_target


if __name__ == "__main__":
    for lang in LANGS:
        print("----------\nProcessing lang: {}".format(lang))
        inpur_dir = os.path.join(DATA_ROOT, "{}_xlsum".format(lang))
        output_dir = os.path.join(DATA_ROOT, "{}_xlsum_head".format(lang))
        get_few_shot_data(inpur_dir, output_dir, NUM_TRAIN, NUM_VAL, NUM_TEST, "head")
        print("Done for lang: {}\n\n".format(lang))
