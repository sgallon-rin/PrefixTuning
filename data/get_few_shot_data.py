#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        : 2023/1/16 12:58
# @Author      : sgallon
# @Email       : shcmsgallon@outlook.com
# @File        : get_few_shot_data.py
# @Description : get few shot data for train and val, keep test the same
"""
Follow original paper, few-shot sizes {50, 100, 200, 500};
For each size, we sample 5 different datasets and average over 2 training random seeds;
We also sample a dev split (with dev size = 30% × training size) for each training set.
We use the dev split to choose hyperparameters and perform early stopping.
"""

import random
import os
import shutil

POLICIES = ["random", "head"]

# DATA_DIR = "/Users/sgallon/data/xsum_data/japanese_xlsum"
# DATA_DIR = "/Users/shenjl/Documents/data/japanese_xlsum"
DATA_DIR = "/home/lr/shenjl/research/data/data_prefix_tuning/japanese_xlsum"


def get_few_shot_data(data_dir, out_dir, train_size, val_size, policy="head", seed=123):
    assert policy in POLICIES, 'Invalid selection policy "{}", should in {}'.format(policy, POLICIES)
    assert data_dir != out_dir, "Input and output dir must be different!"
    print("Processing data dir: {}".format(data_dir))
    print("Output data dir: {}".format(out_dir))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    print("Sample train size: {}\nSample val size: {}\nSample policy: {}".format(train_size, val_size, policy))
    # train
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
    with open(os.path.join(data_dir, "val.source"), "r", encoding="utf-8") as f:
        val_source = f.readlines()
    with open(os.path.join(data_dir, "val.target"), "r", encoding="utf-8") as f:
        val_target = f.readlines()
    sample_val_source, sample_val_target = select(val_source, val_target, val_size, policy, seed)
    with open(os.path.join(out_dir, "val.source"), "w", encoding="utf-8") as f:
        f.writelines(sample_val_source)
    with open(os.path.join(out_dir, "val.target"), "w", encoding="utf-8") as f:
        f.writelines(sample_val_target)
    # keep test the same
    shutil.copy(os.path.join(data_dir, "test.source"), os.path.join(out_dir, "test.source"))
    shutil.copy(os.path.join(data_dir, "test.target"), os.path.join(out_dir, "test.target"))
    print("Done! Sampled sizes: {}, {}".format(len(sample_train_target), len(sample_val_target)))


def select(source_list, target_list, size, policy, seed):
    le = len(source_list)
    assert le == len(target_list), "Error: source and target length mismatch!"
    if size > le:
        print("Warning: required size ({}) larger than source size ({})! Return original input.".format(size, le))
        return source_list, target_list
    if policy == "head":
        res_source = source_list[:size]
        res_target = target_list[:size]
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
    nums = [50, 100, 200, 500]
    seeds = [123, 234, 345, 456, 567]
    for num in nums:
        out_dir = DATA_DIR + "_" + str(num)
        get_few_shot_data(DATA_DIR, out_dir, num, int(num * 0.3), 'head')
        # for seed in seeds:
            # out_dir = DATA_DIR + "_" + str(num) + "_seed" + str(seed)
            # get_few_shot_data(DATA_DIR, out_dir, num, int(num * 0.3), 'random', seed)
