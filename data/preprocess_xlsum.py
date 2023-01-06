#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        : 2023/1/1 16:03
# @Author      : sgallon
# @Email       : shcmsgallon@outlook.com
# @File        : preprocess_xlsum.py
# @Description :

import os
import json
from typing import List, Dict

DATA_DIR = "/Users/sgallon/data/xlsum/japanese_XLSum_v2.0"
TRAIN_FILE = os.path.join(DATA_DIR, "japanese_train.jsonl")
VAL_FILE = os.path.join(DATA_DIR, "japanese_val.jsonl")
TEST_FILE = os.path.join(DATA_DIR, "japanese_test.jsonl")
OUT_DIR = "data/japanese_xlsum"


def load_jsonl(filename: str) -> List[Dict[str, str]]:
    with open(filename, "r", encoding='utf-8') as f:
        lines = f.readlines()
    data = json.loads("[" + ",".join(lines) + "]")
    return data


def make_data(data: List[Dict[str, str]]) -> (List[str], List[str]):
    res_summary = []
    res_text = []
    for d in data:
        summary = d.get("summary") + "\n"
        text = d.get("text") + "\n"
        if not summary or not text:
            print("Got empty summary/text, skip.")
            continue
        res_summary.append(summary)
        res_text.append(text)
    return res_summary, res_text


def preprocess(filename: str, datatype: str = "train") -> None:
    assert datatype in ["train", "val", "test"]
    print("Processing {} file: {}".format(datatype, filename))
    data = load_jsonl(filename)
    summaries, texts = make_data(data)
    source_output = os.path.join(OUT_DIR, datatype + ".source")
    target_output = os.path.join(OUT_DIR, datatype + ".target")
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        print("Created output dir: {}".format(OUT_DIR))
    with open(source_output, "w", encoding="utf-8") as f:
        f.writelines(texts)
    print("Source saved to {}".format(source_output))
    with open(target_output, "w", encoding="utf-8") as f:
        f.writelines(summaries)
    print("Source saved to {}".format(target_output))
    print("Done! {} items in total.".format(len(summaries)))


if __name__ == "__main__":
    preprocess(TRAIN_FILE)
    preprocess(VAL_FILE, "val")
    preprocess(TEST_FILE, "test")
