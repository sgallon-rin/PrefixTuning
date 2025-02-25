#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        : 2023/1/1 16:03
# @Author      : sgallon
# @Email       : shcmsgallon@outlook.com
# @File        : preprocess_xlsum.py
# @Description :

import os
import json
import argparse
from typing import List, Dict

# https://huggingface.co/facebook/mbart-large-cc25
MBART25_LANGS = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX",
                 "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN",
                 "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT",
                 "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO",
                 "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN"]
# https://github.com/csebuetnlp/xl-sum
XLSUM_LANGS = ["arabic", "", "", "english", "spanish",
               "", "", "french", "gujarati", "hindi",
               "", "japanese", "", "korean", "",
               "", "burmese", "nepali", "", "",
               "russian", "sinhala", "turkish", "vietnamese", "chinese_simplified"]
# https://github.com/csebuetnlp/xl-sum/tree/master/multilingual_rouge_scoring
MULTI_GOUGE_LANGS = ["arabic", "", "german", "english", "spanish",
                     "", "finnish", "french", "", "hindi",
                     "italian", "japanese", "", "korean", "",
                     "", "burmese", "", "dutch", "romanian",
                     "russian", "", "turkish", "", "chinese"]
# ref: https://lh.2xlibre.net/locales/
LANGS = ["arabic", "czech", "german", "english", "spanish",
         "estonian", "finnish", "french", "gujarati", "hindi",
         "italian", "japanese", "kazakh", "korean", "lithuanian",
         "latvian", "burmese", "nepali", "dutch", "romanian",
         "russian", "sinhala", "turkish", "vietnamese", "chinese_simplified"]
# Languages used in the experiment
USED_LANGS = ["arabic", "english", "spanish", "french", "hindi",
              "japanese", "korean", "burmese", "russian", "turkish", "chinese_simplified"]
XLSUM_LANGS_DICT = {"arabic": "ar_AR",
                    "english": "en_XX",
                    "spanish": "es_XX",
                    "french": "fr_XX",
                    "hindi": "hi_IN",
                    "japanese": "ja_XX",
                    "korean": "ko_KR",
                    "burmese": "my_MM",
                    "russian": "ru_RU",
                    "turkish": "tr_TR",
                    "chinese_simplified": "zh_CN"}

# DATA_DIR = "/Users/sgallon/data/xlsum/japanese_XLSum_v2.0"
# TRAIN_FILE = os.path.join(DATA_DIR, "japanese_train.jsonl")
# VAL_FILE = os.path.join(DATA_DIR, "japanese_val.jsonl")
# TEST_FILE = os.path.join(DATA_DIR, "japanese_test.jsonl")
# OUT_DIR = "data/japanese_xlsum"

INPUT_ROOT = "/Users/shenjl/Documents/data/xlsum"
OUTPUT_ROOT = "/Users/shenjl/Documents/data/xlsum_preprocessed"

# INPUT_ROOT = "/Users/sgallon/data/xlsum"
# OUTPUT_ROOT = "/Users/sgallon/data/xlsum_preprocessed"


def load_jsonl(filename: str) -> List[Dict[str, str]]:
    with open(filename, "r", encoding='utf-8') as f:
        lines = f.readlines()
    data = json.loads("[" + ",".join(lines) + "]")
    return data


def make_data(data: List[Dict[str, str]]) -> (List[str], List[str]):
    res_summary = []
    res_text = []
    for idx, d in enumerate(data):
        try:
            summary = d["summary"]
            text = d["text"]
        except KeyError:
            print("Got empty summary or text for idx {}, skip.".format(idx))
            continue
        summary = summary.strip()
        text = text.strip()
        if not summary or not text:
            print("Got empty summary or text for idx {}, skip.".format(idx))
            continue
        res_summary.append(summary + "\n")
        res_text.append(text + "\n")
    return res_summary, res_text


# def preprocess(filename: str, datatype: str = "train") -> None:
#     assert datatype in ["train", "val", "test"]
#     print("Processing {} file: {}".format(datatype, filename))
#     data = load_jsonl(filename)
#     summaries, texts = make_data(data)
#     source_output = os.path.join(OUT_DIR, datatype + ".source")
#     target_output = os.path.join(OUT_DIR, datatype + ".target")
#     if not os.path.exists(OUT_DIR):
#         os.makedirs(OUT_DIR)
#         print("Created output dir: {}".format(OUT_DIR))
#     with open(source_output, "w", encoding="utf-8") as f:
#         f.writelines(texts)
#     print("Source saved to {}".format(source_output))
#     with open(target_output, "w", encoding="utf-8") as f:
#         f.writelines(summaries)
#     print("Source saved to {}".format(target_output))
#     print("Done! {} items in total.".format(len(summaries)))


class Preprocessor:
    data_types = ["train", "val", "test"]

    def __init__(self, lang, input_root=INPUT_ROOT, output_root=OUTPUT_ROOT):
        self.lang = lang
        self.input_root = os.path.join(input_root, "{}_XLSum_v2.0".format(lang))
        self.output_root = os.path.join(output_root, "{}_xlsum".format(lang))

    def _check_iodirs(self):
        print("Input dir is {}".format(self.input_root))
        assert os.path.exists(self.input_root), "Input dir does not exist!"
        print("Output dir is {}".format(self.output_root))
        if not os.path.exists(self.output_root):
            os.mkdir(self.output_root)
            print("Created output dir.")
        else:
            print("Output dir already exists. Overwriting to it.")

    def preprocess_single(self, datatype: str = "train"):
        assert datatype in self.data_types
        filename = os.path.join(self.input_root, "{}_{}.jsonl".format(self.lang, datatype))
        print("Processing {} file: {}".format(datatype, filename))
        data = load_jsonl(filename)
        summaries, texts = make_data(data)
        source_output = os.path.join(self.output_root, datatype + ".source")
        target_output = os.path.join(self.output_root, datatype + ".target")
        with open(source_output, "w", encoding="utf-8") as f:
            f.writelines(texts)
        print("Source saved to {}".format(source_output))
        with open(target_output, "w", encoding="utf-8") as f:
            f.writelines(summaries)
        print("Source saved to {}".format(target_output))
        print("Done! {} items in total.".format(len(summaries)))

    def preprocess(self):
        self._check_iodirs()
        for datatype in self.data_types:
            self.preprocess_single(datatype)


if __name__ == "__main__":
    # preprocess(TRAIN_FILE)
    # preprocess(VAL_FILE, "val")
    # preprocess(TEST_FILE, "test")

    # parser = argparse.ArgumentParser(description='Preprocess XLSUM.')
    # parser.add_argument('--lang', type=str, required=True, help='language to preprocess', choices=USED_LANGS)
    # args = parser.parse_args()
    # lang = args.lang
    # print("Preprocessing XLSUM {}".format(lang))
    # preprocessor = Preprocessor(lang)
    # preprocessor.preprocess()

    for lang in USED_LANGS:
        print("Preprocessing XLSUM {}".format(lang))
        preprocessor = Preprocessor(lang)
        preprocessor.preprocess()
        print("Done for lang {}\n\n".format(lang))

    # lang = "chinese_simplified"
    # print("Preprocessing XLSUM {}".format(lang))
    # preprocessor = Preprocessor(lang)
    # preprocessor.preprocess()
