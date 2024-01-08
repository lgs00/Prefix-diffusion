#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import pdb
import random
import torch
import datasets
import stanza
import spacy_stanza
from datasets import load_dataset, load_metric
import pickle

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from custom_trainer import GPT2LMHeadModelCompress, BERTModelCompress, AutoEncoderWithNoise, GPT2VAE, AR_for_cont,\
    Classifier_GPT2, Classifier_Times, Classifier_POS, Classifier_image,Classifier_Tree, Classifier_Consistency

from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0.dev0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False, pad_mask_id=None):
    if pad_mask_id is None:
        pad_mask_id = pad_token_id
    result = torch.full([len(examples), max_length], pad_token_id).tolist()
    mask_ = torch.full([len(examples), max_length], pad_mask_id).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    experiment: Optional[str] = field(
        default='compress',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    padding_mode: Optional[str] = field(
        default='block',
        metadata={"help": "blcok or pad"},
    )
    roc_train: Optional[str] = field(
        default='diffusion_lm/ROCstory',
        metadata={"help": "roc story path"},
    )
    wiki_train: Optional[str] = field(
        default='diffusion_lm/simple_wiki/data.v1.split/simple.training.txt',
        metadata={"help": "simple wiki path"},
    )
    e2e_train: Optional[str] = field(
        default='e2e_data',
        metadata={"help": "simple wiki path"},
    )

    reduced_emb: Optional[int] = field(
        default=8,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    rounding_mode: Optional[str] = field(
        default='gpt2',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    sigma: Optional[float] = field(
        default=1.0,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    n_embd: Optional[int] = field(
        default=16,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    init_emb: Optional[str] = field(
        default="",
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    task: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    synth_config:  Optional[str] = field(
        default='diffusion_lm/synthetic_data/configs/emnlp2020/experiments/difflm_seed0_m3_k32_trainc20000.yaml', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def get_corpus_rocstory(data_args):
    '''

    :param data_args:  --> this is actually the model_args in the main function.
    :return:
    '''
    import csv, json
    from collections import Counter, defaultdict
    from spacy.lang.en import English
    import numpy as np

    # print(data_args.task, 'DEBUG', '*---'*100)
    # print(model_args.task, 'DEBUG', '*---' * 100)
    # experiment=e2e-tgt-pos
    if data_args.experiment.startswith('roc') and data_args.task == 'infill':
        print('loading dataset from ROCStory')
        nlp = English()
        tokenizer = nlp.tokenizer
        sentence_lst = []
        with open(f'{data_args.roc_train}/ROCstory_full.csv', 'r') as csvfile:
            roc_reader = csv.reader(csvfile) #delimiter=' ', quotechar='|')
            for idx, row in enumerate(roc_reader):
                if idx == 0:
                    continue
                sentences = row[2:]
                for ii in [1, 2, 3]:
                    sent = " ".join([sentences[ii-1], sentences[ii+1], sentences[ii]])
                    example = [x.text for x in tokenizer(sent)]
                    sentence_lst.append(example)
        print(sentence_lst[:2])
    elif data_args.experiment.startswith('e2e-tgt'):
        print('loading dataset from simple e2e dataset')
        sentence_lst = []
        nlp = English()
        tokenizer = nlp.tokenizer
        path = f'{data_args.e2e_train}/src1_train.txt'
        with open(path, 'r') as ff:
            for row in ff:
                word_lst = row.split('||')[1]   # 取出句子
                word_lst = [x.text for x in tokenizer(word_lst)]
                sentence_lst.append(word_lst)
        print(sentence_lst[:2])

    # get tokenizer.
    if not data_args.experiment.startswith('e2e-back'):
        counter = Counter()
        for input_ids in sentence_lst:
            counter.update(input_ids)

    vocab_dict = {'START': 0, 'END': 1, 'UNK':2, 'PAD':3}
    for k, v in counter.items():
        if v > 10:
            vocab_dict[k] = len(vocab_dict)     # 如果当前词的数量超过10个，则添加到词典中
    print(len(counter), len(vocab_dict))    # e2e:2974,821

    return sentence_lst, vocab_dict


def get_corpus_img_rocstory(data_args):
    from collections import Counter, defaultdict
    from spacy.lang.en import English
    import numpy as np
    print('loading dataset from simple e2e dataset')
    sentence_lst = []
    nlp = English()
    data_path = 'datasets/coco/coco_train_text_img1.pkl'
    tokenizer = nlp.tokenizer
    with open(data_path,'rb') as f:
        all_data = pickle.load(f)
    print("Data size is %0d"%len(all_data["captions"]))
    sys.stdout.flush()
    #captions_raw = all_data["captions"][0:len(all_data['captions'])//40]
    captions_raw = all_data["captions"][100000:200000]
    random.shuffle(captions_raw)
    caption_lst = [caption['raw'] for caption in captions_raw]
    img_lst = [caption['img'] for caption in captions_raw]
    img_label = [caption['label'] for caption in captions_raw]
    #img0_lst = [caption['img0'] for caption in captions_raw]
    for raw in caption_lst:
        word_lst = [x.text for x in tokenizer(raw)]
        sentence_lst.append(word_lst)
    print(sentence_lst[:2])
    #print("img",img_lst[0])
    print("lebel",img_label[:10])
    if not data_args.experiment.startswith('e2e-back'):
        counter = Counter()
        for input_ids in sentence_lst:
            counter.update(input_ids)
    
    vocab_dict = {'START': 0, 'END': 1, 'UNK':2, 'PAD':3}
    for k, v in counter.items():
        if v > 60:
            vocab_dict[k] = len(vocab_dict)
    print(len(counter), len(vocab_dict))
    return sentence_lst, img_lst,img_label, vocab_dict


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # sys.argv[1]为命令行的第二个参数，为output_dir=*******
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()     # 三类参数
    #training_args._n_gpu=4
    #training_args.dataloader_num_workers=1
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    ###################### LOAD DATASETS & dictionary #########################
    if model_args.experiment.startswith('roc') or\
            model_args.experiment.startswith('e2e-tgt'):
        train_dataset,img,img_label, vocab = get_corpus_img_rocstory(model_args) # TODO: include validation sets.
        print(len(vocab), 'derived vocabs')

        # train_dataset = train_dataset[:100]
        from datasets import Dataset
        #model_args.task = wp
        train_datasets = Dataset.from_dict({'text': train_dataset, 'img': img, 'label': img_label})
        raw_datasets = train_datasets.train_test_split(0.01)    # 41641条训练数据，421条测试数据
        #pdb.set_trace()
        #raw_datasets = train_datasets
        if model_args.experiment in ['e2e-tgt-pos', 'e2e-tgt-gen-pos']:
            pos_vocab = {'START': 0, 'END': 1, 'UNK': 2, 'PAD': 3}
            pos_lst = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB',
                       'ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ',
                       'PUNCT', 'SYM', 'X']
            for x in pos_lst:
                pos_vocab[x] = len(pos_vocab)   # 词典中加上pos标签
        elif model_args.experiment in ['e2e-tgt-img']:
            pos_vocab = {'START': 0, 'END': 1, 'UNK': 2, 'PAD': 3}
            pos_lst = ['n','p']
            for x in pos_lst:
                pos_vocab[x] = len(pos_vocab)
        raw_datasets.vocab = vocab
        raw_datasets['validation'] = raw_datasets['test']

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:  # None
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path: # improved-diffusion/pretrained/bert-base-uncased
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,      # None
        "use_fast": model_args.use_fast_tokenizer,  # True
        "revision": model_args.model_revision,  # 'main'
        "use_auth_token": True if model_args.use_auth_token else None,  # None
    }
    ############# LiOAD TOKENIZER ##############
    if model_args.experiment.startswith('synth') or \
            model_args.experiment.startswith('e2e-tgt') or\
            model_args.experiment.startswith('e2e-back'):
        print('\ninitializing the tokenizer with small vocab\n' + '*'*100)
        # model_args.task=wp
        if model_args.task in ['data_teacher', 'finetune']:
            print('loading from pretrained models tokenizer')
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
            print(type(tokenizer))
            if model_args.experiment == 'e2e-tgt-gen-tree':
                # new_vocabs_added = list(tree_vocab.keys())
                tokenizer.add_tokens(list(tree_vocab.keys()))
                tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            elif model_args.experiment == 'e2e-tgt-gen-pos':
                # print(list(pos_vocab.keys()))
                tokenizer.add_tokens(list(pos_vocab.keys()))
                tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            else:
                tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        else:
            print('loading from dataset-specific vocab')
            tokenizer = raw_datasets.vocab
            reverse_tokenizer = {v:k for k, v in tokenizer.items()}
    # model_args.model_name_or_path=improved-diffusion/pretrained/bert-base-uncased/
    if model_args.model_name_or_path:
        ############# LOAD MODELS for controllable classifier ##############
        if model_args.experiment in ['e2e-back', 'e2e-back_t2', 'e2e-tgt-pos','e2e-tgt-img', 'e2e-tgt-tree']:
            import torch
            config.vocab_size = len(tokenizer)
            print('\n Initializing the model from scratch \n' + '*' * 100)
            # EDIT
            # also loading the diffusion model.
            import json, argparse
            from improved_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
            config_path = os.path.join(model_args.init_emb, "training_args.json")
            print(config_path)      # 训练好的diffusion-model path，
            # parser = argparse.ArgumentParser(description='Process some integers.')
            # args = parser.parse_args()
            with open(config_path, 'rb', ) as f:
                training_args2 = json.load(f)
            # args.__dict__.update(training_args)
            training_args2['sigma_small'] = True
            training_args2['diffusion_steps'] = 200  # 500  # DEBUG
            temp_dict = model_and_diffusion_defaults()
            temp_dict.update(training_args2)
            _, diffusion = create_model_and_diffusion(
                **temp_dict
            )

            config.input_emb_dim = model_args.n_embd    # 16
            config.train_diff_steps = training_args2['diffusion_steps'] # 200

            if model_args.experiment == 'e2e-tgt-pos':
                config.pos_vocab_size = len(pos_vocab)  # len(pos_vocab=21, 17个pos_tag,加四个unk等标签，作为label
                model = Classifier_POS(config=config, diffusion=diffusion, )
            elif model_args.experiment == 'e2e-tgt-img':
                config.pos_vocab_size = 2
                config.vocab_size = 3983
                model = Classifier_image(config=config, diffusion=diffusion, )

            # filename = improved-diffusion/diffusion_models/****
            filename = model_args.init_emb
            path_save = '{}/random_emb.torch'.format(filename)
            path_learned = '{}/ema_0.9999_200000.pt'.format(filename)
            # if model_args.experiment == 'e2e-tgt-img'and model_args.learned_emb == 'no':
            #     model.transformer.embeddings.word_embeddings.load_state_dict(torch.load(path_save))
            #     model.transformer.embeddings.word_embeddings.weight.requires_grad = False
            model_args.learned_emb ='yes'
            if model_args.experiment == 'e2e-tgt-img' and model_args.learned_emb == 'yes':
                print('loading the learned embeddings')
                learned_embeddings = torch.load(path_learned)['word_embedding.weight']
                model.transformer.embeddings.word_embeddings.weight.data = learned_embeddings.clone()   # random_emb.torch,采用训练好的embedding
                model.transformer.embeddings.word_embeddings.weight.requires_grad = False
            elif model_args.experiment =='e2e-tgt-img' and model_args.learned_emb == 'no':
                model.transformer.embeddings.word_embeddings.load_state_dict(torch.load(path_save))
                model.transformer.embeddings.word_embeddings.weight.requires_grad = False
    model.resize_token_embeddings(3985)
    #pdb.set_trace()
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:  # true
        column_names = raw_datasets["train"].column_names   # column_names=text,raw_datasets["train"]={features:['text'],num_rows:41640}
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]    # text_column_name="text"

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    if model_args.experiment in ['e2e-tgt-pos', 'e2e-tgt-img','e2e-tgt-gen-pos']:
        assert model_args.task != 'data_teacher', 'should not be data_teacher.'
        # nlp = stanza.Pipeline(lang='en', processors='mwt,pos')
        nlp = spacy_stanza.load_pipeline("en", processors={"tokenize": "spacy"})
        def tokenize_function(examples):
            vocab_dict = raw_datasets.vocab # len(vocab_dict)=821
            with CaptureLogger(tok_logger) as cl:
                #sent_lst = [" ".join(seq) for seq in examples['text']]
                #sent_full = " ".join(sent_lst)
                #doc = nlp(sent_full)
                #doc_token_pos = [(token.text, token.pos_,) for token in doc]
                #len_lst = [len(seq) for seq in examples['text']]
                # print(sum(len_lst),  len(doc_token_pos))
                #assert sum(len_lst) == len(doc_token_pos)   # 23693,11252
                pos_lst = []
                init_idx = 0

                if model_args.experiment == 'e2e-tgt-pos':
                    input_ids = [[0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for seq in examples['text']]   # 每条句子的embedding，不等长，整数编号
                    pos_tags = [[0] + [pos_vocab[x] for x in seq] + [1] for seq in pos_lst] # 每条句子的的pos_tag,[[0,...1],[0,...1],...]每个句子不等长
                    print(pos_tags) # len(input_ids)=1000
                    result_dict = {'input_ids': input_ids, 'pos_tags':pos_tags}
                elif model_args.experiment == 'e2e-tgt-img':
                    input_ids = [[0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for seq in examples['text']]
                    #with open
                    result_dict = {'input_ids': input_ids,'img':examples['img'],'img_label':examples['label'],'labels':examples['label']}
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
                )
            return result_dict

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(      #raw_datasets是数据集，raw_datasets["train"]等等
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        def pad_function(group_lst):
            if model_args.experiment == 'e2e-tgt-pos' or model_args.experiment == 'e2e-tgt-img':
                vocab_dict = raw_datasets.vocab
                max_length = 32
                # group_lst['input_ids']=[0,***(token 的整数),1,3,3,3,3...],0和1是开始和结尾，3用来做填充。词典里3是pad，64的长度
                group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict['PAD'], max_length)
                max_src_length = 64 #min(seqlen, max_src_length)
                group_lst['img'] = group_lst['img']
                group_lst['img_label'] = group_lst['img_label']
                group_lst['labels']=group_lst['img_label']
                # group_lst['pos_ids']=[0,***(pos tag的整数),1,3,3,3,...] 64的长度
                # group_lst['src_mask'] =[-100,-100...(64*2),0,***(pos tag的整数),1,3,3,3...] 128d的长度
                #group_lst['labels']=[1,1,....,3,3,.....],64的长度
            return group_lst

        with training_args.main_process_first(desc="grouping texts together"):
            lm_datasets = tokenized_datasets.map(
                pad_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,   # None
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"padding",
            )

    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]    # dataset({features: ['input_ids', 'labels', 'pos_ids', 'pos_tags', 'src_mask'],num_rows: 41640})
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            #print(logits[0].shape, logits[1].shape)
            #zero = torch.zeros_like(logits)
            #one = torch.ones_like(logits)
            #logits_label = torch.where(logits>=0,one,logits)
            #logits_label = torch.where(logits_label<0,zero,logits_label)
            m = torch.nn.Softmax(dim=1)
            
            logits = m(logits)
            #print("logits",logits)
            logits_label = logits.argmax(dim=-1)
            return logits_label

        metric = load_metric("accuracy")
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            
            #preds = preds.squeeze(1)
            return metric.compute(predictions=preds, references=labels)

    trainer_tokenizer = None if ((model_args.experiment in ['pos', 'synth', 'roc', 'simple-wiki', 'e2e-tgt','e2e-tgt-img',
                                                            'e2e-tgt-pos','e2e-tgt-tree', 'e2e-back', 'e2e-back_t2']
                                 or model_args.experiment in ['synth_emb', 'pos_emb', 'roc_emb', 'simple-wiki_emb', 'e2e-tgt_emb'])
                                 and model_args.task != 'data_teacher') \
                        else tokenizer
    # Initialize our Trainer
    training_args.learning_rate = 4e-4
    #training_args.label_names =['labels','img','img_label','pos_ids']
    #training_args.resume_from_checkpoint ="classifier_models/e2e-tgt-imglast_e=20_b=20_m=improved-diffusion/pretrained/bert-base-uncased/_wikitext-103-raw-v1_101_wp_None/"
    print("training_args",training_args)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=trainer_tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
    )

    # Training
    #training_args.do_train=False
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    #training_args.do_eval=False
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        #try:
        #    perplexity = math.exp(metrics["eval_loss"])
        #except OverflowError:
        #    perplexity = float("inf")
        #metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    #if training_args.push_to_hub:
    #    trainer.push_to_hub(**kwargs)
    #else:
    #    trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
