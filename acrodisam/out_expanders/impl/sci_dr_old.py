"""
Out expander based on the paper:
Singh, Aadarsh, and Priyanshu Kumar. 
"SciDr at SDU-2020: IDEAS-Identifying and Disambiguating Everyday Acronyms for Scientific Domain."

Original code can be found in this repository:
https://github.com/aadarshsingh191198/AAAI-21-SDU-shared-task-2-AD/

Additional changes had to be perfomed to the code:
- Generalize to other datasets and external data sources
- Code refactoring
- Since the original work was proposed to a dataset containing sentences, we split the article text
 into sentences, for training each sentence is a sample. For prediction we sum the start and end 
 index predicted output like it was being done when merging output from multiple models.
"""
from os import path
import re
import os
import sys
import json
import ast
import pandas as pd
from pathlib import Path
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm
from sklearn.utils.extmath import softmax
from sklearn import model_selection
from sklearn.metrics import classification_report, f1_score

import itertools

import torch
import torch.optim as optim
import torch.nn as nn
import transformers
from transformers import AdamW
import tokenizers
from nltk.tokenize import RegexpTokenizer, sent_tokenize

from Logger import logging
from helper import TrainInstance, TestInstance
from inputters import TrainOutDataManager
from out_expanders._base import OutExpanderArticleInput
from text_preparation import get_expansion_without_spaces

from .._base import OutExpanderFactory, OutExpander
from pip._internal.req.req_uninstall import _unique

logger = logging.getLogger(__name__)


def seed_all(seed=42):
    """
    Fix seed for reproducibility
    """
    # python RNG
    import random

    random.seed(seed)

    # pytorch RNGs
    import torch

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # numpy RNG
    import numpy as np

    np.random.seed(seed)


class config:
    SEED = 42
    #KFOLD = 1
    KFOLD = 5
    TRAIN_FILE = "/home/jpereira/git/AcroDisam/input/sdu-shared/train.csv"
    VAL_FILE = "/home/jpereira/git/AcroDisam/input/sdu-shared/dev.csv"  # Diff
    SAVE_DIR = "/home/jpereira/git/AcroDisam/acrodisam_app/generated_files/ScienceWISE/"
    MAX_LEN = 192
    # MODEL = '../input/scibert-uncased-wiki-article1'
    MODEL = "/home/jpereira/git/AcroDisam/input/scibert_scivocab_uncased"
    TOKENIZER = tokenizers.BertWordPieceTokenizer(f"{MODEL}/vocab.txt", lowercase=True)
    # WEIGHTS_PATH = "/home/jpereira/git/AcroDisam/input/gpu-scibert-uncased-wiki-article-ws-sdu-1"  # Diff
    EPOCHS = 5
    #EPOCHS = 1
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 4
    #DICTIONARY = json.load(
    #    open("/home/jpereira/git/AcroDisam/input/sdu-shared/diction.json")
    #)

    """
    A2ID = {}
    for k, v in DICTIONARY.items():
        for w in v:
            A2ID[w] = len(A2ID)
    """

class AverageMeter:
    """
    Computes and stores the average and current value
    Source : https://www.kaggle.com/abhishek/bert-base-uncased-using-pytorch/
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    Early stopping utility
    Source : https://www.kaggle.com/abhishek/bert-base-uncased-using-pytorch/
    """

    def __init__(self, patience=7, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                "EarlyStopping counter: {} out of {}".format(
                    self.counter, self.patience
                )
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(
                "Validation score improved ({} --> {}). Saving model!".format(
                    self.val_score, epoch_score
                )
            )
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score

def indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError:
            return result
        result.append(offset)
        
def sample_text_old(text, acronym, expansion, max_len):
    if expansion:
        expansion_without_spaces = get_expansion_without_spaces(expansion)
        text = text.replace(expansion_without_spaces, acronym)
        
    text = [item[0] for item in config.TOKENIZER.pre_tokenizer.pre_tokenize_str(text)]
    
    for idx in indices(text, acronym):
        left_idx = max(0, idx - max_len // 2)
        right_idx = min(len(text), idx + max_len // 2)
        sampled_text = text[left_idx:right_idx]
        """
        if idx > left_idx:
            sampled_text = text[left_idx:idx] + [acronym]
        else:
            sampled_text = [acronym]
            
        if idx < right_idx:
            sampled_text += text[idx+1:right_idx]
        """
            
        yield " ".join(sampled_text)

def sample_text(text, acronym, expansion, max_len):
    if expansion:
        expansion_without_spaces = get_expansion_without_spaces(expansion)
        text = text.replace(expansion_without_spaces, acronym)
        
    sentences = sent_tokenize(text)
    for sent in sentences:
        text = [item[0] for item in config.TOKENIZER.pre_tokenizer.pre_tokenize_str(sent)]
        try:
            idx = text.index(acronym)
        #for idx in indices(text, acronym):
            left_idx = max(0, idx - max_len // 2)
            right_idx = min(len(text), idx + max_len // 2)
            sampled_text = text[left_idx:right_idx]
            """
            if idx > left_idx:
                sampled_text = text[left_idx:idx] + [acronym]
            else:
                sampled_text = [acronym]
                
            if idx < right_idx:
                sampled_text += text[idx+1:right_idx]
            """
                
            yield " ".join(sampled_text)
        except ValueError:
            pass #ingnore sentence

def _unique_expansions(exp_articles_list):
    return {item[0] for item in exp_articles_list}

def process_data(text, acronym, expansion, tokenizer, max_len, acronym_db):

    text = str(text)
    expansion = str(expansion)
    acronym = str(acronym)

    """
    n_tokens = len(text.split())
    if n_tokens > 120:
        text = sample_text(text, acronym, 120)
    """

    #answers = acronym + " " + " ".join(config.DICTIONARY[acronym])
    answers = acronym + " " + " ".join(_unique_expansions(acronym_db[acronym]))
    
    start = answers.find(expansion)
    end = start + len(expansion)

    char_mask = [0] * len(answers)
    for i in range(start, end):
        char_mask[i] = 1

    tok_answer = tokenizer.encode(answers)
    answer_ids = tok_answer.ids
    answer_offsets = tok_answer.offsets

    answer_ids = answer_ids[1:-1]
    answer_offsets = answer_offsets[1:-1]

    target_idx = []
    for i, (off1, off2) in enumerate(answer_offsets):
        if sum(char_mask[off1:off2]) > 0:
            target_idx.append(i)

    start = target_idx[0]
    end = target_idx[-1]

    text_ids = tokenizer.encode(text).ids[1:-1]

    token_ids = [101] + answer_ids + [102] + text_ids + [102]
    offsets = [(0, 0)] + answer_offsets + [(0, 0)] * (len(text_ids) + 2)
    mask = [1] * len(token_ids)
    token_type = [0] * (len(answer_ids) + 1) + [1] * (2 + len(text_ids))

    text = answers + text
    start = start + 1
    end = end + 1

    padding = max_len - len(token_ids)

    if padding >= 0:
        token_ids = token_ids + ([0] * padding)
        token_type = token_type + [1] * padding
        mask = mask + ([0] * padding)
        offsets = offsets + ([(0, 0)] * padding)
    else:
        token_ids = token_ids[0:max_len]
        token_type = token_type[0:max_len]
        mask = mask[0:max_len]
        offsets = offsets[0:max_len]

    assert len(token_ids) == max_len
    assert len(mask) == max_len
    assert len(offsets) == max_len
    assert len(token_type) == max_len

    return {
        "ids": token_ids,
        "mask": mask,
        "token_type": token_type,
        "offset": offsets,
        "start": start,
        "end": end,
        "text": text,
        "expansion": expansion,
        "acronym": acronym,
    }


class Dataset:
    def __init__(self, text, acronym, expansion, acronym_db):
        self.text = text
        self.acronym = acronym
        self.expansion = expansion
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        self.acronym_db = acronym_db

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        data = process_data(
            self.text[item],
            self.acronym[item],
            self.expansion[item],
            self.tokenizer,
            self.max_len,
            self.acronym_db
        )

        return {
            "ids": torch.tensor(data["ids"], dtype=torch.long),
            "mask": torch.tensor(data["mask"], dtype=torch.long),
            "token_type": torch.tensor(data["token_type"], dtype=torch.long),
            "offset": torch.tensor(data["offset"], dtype=torch.long),
            "start": torch.tensor(data["start"], dtype=torch.long),
            "end": torch.tensor(data["end"], dtype=torch.long),
            "text": data["text"],
            "expansion": data["expansion"],
            "acronym": data["acronym"],
        }


def get_loss(start, start_logits, end, end_logits):
    loss_fn = nn.CrossEntropyLoss()
    start_loss = loss_fn(start_logits, start)
    end_loss = loss_fn(end_logits, end)
    loss = start_loss + end_loss
    return loss


class BertAD(nn.Module):
    stage_2 = False

    def __init__(self):
        super(BertAD, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(
            config.MODEL, output_hidden_states=True
        )
        self.layer = nn.Linear(768, 2)
        if self.stage_2:
            self.drop_out = nn.Dropout(0.1)

    def forward(self, ids, mask, token_type, start=None, end=None):
        output = self.bert(
            input_ids=ids, attention_mask=mask, token_type_ids=token_type
        )

        if self.stage_2:
            out = self.drop_out(output[0])
            logits = self.layer(out)
        else:
            logits = self.layer(output[0])
        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start is not None and end is not None:
            loss = get_loss(start, start_logits, end, end_logits)
        else:
            loss = 0

        return loss, start_logits, end_logits


def train_fn(data_loader, model, optimizer, device):
    model.train()
    losses = AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk0):
        ids = d["ids"]
        mask = d["mask"]
        token_type = d["token_type"]
        start = d["start"]
        end = d["end"]

        ids = ids.to(device, dtype=torch.long)
        token_type = token_type.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        start = start.to(device, dtype=torch.long)
        end = end.to(device, dtype=torch.long)

        model.zero_grad()
        loss, start_logits, end_logits = model(ids, mask, token_type, start, end)

        loss.backward()
        optimizer.step()
        # xm.optimizer_step(optimizer, barrier=True)

        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def evaluate_jaccard(text, acronym, offsets, idx_start, idx_end, candidates):
    filtered_output = ""
    for ix in range(idx_start, idx_end + 1):
        filtered_output += text[offsets[ix][0] : offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            filtered_output += " "

    #candidates = config.DICTIONARY[acronym]
    candidate_jaccards = [
        jaccard(w.strip(), filtered_output.strip()) for w in candidates
    ]
    idx = np.argmax(candidate_jaccards)

    return candidate_jaccards[idx], candidates[idx]


def eval_fn(data_loader, model, device, acronym_db, exp_to_id):
    model.eval()
    losses = AverageMeter()
    jac = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))

    pred_expansion_ = []
    true_expansion_ = []

    for bi, d in enumerate(tk0):
        ids = d["ids"]
        mask = d["mask"]
        token_type = d["token_type"]
        start = d["start"]
        end = d["end"]

        text = d["text"]
        expansion = d["expansion"]
        offset = d["offset"]
        acronym = d["acronym"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type = token_type.to(device, dtype=torch.long)
        start = start.to(device, dtype=torch.long)
        end = end.to(device, dtype=torch.long)

        with torch.no_grad():
            loss, start_logits, end_logits = model(ids, mask, token_type, start, end)

        start_prob = torch.softmax(start_logits, dim=1).detach().cpu().numpy()
        end_prob = torch.softmax(end_logits, dim=1).detach().cpu().numpy()

        jac_ = []

        for px, s in enumerate(text):
            start_idx = np.argmax(start_prob[px, :])
            end_idx = np.argmax(end_prob[px, :])
            
            candidates = list(_unique_expansions(acronym_db[acronym[px]]))

            js, exp = evaluate_jaccard(
                s, #expansion[px],
                 acronym[px], offset[px], start_idx, end_idx, candidates
            )
            jac_.append(js)
            pred_expansion_.append(exp)
            true_expansion_.append(expansion[px])

        jac.update(np.mean(jac_), len(jac_))
        losses.update(loss.item(), ids.size(0))

        tk0.set_postfix(loss=losses.avg, jaccard=jac.avg)

    #pred_expansion_ = [config.A2ID[w] for w in pred_expansion_]
    #true_expansion_ = [config.A2ID[w] for w in true_expansion_]
    pred_expansion_ = [exp_to_id[w] for w in pred_expansion_]
    true_expansion_ = [exp_to_id[w] for w in true_expansion_]

    f1 = f1_score(true_expansion_, pred_expansion_, average="macro")

    print("Average Jaccard : ", jac.avg)
    print("Macro F1 : ", f1)

    return f1


def run(df_train, df_val, fold, acronym_db):
    stage_2 = False

    train_dataset = Dataset(
        text=df_train.text.values,
        acronym=df_train.acronym_.values,
        expansion=df_train.expansion.values,
        acronym_db=acronym_db
    )

    valid_dataset = Dataset(
        text=df_val.text.values,
        acronym=df_val.acronym_.values,
        expansion=df_val.expansion.values,
        acronym_db=acronym_db
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=2
    )

    model = BertAD()
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    if stage_2:
        model.load_state_dict(
            torch.load(os.path.join(config.WEIGHTS_PATH, f"model_{fold}.bin"))
        )
        print("Loaded phase 1 model weights")
    model.to(device)

    lr = 2e-5
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    es = EarlyStopping(patience=2, mode="max")

    exp_to_id = {}
    for k, v in acronym_db.items():
        for w in _unique_expansions(v):
            exp_to_id[w] = len(exp_to_id)

    print("Starting training....")
    for epoch in range(config.EPOCHS):
        train_fn(train_data_loader, model, optimizer, device)
        valid_loss = eval_fn(valid_data_loader, model, device, acronym_db, exp_to_id)
        print(f"Fold {fold} | Epoch :{epoch + 1} | Validation Score :{valid_loss}")
        if fold is None:
            es(valid_loss, model, model_path=os.path.join(config.SAVE_DIR, "model.bin"))
        else:
            es(
                valid_loss,
                model,
                model_path=os.path.join(config.SAVE_DIR, f"singh_kumar_scibert_scivocab_uncased_model_{fold}.bin"),
            )

    return es.best_score


def run_k_fold(fold_id, train, acronym_db):
    """
    Perform k-fold cross-validation
    """
    seed_all()
    stage_1 = False
    if stage_1:
        train2 = pd.read_csv(config.TRAIN_FILE)
    else:
        df_train = pd.read_csv(config.TRAIN_FILE)
        df_val = pd.read_csv(config.VAL_FILE)

        # concatenating train and validation set
        train2 = pd.concat([df_train, df_val]).reset_index()

    # dividing folds
    kf = model_selection.StratifiedKFold(
        n_splits=config.KFOLD, shuffle=True, random_state=config.SEED
    )
    for fold, (train_idx, val_idx) in enumerate(
        kf.split(X=train, y=train.acronym_.values)
    ):
        train.loc[val_idx, "kfold"] = fold

    print(
        f"################################################ Fold {fold_id} #################################################"
    )
    df_train = train[train.kfold != fold_id]
    df_val = train[train.kfold == fold_id]

    return run(df_train, df_val, fold_id, acronym_db)

class SinghKumarVoteFactory(
    OutExpanderFactory
):  # pylint: disable=too-few-public-methods
    """
    Out expander factory to predict the expansion for an article based on doc2vec models per acronym
    """

    def __init__(self, *args, **kwargs):
        pass

    def get_expander(
        self, train_data_manager: TrainOutDataManager, execution_time_observer=None
    ):
        data = []
        article_acronym_db = train_data_manager.get_article_acronym_db()
        for article_id, article_text in train_data_manager.get_raw_articles_db().items():
            acro_exp = article_acronym_db[article_id]
            
            for acronym, expansion in acro_exp.items():
                for text in sample_text(article_text, acronym, expansion, 120):
                    data.append([acronym, expansion, text])
        
        train = pd.DataFrame(data, columns = ['acronym_', 'expansion', 'text'])
        
        models=[]
        for fold in range(config.KFOLD):
            model_path=os.path.join(config.SAVE_DIR, f"singh_kumar_scibert_scivocab_uncased_model_{fold}.bin")
            if path.exists(model_path):
                device = torch.device("cuda")
                #model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)
                #model_config.output_hidden_states = True
                
                model = BertAD()#conf=model_config)
                model.to(device)
                
                model.load_state_dict(torch.load(model_path))
                model.eval()
            else:
                model = run_k_fold(fold, train, train_data_manager.get_acronym_db())
        
            models.append(model)
        
        return _SinghKumarVote(models)




class _SinghKumarVote(OutExpander):
    def __init__(self, models):
        self.models = models


    def models_infer(self, texts, acronym, candidate_expansions):
        #final_output = []
        
        test_dataset = Dataset(
            text=texts,
            acronym=list(itertools.repeat(acronym, len(texts))),
            expansion=list(itertools.repeat(acronym, len(texts))),
            acronym_db={acronym:[[cand_exp] for cand_exp in candidate_expansions]}
        )
        
        data_loader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=config.VALID_BATCH_SIZE,
            num_workers=1
        )
        
        with torch.no_grad():
            outputs_start = None
            outputs_end = None
            #tk0 = tqdm(data_loader, total=len(data_loader))
            tk0 = data_loader
            for bi, d in enumerate(tk0):
                
                ids = d["ids"]
                mask = d["mask"]
                token_type = d["token_type"]
                start = d["start"]
                end = d["end"]
        
                text = d["text"]
                #expansion = d["expansion"]
                #offset = d["offset"]
                acronym = d["acronym"]
                device = torch.device("cuda")
                ids = ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)
                token_type = token_type.to(device, dtype=torch.long)
                start = start.to(device, dtype=torch.long)
                end = end.to(device, dtype=torch.long)
            
                offsets = d["offset"].numpy()
                #if len(text) > 1:
                #    print("here")
                for model in self.models:
                    _, outputs_start1, outputs_end1 = model(
                        ids=ids,
                        mask=mask,
                        token_type=token_type
                    )
                    #outputs_start = (outputs_start + outputs_start1) if outputs_start is not None else outputs_start1
                    outputs_start = (outputs_start + outputs_start1.sum(dim=0)) if outputs_start is not None else outputs_start1.sum(dim=0)
                    outputs_end = (outputs_end + outputs_end1.sum(dim=0)) if outputs_end is not None else outputs_end1.sum(dim=0)
                
            
            num_predictions = len(self.models) * len(texts)
            outputs_start /= num_predictions
            outputs_end /= num_predictions
            
            outputs_start = torch.softmax(outputs_start, dim=0).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=0).cpu().detach().numpy()
    
            start_idx = np.argmax(outputs_start)
            end_idx = np.argmax(outputs_end)
            js, exp = evaluate_jaccard(
                    text[0], #expansion[px],
                     acronym[0], offsets[0], start_idx, end_idx, list(candidate_expansions)
                )
            return js, exp
            """
            jac_ = []        
            for px, s in enumerate(text):
                start_idx = np.argmax(outputs_start[px, :])
                end_idx = np.argmax(outputs_end[px, :])
    
                js, exp = evaluate_jaccard(
                    s, #expansion[px],
                     acronym[px], offsets[px], start_idx, end_idx, candidate_expansions
                )
                jac_.append(js)
                pred_expansion_.append(exp)
                final_output.append(output_sentence)
            """

    def process_article(self, out_expander_input: OutExpanderArticleInput):

        predicted_expansions = []

        #x_train_list = out_expander_input.get_train_instances_list()

        #y_train_list = out_expander_input.train_instances_expansions_list
        acronyms_list = out_expander_input.acronyms_list
        distinct_expansions_list = out_expander_input.distinct_expansions_list

        article_text = out_expander_input.article.get_raw_text()
        for acronym, distinct_expansions in zip(acronyms_list, distinct_expansions_list):
            texts = list(sample_text(article_text, acronym, None, 120))

            jaccard_score, predct_exp = self.models_infer(texts, acronym, distinct_expansions)
            
            result = predct_exp
            confidence = jaccard_score

            predicted_expansions.append((result, confidence))
        return predicted_expansions
