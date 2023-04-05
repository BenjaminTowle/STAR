import torch
import random
import numpy as np
import re
import torch

from typing import List, Optional
from collections import Counter
from transformers import BertTokenizer, GPT2Tokenizer

# A global dictionary for storing shared data.
shared_storage = {}


# https://medium.com/@vadimpushtaev/python-choosing-subclass-cf5b1b67c696
class RegistryMixin:
    """Class for registering subclasses."""

    subclasses = {}  # Mapping of strs to subclasses.

    @classmethod
    def register_subclass(cls, name: str):
        """Register a subclass with a given name."""

        def decorator(subclass):
            cls.subclasses[name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, name: str, *args, **kwargs):
        if name not in cls.subclasses:
            raise ValueError(f"Bad type: {name}.")
        
        return cls.subclasses[name](*args, **kwargs)

# https://github.com/facebookresearch/ParlAI/blob/main/parlai/core/metrics.py
re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_tokenizer(args):
    gpt_tokenizer = GPT2Tokenizer.from_pretrained(args.gpt_tokenizer_path)
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_path)
    
    return {"gpt": gpt_tokenizer, "bert": bert_tokenizer}


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    # TODO: this could almost certainly be faster with a regex \s+ -> ' '
    s = ' '.join(s.split())
    return s


def get_stopwords(
    texts: List[str], threshold: float = 0.01
):
    token2freq = {}
    for text in texts:
        g_tokens = normalize_answer(text).split()

        for token in g_tokens:
            if token not in token2freq:
                token2freq[token] = 1
            else:
                token2freq[token] += 1

    threshold_idx = int(len(token2freq) * threshold)
    tokenfreqs = sorted(token2freq.items(), key=lambda x: x[1])
    tokens = [t[0] for t in tokenfreqs[-threshold_idx:]]

    return tokens


# https://github.com/facebookresearch/ParlAI/blob/main/parlai/core/metrics.py
def _prec_recall_f1_score(pred_items, gold_items):
    """
    Compute precision, recall and f1 given a set of gold and prediction items.
    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values
    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def remove_stopwords(tokens: List[str], stopwords: List[str] = None):
    if stopwords is None:
        return tokens

    new_tokens = []
    for token in tokens:
        if token not in stopwords:
            new_tokens.append(token)

    return new_tokens


 # https://github.com/facebookresearch/ParlAI/blob/main/parlai/core/metrics.py 
def compute_f1(
    guess: str, answers: List[str], expose_p_and_r: bool = False, stopwords = None
):
    if guess is None or answers is None:
        return 0
    g_tokens = remove_stopwords(normalize_answer(guess).split(), stopwords)
    scores = [
        _prec_recall_f1_score(g_tokens, remove_stopwords(normalize_answer(a).split(), stopwords))
        for a in answers
    ]
    max_p, max_r, max_f1 = 0, 0, 0
    for p, r, f1 in scores:
        max_p, max_r, max_f1 = max(max_p, p), max(max_r, r), max(f1, max_f1)
    if expose_p_and_r:
        return max_p, max_r, max_f1,
    else:
        return max_f1


def compute_f1_matrix(
    guesses: List[str], answers: Optional[List[str]] = None, stopwords: Optional[List[str]] = None, do_approximate: bool = False
):

    """Computes a 2D table of guesses versus answers"""
    g_tokens = [remove_stopwords(normalize_answer(g).split(), stopwords) for g in guesses]
    
    if answers is None:
        a_tokens = g_tokens
    else:
        a_tokens = [remove_stopwords(normalize_answer(a).split(), stopwords) for a in answers]

    sample_size = 100
    if do_approximate and len(a_tokens) > sample_size:
        a_tokens = [random.choices(a_tokens, k=sample_size) for _ in g_tokens]

        scores = np.zeros([len(g_tokens), sample_size])
        for i in range(len(g_tokens)):
            for j in range(sample_size):
                scores[i, j] = _prec_recall_f1_score(g_tokens[i], a_tokens[i][j])[-1]
        
        return scores

    scores = np.zeros([len(g_tokens), len(a_tokens)])
    for i in range(len(g_tokens)):
        for j in range(len(a_tokens)):
            scores[i, j] = _prec_recall_f1_score(g_tokens[i], a_tokens[j])[-1]

    return scores


class F1Score:
    """
    A stateful class to rapidly compute F1 score while caching
    previously computed information.
    """
    def __init__(self) -> None:
        
        # Track dictionary of tokens to doc idx pairs
        # Basically an inverted index built on the fly
        self.words = {}

        self.guesses = {}
        self.answers = {}

    def compute(self, guesses: List[str], answers: List[str]):

        for answer in answers:
            if answer not in self.answers:
                self.answers[answer] = len(self.answers)
                tokens = normalize_answer(answer).split()
                for key, value in Counter(tokens).items():
                    if key in self.words:
                        # Finally update words
                        self.words[key][answer] = value

                    else:
                        self.words[key] = {answer: value}


        for guess in guesses:
            if guess not in self.guesses:
                self.guesses[guess] = len(self.guesses)

        g_tokens = [normalize_answer(g).split() for g in guesses]
        a_tokens = [normalize_answer(a).split() for a in answers]
        scores = np.zeros([len(g_tokens), len(a_tokens)])

        # Iterate over guesses
        for i, tokens in enumerate(g_tokens):
            doc2num_same = {}
            # key, value maps words to frequencies
            for key, value in Counter(tokens).items():
                if key in self.words:
                    for doc, count in self.words[key].items():
                        if doc not in answers:
                            continue
                        if doc not in doc2num_same:
                            doc2num_same[doc] = min(value, count)
                        else:
                            doc2num_same[doc] += min(value, count)      

            doc2s = {}
            for doc, num_same in doc2num_same.items():
                if num_same == 0:
                    doc2s[doc] = 0
                    continue
                precision = 1.0 * num_same / len(tokens)
                recall = 1.0 * num_same / len(doc.split())
                f1 = (2 * precision * recall) / (precision + recall)
                doc2s[doc] = f1

            for j, (doc, s) in enumerate(doc2s.items()):
                scores[i, j] = s
        

        return scores



# Faster implementation of compute_f1_matrix
def compute_f1_matrix_fast(
    guesses: List[str], answers: List[str]
):

    g_tokens = [normalize_answer(g).split() for g in guesses]
    a_tokens = [normalize_answer(a).split() for a in answers]
    scores = np.zeros([len(g_tokens), len(a_tokens)])

    # Track dictionary of tokens to doc idx pairs
    # Basically an inverted index built on the fly
    words = {}
    for i, tokens in enumerate(a_tokens):
        for key, value in Counter(tokens).items():
            if key in words:
                # Finally update words
                words[key][i] = value

            else:
                words[key] = {i: value}

    # Iterate over guesses
    for i, tokens in enumerate(g_tokens):
        doc2num_same = {}
        # key, value maps words to frequencies
        for key, value in Counter(tokens).items():
            if key in words:
                for doc, count in words[key].items():
                    if doc not in doc2num_same:
                        doc2num_same[doc] = min(value, count)
                    else:
                        doc2num_same[doc] += min(value, count)      

        doc2s = {}
        for doc, num_same in doc2num_same.items():
            if num_same == 0:
                doc2s[doc] = 0
                continue
            precision = 1.0 * num_same / len(tokens)
            recall = 1.0 * num_same / len(a_tokens[doc])
            f1 = (2 * precision * recall) / (precision + recall)
            doc2s[doc] = f1

        for doc, s in doc2s.items():
            scores[i, doc] = s
    

    return scores