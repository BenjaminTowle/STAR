import numpy as np
import os
import random

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase
from typing import List, Optional, Tuple

from src.corpora.corpus import Corpus

NUM_CANDS = 20

def add_candidates(sample, candidate_pool):
    neg_cands = random.sample(candidate_pool, k=NUM_CANDS)
    if sample["responses"] in neg_cands:
        neg_cands.remove(sample["responses"])
    sample["candidates"] = neg_cands[:NUM_CANDS-1] + [sample["responses"]]
    sample["labels"] = NUM_CANDS - 1

    return sample

def tokenize(
    samples,
    tokenizer,
    max_context_length,
    max_response_length
):

    samples["input_ids"] = tokenizer(
        samples["messages"], max_length=max_context_length, padding="max_length", truncation=True).input_ids
    samples["y_input_ids"] = tokenizer(
        samples["responses"], max_length=max_response_length, padding="max_length", truncation=True).input_ids

    if "candidates" in samples:
        cands = np.array(samples["candidates"]).reshape(-1).tolist()
        cands_ids = tokenizer(
            cands, max_length=max_response_length, padding="max_length", truncation=True, return_tensors="np").input_ids
        samples["candidate_input_ids"] = cands_ids.reshape([-1, NUM_CANDS, max_response_length]).tolist()

    return samples


@Corpus.register_subclass("reddit")
class RedditCorpus(Corpus):

    split2files = {
        "train": "reddit_train.tsv",
        "test": "reddit_test.tsv",
        "valid": "reddit_valid.tsv"
    }

    def get_dataset(
        self, 
        split: str,
        data_dir: str = "../data/reddit", 
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_context_length: int = 64,
        max_response_length: int = 64,
        **kwargs
    ):
        context, responses = zip(*[(l.split("\t")[0], l.split("\t")[1]) for l in open(
            os.path.join(data_dir, self.split2files[split]), encoding="utf-8")])

        responses = [r.replace("\n", " ") for r in responses]

        dataset = Dataset.from_dict({
            "messages": context,
            "responses": responses
        })

        if tokenizer is None:
            return dataset
        
        if split in ["test", "valid"]:
            dataset = dataset.map(
                lambda x: add_candidates(x, dataset["responses"]))

        dataset = dataset.map(
            lambda x: tokenize(
                x,
                tokenizer=tokenizer,
                max_context_length=max_context_length,
                max_response_length=max_response_length
            ), batched=True, batch_size=100
        )

        return dataset
