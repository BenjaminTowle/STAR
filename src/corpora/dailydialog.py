import numpy as np
import random

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from typing import Optional

from src.corpora.corpus import Corpus

NUM_CANDS = 20

def add_candidates(samples, candidate_pool, tokenizer, max_response_length: int = 64, **kwargs):
    candidates = []
    for i in range(len(samples["responses"])):
        neg_cands = random.sample(candidate_pool, k=NUM_CANDS)
        if samples["responses"][i] in neg_cands:
            neg_cands.remove(samples["responses"][i])
        candidates.append(neg_cands[:NUM_CANDS-1] + [samples["responses"][i]])
    samples["candidates"] = candidates
    neg_cands = random.sample(candidate_pool, k=NUM_CANDS)
    samples["labels"] = [NUM_CANDS - 1 for _ in range(len(samples["responses"]))]

    cands = np.array(samples["candidates"]).reshape(-1).tolist()
    cands_ids = tokenizer(
        cands, max_length=max_response_length, padding="max_length", truncation=True, return_tensors="np").input_ids
    samples["candidate_input_ids"] = cands_ids.reshape([-1, NUM_CANDS, max_response_length]).tolist()

    return samples

def map_fn(
    samples, 
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    max_context_length: int = 64,
    max_response_length: int = 64,
    debug: bool = False,
    **kwargs
):
    """Maps dialogs to message, responses"""
    messages = []
    responses = []

    for i in range(len(samples["dialog"])):
        messages += samples["dialog"][i][:-1]
        responses += samples["dialog"][i][1:]

        if debug and len(messages) >= 100:
            break

    # Prefix allows model to distinguish between messages and responses
    messages = ["message: " + m for m in messages]
    responses = ["reply: " + r for r in responses]

    inputs = {
        "messages": messages,
        "responses": responses
    }

    if tokenizer is None:
        return inputs

    message_inputs = tokenizer(
        messages, 
        max_length=max_context_length,
        padding="max_length",
        truncation=True
    )

    response_inputs = tokenizer(
        responses, 
        max_length=max_response_length,
        padding="max_length",
        truncation=True
    )

    inputs["input_ids"] = message_inputs["input_ids"]
    inputs["attention_mask"] = message_inputs["attention_mask"]
    inputs["y_input_ids"] = response_inputs["input_ids"]
    inputs["y_attention_mask"] = response_inputs["attention_mask"]

    return inputs


@Corpus.register_subclass("dailydialog")
class DailyDialogCorpus(Corpus):

    dataset = load_dataset("daily_dialog")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Rename the validation split of the dataset to valid
        self.dataset["valid"] = self.dataset["validation"]

    def get_dataset(
        self, split: str, 
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        **kwargs
    ):
        dataset = self.dataset[split]

        dataset = dataset.map(
            lambda x: map_fn(x, tokenizer=tokenizer, **kwargs), batched=True, batch_size=100, remove_columns=dataset.column_names
        )

        if split in ["test", "validation"] and tokenizer is not None:
            dataset = dataset.map(
                lambda x: add_candidates(x, dataset["responses"], tokenizer, **kwargs), batched=True, batch_size=100)

        return dataset
