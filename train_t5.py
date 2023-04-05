# TODO: rerun model now that kl divergence correctly implements temperature
# T5 model which takes message as input and outputs 3 replies.
# This is a differentiable search index, i.e. each reply is a string key (e.g. 'd345')
import jsonlines
import numpy as np
import os
import random
import torch.nn.functional as F
import pickle

from datasets import Dataset, concatenate_datasets
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    T5Config,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
    T5Model,
    TrainerState,
    EvalPrediction,
    TrainerControl,
    
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer
from torch import nn
import torch
from statistics import mean
from tqdm import tqdm
from scipy.stats import rankdata

from src.utils import set_random_seed
from eval import rouge, self_rouge

os.environ["WANDB_DISABLED"] = "true"
debug = False

tokenizer = T5Tokenizer.from_pretrained("t5-small")


set_random_seed(42)

class CustomT5(T5ForConditionalGeneration):


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        scores=None,
        idxs=None,
        returns: torch.Tensor = None,
        **kwargs,
    ):
        # Run the T5 model - must return decoder_hidden_states
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        if True:
            return outputs

        if scores is None or not self.training:
            return outputs

        kl_loss = 0.0
        tau = 0.1  # https://arxiv.org/pdf/1511.06295.pdf
        for i in range(scores.shape[1]):

            # Create probability vectors for knowledge distillation
            top_probs = F.softmax(scores[:, i, :] / tau, dim=1)
            target_probs = torch.zeros_like(outputs.logits[:, i, :])

            # Assign values in top_probs to zeros according to idxs_a
            target_probs.scatter_(1, idxs, top_probs)

            # Knowledge distillation loss
            kl_loss += F.kl_div((outputs.logits[:, i, :] / tau).log_softmax(dim=1), target_probs, reduction="batchmean") * tau * tau

        kl_loss /= scores.shape[1]
        outputs.loss = outputs.loss + kl_loss
        #outputs.loss = kl_loss # Only use knowledge distillation loss

        return outputs


class QueryEvalCallback(TrainerCallback):
    def __init__(self, messages, targets, restrict_decode_vocab, id2reply, args: TrainingArguments, tokenizer: T5Tokenizer):
        self.tokenizer = tokenizer
        self.args = args
        self.restrict_decode_vocab = restrict_decode_vocab

        self.targets = targets
        self.id2reply = id2reply

        self.messages = ["relevance: 0.99 diversity: 0.02 " + m for m in messages]

        self.best_rouge = 0.0
        self.best_self_rouge = 1.0
        self.rouge_is_converged = False
        self.self_rouge_is_converged = False

    def on_epoch_begin(self, args, state, control, **kwargs):

        model = kwargs['model'].eval()
        # Randomly sample 100 messages from the test set and evaluate the model on them.
        messages = random.sample(self.messages, 2)
        inputs = self.tokenizer.batch_encode_plus(messages, padding=True, return_tensors='pt')
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.generate(
            **inputs,
            max_length=4,
            min_length=4,
            prefix_allowed_tokens_fn=self.restrict_decode_vocab,
            do_sample=False,
            no_repeat_ngram_size=1,
        ).cpu().numpy().tolist()

        texts = [[self.id2reply[o] for o in output if o in self.id2reply] for output in outputs]
        idx = random.randint(0, len(texts) - 1)

        #print("Message: ", messages[idx])
        #print("Predicted replies: ", texts[idx])

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Generate predictions for every message in the test set
        model = kwargs['model'].eval()

        # Do it in batches to avoid OOM
        batch_size = 32
        outputs = []
        for i in range(0, len(self.messages), batch_size):
            inputs = self.tokenizer.batch_encode_plus(self.messages[i:i+batch_size], padding=True, return_tensors='pt')
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs.extend(model.generate(
                **inputs,
                max_length=4,
                min_length=4,
                prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                do_sample=False,
                no_repeat_ngram_size=1,
            ).cpu().numpy().tolist())
        
        preds = [[self.id2reply[o] for o in output if o in self.id2reply] for output in outputs]

        # Compute ROUGE scores
        rouge_scores = mean(rouge(self.targets, preds))
        self_rouge_scores = mean(self_rouge(preds))
        
        print("\n")
        print("ROUGE: ", rouge_scores)
        print("Self ROUGE: ", self_rouge_scores)

        # Save the model if it's the best so far
        if rouge_scores > self.best_rouge:
            self.best_rouge = rouge_scores
        else:
            self.rouge_is_converged = True

        if self_rouge_scores < self.best_self_rouge:  # Lower is better
            self.best_self_rouge = self_rouge_scores
        else:
            self.self_rouge_is_converged = True

        # Terminate training if rouge and self rouge have converged
        if self.rouge_is_converged and self.self_rouge_is_converged:
            control.should_training_stop = True
            print("Stopping training because ROUGE and self ROUGE have converged.")

def load_data(path, tokenizer, reply2id, debug=False):
    messages = []
    reply_sets = []
    targets = []
    scores = []
    idxs = []
    rge = []
    self_rge = []

    with jsonlines.open(path) as reader:
        for i, line in enumerate(reader):
            messages.append(line["observation"].replace("[SEP]", " "))
            targets.append(line["next_observation"])
            reply_sets.append(line["action"])
            scores.append(line["scores"])
            idxs.append([i + len(tokenizer) for i in line["doc_indices"]])
            rge.append(line["rouge"])
            self_rge.append(line["self_rouge"])

            if i >= 1000 and debug:
                break

    # Convert to percentile ranks
    rge = (rankdata(rge) / len(rge)).tolist()
    self_rge = (rankdata(self_rge) / len(self_rge)).tolist()

    # Convert to 2 dp
    rge = [round(r, 2) for r in rge]
    self_rge = [round(r, 2) for r in self_rge]

    # Tokenize messages and reply sets. Note T5 automatically adds <sos> and <eos> tokens.
    reply_sets = [[reply2id[reply] for reply in reply_set] for reply_set in reply_sets]
    messages_with_prompt = ["relevance: " + str(r) + " diversity: " + str(sr) + " message: " + message for r, sr, message in zip(rge, self_rge, messages)]
    message_inputs = tokenizer(messages_with_prompt, padding="max_length", truncation=True, max_length=72)

    #unique_replies = set([reply for reply_set in reply_sets for reply in reply_set])

    """
    # Set scores to -999 where idx is not in unique_replies
    for i, idx in enumerate(idxs):
        for j, id in enumerate(idx):
            if id not in unique_replies:
                for k in range(len(reply_sets[i])):
                    scores[i][k][j] = -999."""

    dataset = Dataset.from_dict(
        {
            "input_ids": message_inputs["input_ids"], 
            "attention_mask": message_inputs["attention_mask"], 
            "labels": reply_sets, 
            "scores": scores,
            "idxs": idxs,
        }
    )

    return dataset, messages_with_prompt, targets


def main():
    # Load vocabulary of replies
    replies = Dataset.load_from_disk("../data/personachat_reply_set")["responses"]

    # Create reply2id and id2reply dictionaries
    reply2id = {}
    id2reply = {}
    for i, reply in enumerate(replies):
        idx = i + len(tokenizer)
        reply2id[reply] = idx
        id2reply[idx] = reply

    # Load model
    model = CustomT5.from_pretrained("results-distillation2/checkpoint-22000").to("cuda")

    # Extend model embeddings to account for replies
    model.resize_token_embeddings(len(tokenizer) + len(replies))

    # Load data
    train_dataset, _, _ = load_data("distillation_pc_train_0-05.json_rewards", tokenizer, reply2id, debug=debug)
    valid_dataset, messages, targets = load_data("distillation_pc_valid_0-05.json_rewards", tokenizer, reply2id, debug=debug)

    # Compare number of unique replies for NLL and KL training
    if debug:
        R = train_dataset["labels"]
        R = [reply for reply_set in R for reply in reply_set]
        print("NLL: Number of unique replies in training set: ", len(set(R)))

        R = train_dataset["idxs"]
        R = [reply for reply_set in R for reply in reply_set]
        print("KLD: Number of unique replies in training set: ", len(set(R)))


    # Define function to restrict decoding vocabulary to replies
    INT_TOKEN_IDS = list(reply2id.values())
    INT_TOKEN_IDS.append(tokenizer.eos_token_id)

    def restrict_decode_vocab(batch_idx, prefix_beam):
        return INT_TOKEN_IDS
    
    # Initialise reply embeddings with by averaging over token embeddings
    reply_tokens = tokenizer(replies, truncation=True, max_length=64).input_ids

    # Go individually to avoid padding
    with torch.no_grad():
        for i in range(0, len(reply_tokens)):

            if debug:
                break
            
            tensor = torch.tensor(reply_tokens[i]).unsqueeze(0).to("cuda")
            input_embeddings = model.get_input_embeddings()(tensor).mean(dim=1)
            model.get_input_embeddings().weight.data[len(tokenizer) + i:len(tokenizer) + i + len(input_embeddings)] = input_embeddings

    # Using same training parameters as https://github.com/ArvinZhuang/DSI-transformers/blob/main/train.py

    training_args = TrainingArguments(
        output_dir="./results-pg",          # output directory
        learning_rate=0.0005,
        per_device_train_batch_size=128,  # batch size per device during training
        per_device_eval_batch_size=128,   # batch size for evaluation
        warmup_steps=1000,                # number of warmup steps for learning rate scheduler
        evaluation_strategy="steps",
        eval_steps=2000,
        save_steps=2000,
        max_steps=100000,
        dataloader_drop_last=False,
        save_total_limit=5,
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,        # training dataset
        eval_dataset=valid_dataset,    # evaluation dataset
        callbacks=[QueryEvalCallback(messages, targets, restrict_decode_vocab, id2reply, training_args, tokenizer)],
    )

    trainer.train()


if __name__ == "__main__":
    main()
