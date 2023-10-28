import jsonlines
import numpy as np
import os
import pickle
import re

from dataclasses import dataclass, field
from datasets import Dataset, set_caching_enabled
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    AutoTokenizer   
)
from transformers.utils import logging
from transformers.trainer import Trainer
import torch
from statistics import mean


from src.utils import set_random_seed, parse_args
from src.simulation.metrics import rouge, self_rouge

os.environ["WANDB_DISABLED"] = "true"
set_caching_enabled(False)
logger = logging.get_logger("transformers")


@dataclass
class Args:
    # Training args
    output_dir: str = field(
        default= "../data/prefix-personachat-star",
        metadata={"help": "Path to save model"}
    )

    train_data_path: str = field(
        default="../data/prefix-personachat-train2.jsonl",
        metadata={"help": "JSONL file containing training data"}
    )

    valid_data_path: str = field(
        default="../data/prefix-personachat-valid2.jsonl",
        metadata={"help": "JSONL file containing validation data"}
    )

    model_path: str = field(
        default="t5-small",
        metadata={"help": "Path to pretrained model"}
    )
    tokenizer_path: str = field(
        default="t5-small",
        metadata={"help": "Path to pretrained tokenizer"}
    )

    id2reply_path: str = field(
        default="id2reply.pkl",
        metadata={"help": "Path to save id2reply dictionary"}
    )

    embedding_type: str = field(
        default="bow",
        metadata={"choices": ["random", "bow"]}
    )

    tokenizer_type: str = field(
        default="atomic",
        metadata={"choices": ["set", "atomic", "split"], 
        "help": "Type of tokenization to use"}
    )

    max_context_length: int = 64
    learning_rate: float = 0.0005
    per_device_train_batch_size: int = 128
    per_device_eval_batch_size: int = 128
    warmup_steps: int = 1000
    eval_steps: int = 2000
    save_steps: int = 2000
    max_steps: int = 100000
    save_total_limit: int = 1

    # General args
    seed: int = 0
    device: str = "cuda"
    debug: bool = field(default=False, metadata={"help": "Debug mode"})


class RougeEvalCallback(TrainerCallback):
    def __init__(
        self, 
        messages, 
        targets, 
        restrict_decode_vocab, 
        id2reply, 
        args: TrainingArguments, 
        tokenizer: T5Tokenizer,
        tokenizer_type: str,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type
        self.args = args
        self.restrict_decode_vocab = restrict_decode_vocab

        self.targets = targets
        self.id2reply = id2reply

        self.messages = messages

        self.best_rouge = 0.0
        self.best_self_rouge = 1.0
        self.rouge_is_converged = False
        self.self_rouge_is_converged = False


    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Generate predictions for every message in the test set
        model = kwargs['model'].eval()

        # Do it in batches to avoid OOM
        batch_size = 32
        outputs = []
        length = 4 if self.tokenizer_type == "atomic" else 2
        if self.tokenizer_type != "split":
            for i in range(0, len(self.messages), batch_size):
                inputs = self.tokenizer.batch_encode_plus(self.messages[i:i+batch_size], padding=True, return_tensors='pt')
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                outputs.extend(model.generate(
                    **inputs,
                    max_length=length,
                    min_length=length,
                    prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                    do_sample=False,
                    no_repeat_ngram_size=1,
                ).cpu().numpy().tolist())
            

            if self.tokenizer_type == "atomic":
                preds = [[self.id2reply[o] for o in output if o in self.id2reply] for output in outputs]
            else:
                preds = [list(self.id2reply[output[-1]]) for output in outputs]
        else:
            for i in range(0, len(self.messages), batch_size):
                inputs = self.tokenizer.batch_encode_plus(self.messages[i:i+batch_size], padding=True, return_tensors='pt')
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                outputs.extend(model.generate(
                    **inputs,
                    max_length=length,
                    min_length=length,
                    prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                    do_sample=False,
                    no_repeat_ngram_size=1,
                    num_beams=3,
                    num_return_sequences=3,
                ).cpu().numpy().tolist())

            preds = [self.id2reply[output[-1]] for output in outputs]

            # Unflatten preds
            preds = np.array(preds).reshape(-1, 3).tolist()

        # Compute ROUGE scores
        rouge_scores = mean(rouge(self.targets, preds))
        self_rouge_scores = mean(self_rouge(preds))
        
        logger.info("\n")
        logger.info(f"ROUGE: {rouge_scores}")
        logger.info(f"Self ROUGE: {self_rouge_scores}")

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
            logger.info("Stopping training because ROUGE and self ROUGE have converged.")


def load_data(path, debug=False):
    messages = []
    reply_sets = []
    targets = []

    with jsonlines.open(path) as reader:
        for i, line in enumerate(reader):
            messages.append(line["observation"].replace("[SEP]", " "))
            targets.append(line["next_observation"])
            reply_sets.append(line["action"])

            if i >= 1000 and debug:
                break

    messages_with_prompt = ["message: " + message for message in messages]

    return messages_with_prompt, reply_sets, targets

def build_vocab_atomic(train_reply_sets, valid_reply_sets, tokenizer):
    # Build vocabulary of replies
    reply2id = {}
    for reply_set in train_reply_sets + valid_reply_sets:
        for reply in reply_set:
            if reply not in reply2id:
                reply2id[reply] = len(reply2id) + len(tokenizer)
    id2reply = {v: k for k, v in reply2id.items()}

    # Get unique ids of replies in train_reply_sets
    # This is used to constrain the decoder to only generate replies in the train set
    unique_ids = set()
    for reply_set in train_reply_sets:
        for reply in reply_set:
            unique_ids.add(reply2id[reply])

    return reply2id, id2reply, unique_ids

def build_vocab_set(train_reply_sets, valid_reply_sets, tokenizer):
    # Build vocabulary of reply sets
    reply2id = {}
    for reply_set in train_reply_sets + valid_reply_sets:
        # We sort the reply sets to ensure that the same reply set always has the same id
        R = tuple(sorted(reply_set))
        if R not in reply2id:
            reply2id[R] = len(reply2id) + len(tokenizer)
    id2reply = {v: k for k, v in reply2id.items()}

    # Get unique ids of replies in train_reply_sets
    # This is used to constrain the decoder to only generate replies in the train set
    unique_ids = set()
    for reply_set in train_reply_sets:
        R = tuple(sorted(reply_set))
        unique_ids.add(reply2id[R])

    return reply2id, id2reply, unique_ids

def build_dataset_atomic(messages, reply_sets, tokenizer, reply2id, args):

    # Tokenize messages and reply sets. Note T5 automatically adds <sos> and <eos> tokens.
    reply_sets = [[reply2id[reply] for reply in reply_set] for reply_set in reply_sets]
    message_inputs = tokenizer(messages, padding="max_length", truncation=True, max_length=args.max_context_length)

    dataset = Dataset.from_dict(
        {
            "input_ids": message_inputs["input_ids"], 
            "attention_mask": message_inputs["attention_mask"], 
            "labels": reply_sets, 
        }
    )

    return dataset

def build_dataset_set(messages, reply_sets, tokenizer, reply2id, args):

    # Tokenize messages and reply sets. Note T5 automatically adds <sos> and <eos> tokens.
    reply_sets = [[reply2id[tuple(sorted(reply_set))]] for reply_set in reply_sets]
    message_inputs = tokenizer(messages, padding="max_length", truncation=True, max_length=args.max_context_length)

    dataset = Dataset.from_dict(
        {
            "input_ids": message_inputs["input_ids"], 
            "attention_mask": message_inputs["attention_mask"], 
            "labels": reply_sets, 
        }
    )

    return dataset

def build_dataset_split(messages, reply_sets, tokenizer, reply2id, args):

    # We separate each message, replt set pair into multiple examples
    k = len(reply_sets[0])
    reply_sets = [[reply2id[reply]] for reply_set in reply_sets for reply in reply_set]
    messages = [message for message in messages for _ in range(k)]

    message_inputs = tokenizer(messages, padding="max_length", truncation=True, max_length=args.max_context_length)

    dataset = Dataset.from_dict(
        {
            "input_ids": message_inputs["input_ids"], 
            "attention_mask": message_inputs["attention_mask"], 
            "labels": reply_sets, 
        }
    )

    return dataset

def initialise_model_atomic(args, tokenizer, reply2id):
    # Load model
    model = T5ForConditionalGeneration.from_pretrained(args.model_path).to(args.device)

    model.resize_token_embeddings(len(tokenizer) + len(reply2id))

    # ordered list of replies from reply2id
    replies = [reply for reply, _ in sorted(reply2id.items(), key=lambda item: item[1])]

    # Initialise reply embeddings with by averaging over token embeddings
    reply_tokens = tokenizer(replies, truncation=True, max_length=args.max_context_length).input_ids

    if args.embedding_type == "random":
        return model

    # Go individually to avoid padding
    with torch.no_grad():
        for i in range(0, len(reply_tokens)):

            if args.debug:
                break
            
            tensor = torch.tensor(reply_tokens[i]).unsqueeze(0).to("cuda")
            input_embeddings = model.get_input_embeddings()(tensor).mean(dim=1)
            model.get_output_embeddings().weight.data[len(tokenizer) + i:len(tokenizer) + i + len(input_embeddings)] = input_embeddings

    return model


def initialise_model_set(args, tokenizer, reply2id):
    # Load model
    model = T5ForConditionalGeneration.from_pretrained(args.model_path).to(args.device)

    model.resize_token_embeddings(len(tokenizer) + len(reply2id))

    # ordered list of replies from reply2id
    replies = [" ".join(reply) for reply, _ in sorted(reply2id.items(), key=lambda item: item[1])]

    # Initialise reply embeddings with by averaging over token embeddings
    reply_tokens = tokenizer(replies, truncation=True, max_length=args.max_context_length).input_ids

    if args.embedding_type == "random":
        return model

    # Go individually to avoid padding
    with torch.no_grad():
        for i in range(0, len(reply_tokens)):

            if args.debug:
                break
            
            tensor = torch.tensor(reply_tokens[i]).unsqueeze(0).to("cuda")
            input_embeddings = model.get_input_embeddings()(tensor).mean(dim=1)
            model.get_output_embeddings().weight.data[len(tokenizer) + i:len(tokenizer) + i + len(input_embeddings)] = input_embeddings

    return model


def main():
    args = parse_args(Args)
    set_random_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Load data
    train_messages, train_reply_sets, _ = load_data(args.train_data_path, debug=args.debug)
    valid_messages, valid_reply_sets, valid_targets = load_data(args.valid_data_path, debug=args.debug)

    if args.tokenizer_type == "atomic":

        # Build vocabulary of replies
        reply2id, id2reply, unique_ids = build_vocab_atomic(train_reply_sets, valid_reply_sets, tokenizer)

        # make dir at args.output_dir if it doesn't already exist
        os.makedirs(args.output_dir, exist_ok=True)
        pickle.dump(id2reply, open(os.path.join(args.output_dir, "id2reply.pkl"), "wb"))

        # Build datasets
        train_dataset = build_dataset_atomic(train_messages, train_reply_sets, tokenizer, reply2id, args)
        valid_dataset = build_dataset_atomic(valid_messages, valid_reply_sets, tokenizer, reply2id, args)

        # Initialise model
        model = initialise_model_atomic(args, tokenizer, reply2id)

    elif args.tokenizer_type == "set":

        # Build vocabulary of replies
        reply2id, id2reply, unique_ids = build_vocab_set(train_reply_sets, valid_reply_sets, tokenizer)

        # make dir at args.output_dir if it doesn't already exist
        os.makedirs(args.output_dir, exist_ok=True)
        pickle.dump(id2reply, open(os.path.join(args.output_dir, "id2reply.pkl"), "wb"))

        # Build datasets
        train_dataset = build_dataset_set(train_messages, train_reply_sets, tokenizer, reply2id, args)
        valid_dataset = build_dataset_set(valid_messages, valid_reply_sets, tokenizer, reply2id, args)

        # Initialise model
        model = initialise_model_set(args, tokenizer, reply2id)

    elif args.tokenizer_type == "split":

        # Build vocabulary of replies
        reply2id, id2reply, unique_ids = build_vocab_atomic(train_reply_sets, valid_reply_sets, tokenizer)

        # make dir at args.output_dir if it doesn't already exist
        os.makedirs(args.output_dir, exist_ok=True)
        pickle.dump(id2reply, open(os.path.join(args.output_dir, "id2reply.pkl"), "wb"))

        # Build datasets
        train_dataset = build_dataset_split(train_messages, train_reply_sets, tokenizer, reply2id, args)
        valid_dataset = build_dataset_split(valid_messages, valid_reply_sets, tokenizer, reply2id, args)

        # Initialise model
        model = initialise_model_set(args, tokenizer, reply2id)

    else:
        raise ValueError("Invalid tokenizer type")

    # Define function to restrict decoding vocabulary to replies
    INT_TOKEN_IDS = list(unique_ids)
    INT_TOKEN_IDS.append(tokenizer.eos_token_id)

    def restrict_decode_vocab(batch_idx, prefix_beam):
        return INT_TOKEN_IDS
    
    # Using same training parameters as https://github.com/ArvinZhuang/DSI-transformers/blob/main/train.py
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        dataloader_drop_last=False,
        save_total_limit=args.save_total_limit,
    )

    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[RougeEvalCallback(valid_messages, valid_targets, restrict_decode_vocab, id2reply, training_args, tokenizer, args.tokenizer_type)],
    )

    trainer.train()

    # Save model
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
