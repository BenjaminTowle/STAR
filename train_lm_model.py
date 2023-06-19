# Training script for language modeling that provides bias weights for retrieval.

from dataclasses import dataclass, field
from datasets import set_caching_enabled
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
)

import os
import re

from src.corpora.corpus import Corpus
from src.utils import parse_args, set_random_seed

os.environ["WANDB_DISABLED"] = "true"
set_caching_enabled(False)

@dataclass
class Args:  
    # Training args
    output_dir: str = field(
        default= "../data/personachat-lm",
        metadata={"help": "Path to save model"}
    )
    
    data_dir: str = field(
        default="../data/personachat",
        metadata={"help": "Directory that unprocessed datasets are stored in"}
    )

    task: str = field(
        default="personachat", 
        metadata={"choices": ["personachat", "dailydialog", "reddit"], 
        "help": "Task to train on"}
    )
    
    model_path: str = field(
        default="microsoft/DialoGPT-small",
        metadata={"help": "Path to pretrained model"}
    )

    tokenizer_path: str = field(
        default="microsoft/DialoGPT-small",
        metadata={"help": "Path to pretrained tokenizer"}
    )

    num_train_epochs: int = field(
        default=3, metadata={"help": "Number of epochs to train for"}
    )

    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    
    learning_rate: float = 5e-5
    use_symmetric_loss: bool = True
    max_response_length: int = 32

    # General args
    seed: int = 0
    device: str = "cuda"
    debug: bool = field(default=False, metadata={"help": "Debug mode"})


def map_fn(batch, tokenizer, args):
    inputs = tokenizer(
        batch["responses"], max_length=args.max_response_length, 
        padding="max_length", truncation=True
    )

    input_ids = inputs["input_ids"]
    labels = inputs["input_ids"]

    return {
        "input_ids": input_ids,
        "labels": labels
    }


def main():
    args = parse_args(Args)
    set_random_seed(args.seed)

    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    corpus = Corpus.create(args.task)
    dataset_dict = corpus.get_dataset_dict(splits=["train", "valid"])

    # Remove prefix from replies
    dataset_dict = dataset_dict.map(
        lambda x: {"responses": re.sub(r"^reply: ", "", x["responses"])}, batched=False
    )

    # Tokenize responses
    dataset_dict = dataset_dict.map(
        lambda x: map_fn(x, tokenizer, args),
        batched=True, batch_size=1000
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        num_train_epochs=args.num_train_epochs,
        save_steps=2000,
        save_total_limit=1,
        evaluation_strategy="steps",
        eval_steps=2000,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        run_name="dailydialog-lm",
        seed=args.seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["valid"],
    )

    trainer.train()

    # Save model
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
