from dataclasses import dataclass, field
from datasets import set_caching_enabled
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Trainer,
    TrainingArguments,
)

import os

from src.corpora.corpus import Corpus
from src.utils import parse_args, set_random_seed

os.environ["WANDB_DISABLED"] = "true"
set_caching_enabled(False)

@dataclass
class Args:  
    # Training args
    output_dir: str = field(
        default= "../data/t5-reddit-seq2seq",
        metadata={"help": "Path to save model"}
    )
    
    data_dir: str = field(
        default="../data/personachat",
        metadata={"help": "Directory that unprocessed datasets are stored in"}
    )

    task: str = field(
        default="reddit", 
        metadata={"choices": ["personachat", "dailydialog", "reddit"], 
        "help": "Task to train on"}
    )
    
    model_path: str = field(
        default="t5-small",
        metadata={"help": "Path to pretrained model"}
    )

    tokenizer_path: str = field(
        default="t5-small",
        metadata={"help": "Path to pretrained tokenizer"}
    )

    max_steps: int = field(
        default=10000, metadata={"help": "Max steps to train for"}
    )

    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    
    learning_rate: float = 5e-5
    use_symmetric_loss: bool = True
    max_context_length: int = 64
    max_response_length: int = 32
    max_turns: int = 1

    # General args
    seed: int = 0
    device: str = "cuda"
    debug: bool = field(default=False, metadata={"help": "Debug mode"})


def map_fn(batch, tokenizer, args):
    encoder_inputs = tokenizer(
        batch["messages"], max_length=args.max_context_length, 
        padding="max_length", truncation=True
    )
    decoder_inputs = tokenizer(
        batch["responses"], max_length=args.max_response_length, 
        padding="max_length", truncation=True
    )

    input_ids = encoder_inputs["input_ids"]
    attention_mask = encoder_inputs["attention_mask"]
    labels = decoder_inputs["input_ids"]
    decoder_attention_mask = decoder_inputs["attention_mask"]

    # Set labels to -100 where decoder_attention_mask is 0
    """
    labels = [
        [-100 if mask == 0 else label for mask, label in zip(mask, label)]
        for mask, label in zip(decoder_attention_mask, labels)
    ]"""

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def main():
    args = parse_args(Args)
    set_random_seed(args.seed)

    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    tokenizer = T5TokenizerFast.from_pretrained(args.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    corpus = Corpus.create(args.task)
    dataset_dict = corpus.get_dataset_dict(splits=["train", "valid"])

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
        max_steps=args.max_steps,
        save_steps=2000,
        save_total_limit=1,
        evaluation_strategy="steps",
        eval_steps=2000,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        run_name="t5-personachat",
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
