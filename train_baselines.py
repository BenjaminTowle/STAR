import numpy as np
import os

from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer, 
    TrainingArguments, 
)
from dataclasses import dataclass, field
from datasets import set_caching_enabled, DatasetDict

from src.corpora.corpus import Corpus
from src.utils import set_random_seed
from src.modeling import get_model

from constants import *

os.environ["WANDB_DISABLED"] = "true"
set_caching_enabled(False)

@dataclass
class Args:  
    # Training args
    output_dir: str = field(
        default="prefix-matching-dailydialog-new",
        metadata={"help": "Path to save model"}
    )
    
    data_dir: str = field(
        default="../data/personachat",
        metadata={"help": "Directory that unprocessed datasets are stored in"}
    )

    dataset_save_path: str = field(
        default=None,
        metadata={"help": "Path to save dataset"}
    )

    dataset_load_path: str = field(
        default=None,
        metadata={"help": "Path to load dataset"}
    )

    task: str = field(
        default="dailydialog", 
        metadata={"choices": ["personachat", "dailydialog", "reddit", "offline"], 
        "help": "Task to train on"}
    )

    model_type: str = field(default="matching", 
        metadata={"choices": ["matching", "mcvae", "star"],
        "help": "Type of model to train"}
    )

    model_name: str = field(default="distilbert",
        metadata={"choices": ["t5", "bert", "distilbert"],
        "help": "Which base model architecture to use."}
    )
    
    model_path: str = field(
        default="distilbert-base-uncased",
        metadata={"help": "Path to pretrained model"}
    )

    tokenizer_path: str = field(
        default="distilbert-base-uncased",
        metadata={"help": "Path to pretrained tokenizer"}
    )

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    use_symmetric_loss: bool = True
    max_context_length: int = 64
    max_response_length: int = 32
    max_turns: int = 1

    # MCVAE args
    z: int = field(default=256, metadata={"help": "Size of latent vector for MCVAE model"})
    kld_weight: float = 0.05
    use_kld_annealling: bool = False
    kld_annealling_steps: int = -1
    use_message_prior: bool = False

    # General args
    seed: int = 0
    device: str = "cuda"
    debug: bool = field(default=False, metadata={"help": "Debug mode"})
    
def parse_args():
    parser = HfArgumentParser([Args])
    args, = parser.parse_args_into_dataclasses()
    return args

def accuracy_metric_matching(eval_preds):
    """Compute recall@k metric for a given set of predictions. Ground truth is assumed to be last index in predictions."""
    acc = (np.argmax(eval_preds.predictions, axis=-1) == (eval_preds.predictions.shape[-1] - 1)).mean().item()

    return {"accuracy": acc}

def _get_dataset(args, tokenizer):
    if args.dataset_load_path == None:
        corpus = Corpus.create(args.task)
        dataset_dict = corpus.get_dataset_dict(
            tokenizer=tokenizer,
            data_dir=args.data_dir,
            max_context_length=args.max_context_length,
            max_response_length=args.max_response_length,
        )
        if args.dataset_save_path != None:
            dataset_dict.save_to_disk(args.dataset_save_path)
        return dataset_dict
    
    dataset_dict = DatasetDict.load_from_disk(args.dataset_load_path)

    return dataset_dict


def main():
    args = parse_args()
    set_random_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    dataset_dict = _get_dataset(args, tokenizer)
    model = get_model(args)

    if args.output_dir != None:
        save_strategy = "epoch"
    else:
        save_strategy = "no"
    
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        save_strategy=save_strategy,
        evaluation_strategy="epoch",
        eval_steps=1,
        save_total_limit=5,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        disable_tqdm=True,
        dataloader_drop_last=True,
    )

    trainer = Trainer(
        args=training_arguments,
        model=model,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["valid"],
        compute_metrics=accuracy_metric_matching if args.model_type == "matching" else None,
    )

    trainer.train()

    # Save model
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
