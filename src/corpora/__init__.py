from src.corpora import (
    dailydialog,
    reddit, 
    simulation,
    scoring_fn,
    personachat
)
from .corpus import Corpus

def get_dataset(args, tokenizer):

    dataset_fns = {
        "daily_dialog": dailydialog.get_dataset,
        "personachat": personachat.get_dataset,
        "reddit": reddit.get_dataset,
        "simulation": simulation.get_dataset,
        "scoring_fn": scoring_fn.get_dataset
    }

    if args.task in dataset_fns:
        dataset_fn = dataset_fns[args.task]
    else:
        raise ValueError(f"Task: {args.task} not recognised")

    return dataset_fn(args, tokenizer)
