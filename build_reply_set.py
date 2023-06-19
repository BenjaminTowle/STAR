import logging
import torch
import re

from dataclasses import dataclass
from datasets import set_caching_enabled
from functools import partial
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from src.corpora.corpus import Corpus
from src.utils import parse_args, set_random_seed

from constants import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class Args:
    task: str = "personachat"
    model_path: str = PERSONACHAT_MCVAE
    tokenizer_path: str = "distilbert-base-uncased"
    dataset_save_path: str = "../data/prefix_personachat_index"

    max_response_length: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0

    lm_model_path: str = "../data/personachat-lm"
    lm_tokenizer_path: str = "microsoft/DialoGPT-small"

torch.set_grad_enabled(False)
set_caching_enabled(False)


@torch.no_grad()
def _map_fn(samples, tokenizer, model, device):
    responses = [tokenizer.bos_token + r.replace("\n", "") for r in samples["text"]]
    length = tokenizer(
        responses,
        max_length=32,
        truncation=True,
        return_length=True
    ).length
    inputs = tokenizer(
        responses,
        max_length=32,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    
    inputs = {key: val.to(device) for key, val in inputs.items()}
    

    # Shift so that tokens < n predict n

    lm_logits = model(**inputs).logits
    labels = inputs["input_ids"]
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.reshape([-1, shift_logits.size(1)])  # bsz x words
    lm_score = [lss[:l].sum().cpu().numpy().item() * -1 for lss, l in zip(loss, length)]
    samples["lm_score"] = lm_score

    return samples

def main():
    args = parse_args(Args)
    set_random_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = AutoModel.from_pretrained(args.model_path).to(args.device)
    model.eval()

    lm_tokenizer = AutoTokenizer.from_pretrained(args.lm_tokenizer_path)
    lm_tokenizer.pad_token = lm_tokenizer.eos_token
    lm_model = AutoModelForCausalLM.from_pretrained(args.lm_model_path).to(args.device)
    lm_model.eval()

    logger.info("Building dataset")

    corpus = Corpus.create(args.task)
    dataset = corpus.get_dataset(
        split="train", 
        tokenizer=tokenizer, 
        max_response_length=args.max_response_length
    )

    # Rename responses column to text
    dataset = dataset.rename_column("responses", "text")

    logger.info("Preprocessing replies")

    # Convert to lower case
    dataset = dataset.map(
        lambda x: {"text": x["text"].lower()}, batched=False
    )

    # Remove all duplicate examples
    unique_texts = set()
    def remove_duplicates(example):
        if example["text"] in unique_texts:
            return False
        else:
            unique_texts.add(example["text"])
            return True
    dataset = dataset.filter(remove_duplicates)

    logger.info(f"Number of unique replies: {len(dataset)}")

    logger.info("Obtaining embeddings for replies")

    # Obtain embeddings for text
    def embed_fn(batch):
        batch["embeddings"] = model(
            torch.tensor(batch["y_input_ids"]).to(args.device), 
            attention_mask=torch.tensor(batch["y_attention_mask"]).to(args.device)
        )[0][:, 0, :].cpu().numpy()

        return batch

    dataset = dataset.map(embed_fn, batched=True, batch_size=100)

    # Remove prefix from replies
    dataset = dataset.map(
        lambda x: {"text": re.sub(r"^reply: ", "", x["text"])}, batched=False
    )

    dataset = dataset.map(
        partial(_map_fn, tokenizer=lm_tokenizer, model=lm_model, device=args.device),
        batched=True,
        batch_size=100
    )

    # Remove all other columns except text and embeddings
    keep_cols = ["embeddings", "text", "lm_score"]
    cols_to_remove = [col for col in dataset.column_names if col not in keep_cols]
    dataset = dataset.remove_columns(cols_to_remove)

    dataset.save_to_disk(args.dataset_save_path)

if __name__ == "__main__":
    main()
