import jsonlines

from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from typing import Optional

from . import Corpus


split2file = {
    "train": "train.jsonl",
    "valid": "valid.jsonl",
    "test": "test.jsonl",
}

@Corpus.register_subclass("offline")
class Offline(Corpus):

    def get_dataset(
        self, 
        tokenizer: Optional[PreTrainedTokenizerBase] = None, 
        data_dir: str = "../data/personachat",
        max_context_length: int = 64,
        max_response_length: int = 32,
        max_turns: int = 1, # -1 for all
        debug: bool = False,
        split: str = "train",
    ):
        assert tokenizer is not None, "Tokenizer must be provided for offline dataset."
        
        messages = []
        reply_sets = []
        targets = []
        scores = []
        idxs = []
        rge = []
        self_rge = []

        with jsonlines.open(split2file[split]) as reader:
            for i, line in enumerate(reader):
                messages.append(line["observation"].replace("[SEP]", " "))
                targets.append(line["next_observation"])
                reply_sets.append(line["action"])
                scores.append(line["scores"])
                idxs.append([i + len(tokenizer) for i in line["doc_indices"]])
                rge.append(line["rouge"])
                self_rge.append(line["self_rouge"])

                if i >= 100 and debug:
                    break


        # Tokenize messages and reply sets. Note T5 automatically adds <sos> and <eos> tokens.
        reply_sets = [[reply2id[reply] for reply in reply_set] for reply_set in reply_sets]
        messages_with_prompt = ["relevance: " + str(r) + " diversity: " + str(sr) + " message: " + message for r, sr, message in zip(rge, self_rge, messages)]
        message_inputs = tokenizer(messages_with_prompt, padding="max_length", truncation=True, max_length=max_context_length)

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
