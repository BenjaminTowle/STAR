import numpy as np
import os
import re

from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from typing import Optional

from src.corpora.corpus import Corpus


split2file = {
        "train": "train_both_original.txt",
        "valid": "valid_both_original.txt",
        "test": "test_both_original.txt"
    }


def get_contexts_persona_responses(f, split, max_turns=1, debug=False, use_past_contexts=False):
    persona_a = []
    personae_a = []
    persona_b = []
    personae_b = []

    dialog = []
    contexts = []
    responses = []
    persona = []
    candidates = []

    reading_persona = True
    lines = f.readlines()
    for line in lines:
        if "your persona:" in line:
            if reading_persona is False:
                personae_a.append(persona_a)
                personae_b.append(persona_b)
                persona_a = []
                persona_b = []
                dialog = []
                reading_persona = True
            persona_a.append(re.sub(r"\A[0-9]+ your persona: ", "", line).replace("\n", ""))
        elif "partner's persona:" in line:
            persona_b.append(re.sub(r"\A[0-9]+", "", line))
        else:
            # utterance line is split into speaker A + \t + speaker B + \t\t + candidate_1|candidate_2 etc.
            utts = line.split("\t")
            c = utts[3].replace("\n", "").split("|") if split != "train" else None  # No candidates during training
            context = re.sub(r"\A[0-9]+ ", "", utts[0])  # remove line numbering
            response = utts[1]
            dialog.append(context)
            if use_past_contexts:
                if len(dialog) >= 2:
                    contexts.append(dialog[-2])
                    responses.append(response)
                    persona.append(persona_a)
                    dialog.append(response)  #  MUST BE ADDED AFTER THE CONTEXT!

                    if split != "train":
                        # Ground truth contained in last position
                        candidates.append(c)
                    reading_persona = False
            else:
                contexts.append(dialog[-min(max_turns, len(dialog)):]) if max_turns != -1 else contexts.append(dialog[:])
                responses.append(response)
                persona.append(persona_a)
                dialog.append(response)  #  MUST BE ADDED AFTER THE CONTEXT!

                if split != "train":
                    # Ground truth contained in last position
                    candidates.append(c)
                reading_persona = False

        if not debug:
            continue

        if len(contexts) > 100:
            break


    return contexts, persona, responses, candidates

@Corpus.register_subclass("personachat")
class PersonaChatCorpus(Corpus):

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
        """
        Load dataset from Persona Chat paper: https://arxiv.org/abs/1801.07243
        :return: list of contexts, list of responses, list of personae
        """

        with open(os.path.join(data_dir, split2file[split])) as f:
            contexts, persona, responses, candidates = get_contexts_persona_responses(f, split, max_turns=max_turns, debug=debug)

        # Dict to be fed to dataset
        inputs = {}

        # Tokenize
        contexts = [" ".join(C) for C in contexts]
        persona = ["".join(P) for P in persona]
        augmented_contexts = [" ".join([p, c]) for p, c in zip(persona, contexts)]
        inputs["messages"] = augmented_contexts
        inputs["responses"] = responses

        # Skip tokenization if tokenizer is None
        if tokenizer is None:
            return Dataset.from_dict(inputs)
        
        input_ids = tokenizer(augmented_contexts, padding="max_length", max_length=max_context_length, truncation=True).input_ids
        inputs["input_ids"] = input_ids
        
        
        if split == "train":
            inputs["y_input_ids"] = tokenizer(responses, padding="max_length", max_length=max_response_length, truncation=True).input_ids
        else:
            inputs["labels"] = [19 for _ in range(len(contexts))]

            candidates = np.array(candidates).reshape(-1).tolist()
            candidate_ids = tokenizer(candidates, padding="max_length", max_length=max_response_length, truncation=True, return_tensors="np").input_ids
            inputs["candidate_input_ids"] = candidate_ids.reshape([len(contexts), -1, max_response_length])

        return Dataset.from_dict(inputs)
