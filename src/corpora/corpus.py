import abc

from datasets import DatasetDict
from typing import List

from ..utils import RegistryMixin


class Corpus(RegistryMixin, abc.ABC):

    subclasses = {}

    abc.abstractmethod
    def get_dataset(self, split: str, **kwargs):
        pass

    def get_dataset_dict(
        self,
        splits: List[str] = ["train", "valid", "test"],
        **kwargs
    ):
        datasets = {}
        for split in splits:

            tokenizer = kwargs.get("tokenizer", None)
            max_context_length = kwargs.get("max_context_length", 64)
            max_response_length = kwargs.get("max_response_length", 64)

            dataset = self.get_dataset(
                split=split, 
                tokenizer=tokenizer,
                max_context_length=max_context_length,
                max_response_length=max_response_length
            )

            datasets[split] = dataset
        dataset_dict = DatasetDict(datasets)

        return dataset_dict
