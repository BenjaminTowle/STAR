import abc

from datasets import DatasetDict
from typing import List

from src.utils import RegistryMixin

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

            dataset = self.get_dataset(
                split=split, 
                tokenizer=kwargs["tokenizer"],
                max_context_length=kwargs["max_context_length"],
                max_response_length=kwargs["max_response_length"]
            )

            datasets[split] = dataset
        dataset_dict = DatasetDict(datasets)

        return dataset_dict
