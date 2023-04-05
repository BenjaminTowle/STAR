import abc
import copy
import logging
import numpy as np
import torch

from contextlib import contextmanager
from datasets import Dataset
from typing import Optional, List

from .agent_utils import AgentBatchOutput
from ..utils import RegistryMixin

logger = logging.getLogger(__name__)

class Index(RegistryMixin, abc.ABC):
    """
    Abstract wrapper for either a FAISS index or an in-memory index.
    """

    def __init__(
        self, 
        dataset, 
        tokenizer, 
        model, 
        device="cuda",
        query_column = "responses",
        embedding_column = "embeddings",
        payload_column: Optional[str] = None
    ):
        self.device = device
        self.tokenizer = tokenizer
        self.model = model

        self.query_column = query_column
        self.embedding_column = embedding_column
        self.payload_column = payload_column if payload_column is not None else query_column

        self._index = self.build_index(dataset)
        self.mask_idxs = None


    @abc.abstractmethod
    def build_index(self, dataset: Dataset):
        pass

    @abc.abstractmethod
    def search(self, query_embed: torch.Tensor, k: int, lm_bias: bool = True):
        pass


@Index.register_subclass("in_memory")
class InMemoryIndex(Index):
    """
    In-memory index.
    """
    def __init__(self, *args, lm_bias: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_bias = lm_bias

    def _build_index(self, samples):
        """Map fn"""
        inputs = self.tokenizer(
            samples[self.query_column], max_length=32, padding="max_length", truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        samples[self.embedding_column] = self.model(**inputs).last_hidden_state[:, 0, :].detach().cpu().numpy()

        return samples

    @contextmanager
    def set_mask_idxs(self, targets_to_mask: List[str]):
        batch_idxs = []
        for target in targets_to_mask:
            sample_idxs = []
            for i, cand in enumerate(self._index[self.payload_column]):
                if target == cand:
                    sample_idxs.append(i)
            batch_idxs.append(sample_idxs)
        self.mask_idxs = batch_idxs
        yield
        self.mask_idxs = None
        
    @torch.no_grad()
    def build_index(
        self, 
        dataset: Dataset,
    ):

        if self.embedding_column not in dataset.column_names:
            # If no embeddings, must include query to be embedded
            assert self.query_column in dataset.column_names
            dataset = dataset.map(self._build_index, batched=True, batch_size=100)

        outputs = dict()
        outputs[self.embedding_column] = torch.tensor(dataset[self.embedding_column], device=self.device)
        outputs[self.payload_column] = dataset[self.payload_column]

        if "lm_score" in dataset.column_names:
            lm_alpha = 0.5
            outputs["lm_score"] = (torch.tensor(dataset["lm_score"]) * lm_alpha).to(self.device)
        else:
            logger.warn("index does not contain lm_score. This will cause an error later on if lm_bias is set to True.")

        return outputs

    def search(self, query_embed: torch.Tensor, k: int):
        """
        Returns top k responses for a given query embedding.
        Note: np.argpartition is faster than torch.topk
        """
        scores = torch.matmul(query_embed, self._index[self.embedding_column].T)
        if self.lm_bias:
            scores += self._index["lm_score"]
        scores = scores.cpu().numpy()

        if self.mask_idxs is not None:
            for i, idxs in enumerate(self.mask_idxs):
                scores[i, idxs] = -999.

        topk = np.argpartition(scores, -k, axis=-1)[:, -k:]

        return AgentBatchOutput(
            docs=[[self._index[self.payload_column][topk[i, j]] for j in range(topk.shape[1])] for i in range(topk.shape[0])],
            #doc_embeds=[[self.index[self.embedding_column][topk[i, j]].cpu().numpy() for j in range(topk.shape[1])] for i in range(topk.shape[0])],
            doc_scores=[np.array([scores[i, topk[i, j]] for j in range(topk.shape[1])]) for i in range(topk.shape[0])],
            doc_indices=topk,
        )


@Index.register_subclass("faiss")
class FAISSIndex(Index):

    def _build_index(self, samples):
        """Map fn"""
        inputs = self.tokenizer(
            samples[self.query_column], max_length=32, padding="max_length", truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        samples[self.embedding_column] = self.model(**inputs).last_hidden_state[:, 0, :].detach().cpu().numpy()
        return samples
        
    @torch.no_grad()
    def build_index(self, dataset: Dataset):

        if self.embedding_column not in dataset.column_names:
            # If no embeddings, must include query to be embedded
            assert self.query_column in dataset.column_names
            dataset = dataset.map(lambda x: self._build_index(x), batched=True, batch_size=100)
        dataset.add_faiss_index(column=self.embedding_column)
        return dataset

    def search(self, query_embed: torch.Tensor, k: int):
        """
        Returns top k responses for a given query embedding.
        """
        query_embed = query_embed.cpu().numpy()
        scores, retrieved_examples = self._index.get_nearest_examples_batch(self.embedding_column, query_embed, k=k)
        docs = [s[self.payload_column] for s in retrieved_examples]
        doc_embeds = [s[self.embedding_column] for s in retrieved_examples]
        
        return AgentBatchOutput(
            docs=docs,
            doc_embeds=doc_embeds, 
            doc_scores=scores,
            doc_indices=None,
        )
