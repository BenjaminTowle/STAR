import abc
import logging
import numpy as np
import torch

from datasets import Dataset
from typing import Optional

from .agent_utils import AgentBatchOutput
from ..utils import RegistryMixin

logger = logging.getLogger(__name__)

class Index(RegistryMixin, abc.ABC):
    """
    Abstract wrapper for either a FAISS index or an in-memory index.
    """

    def __init__(
        self, 
        dataset: Dataset,  
        device="cuda",
        text_column: str = "text",
        embeddings_column: str = "embeddings",
    ):
        self.dataset = dataset
        self.device = device

        # Check columns exist in dataset
        assert text_column in self.dataset.column_names, \
            f"Column {text_column} not found in dataset"
        assert embeddings_column in self.dataset.column_names, \
            f"Column {embeddings_column} not found in dataset"

        self.text_column = text_column
        self.embeddings_column = embeddings_column

    @abc.abstractmethod
    def search(
        self, 
        query_embed: torch.Tensor, 
        k: int = 3,
        output_scores: bool = True,
        output_embeddings: bool = False
    ):
        pass


@Index.register_subclass("in_memory")
class InMemoryIndex(Index):
    """
    In-memory index.
    """
    def __init__(
        self, 
        *args, 
        lm_bias_column: Optional[str] = None, 
        lm_bias_alpha: float = 0.5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lm_bias_column = lm_bias_column
        self.lm_bias_alpha = lm_bias_alpha
        if lm_bias_column is not None:
            assert lm_bias_column in self.dataset.column_names, \
                f"Column {lm_bias_column} not found in dataset"
            self.lm_bias = torch.tensor(self.dataset[lm_bias_column]).to(self.device) * lm_bias_alpha
        else:
            self.lm_bias = None
        self.embeddings = torch.tensor(self.dataset[self.embeddings_column]).to(self.device)
        self.texts = self.dataset[self.text_column]

    def find_rank(
        self, 
        query_embed: torch.Tensor,
        text: str,
    ):

        """
        Gets the ranking of a particular text in the index.
        """
        scores = torch.matmul(query_embed, self.embeddings.T)
        if self.lm_bias is not None:
            scores += self.lm_bias
        scores = scores.cpu().numpy()

        # Sort scores in descending order
        sorted_idxs = np.argsort(scores, axis=-1)[0][::-1]

        # Get rank of text
        rank = np.where(sorted_idxs == self.texts.index(text))[0][0]

        return rank

    def get_embedding(self, text: str):
        """
        Gets the embedding of a particular text in the index.
        """
        idx = self.texts.index(text)
        return self.embeddings[idx].cpu().numpy()

    def search(self, 
        query_embed: torch.Tensor, 
        k: int = 3,
        output_scores: bool = True,
        output_embeddings: bool = False
    ):
        """
        Returns top k responses for a given query embedding.
        Note: np.argpartition is faster than torch.topk
        """
        scores = torch.matmul(query_embed, self.embeddings.T)
        if self.lm_bias is not None:
            scores += self.lm_bias
        scores = scores.cpu().numpy()

        topk = np.argpartition(scores, -k, axis=-1)[:, -k:]

        docs = [[self.texts[idx] for idx in idxs] for idxs in topk]

        doc_embeds = None
        if output_embeddings:
            doc_embeds = [[self.embeddings[idx].cpu().numpy() for idx in idxs] for idxs in topk]
        
        doc_scores = None
        if output_scores:
            doc_scores = [[scores[i, idx] for idx in idxs] for i, idxs in enumerate(topk)]

        return AgentBatchOutput(
            docs=docs,
            doc_embeds=doc_embeds,
            doc_scores=doc_scores,
            doc_indices=topk,
        )


@Index.register_subclass("faiss")
class FAISSIndex(Index):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset.add_faiss_index(
            self.embeddings_column
        )

    def search(self, query_embed: torch.Tensor, k: int, output_embeddings: bool = False):
        """
        Returns top k responses for a given query embedding.
        """
        query_embed = query_embed.cpu().numpy()
        scores, retrieved_examples = self.dataset.get_nearest_examples_batch(
            self.embeddings_column, query_embed, k=k
        )
        docs = [s[self.text_column] for s in retrieved_examples]

        doc_embeds = None
        if output_embeddings:
            doc_embeds = [s[self.embeddings_column] for s in retrieved_examples]
        
        return AgentBatchOutput(
            docs=docs,
            doc_embeds=doc_embeds, 
            doc_scores=scores,
            doc_indices=None,
        )
