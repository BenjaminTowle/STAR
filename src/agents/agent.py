import torch

from abc import ABC, abstractmethod
from datasets import Dataset
from typing import List, Optional, Union
from transformers import (
    T5ForConditionalGeneration, 
    T5TokenizerFast,
    PreTrainedTokenizerBase,
)

from ..agents.index import Index
from ..modeling.matching import MatchingMixin
from ..utils import RegistryMixin
from .agent_utils import AgentBatchOutput
from .diversity import DiversityStrategy


class Agent(RegistryMixin, ABC):
    subclasses = {}

    @abstractmethod
    @torch.no_grad()
    def batch_act(self, queries: List[str]) -> AgentBatchOutput:
        pass


@Agent.register_subclass("retrieval")
class RetrievalAgent(Agent):

    def __init__(
        self,
        model: MatchingMixin,
        tokenizer: PreTrainedTokenizerBase,
        k: int = 3,
        n: int = 3,
        device: str = "cuda",
        index: Union[str, Index] = "faiss",
        response_set: Optional[Union[str, Dataset]] = None,
        diversity_strategy: Optional[Union[str, DiversityStrategy]] = None,
        use_posterior: bool = False,
    ):

        self.k = k  # Must pass to init rather than batch_act as we need to maintain liskov substitution principle.
        self.n = n
        
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = model.device

        # Load index
        if isinstance(index, str):
            # Assert response set path is provided
            assert response_set is not None, "Must provide response set path if index is not provided."
            response_set = Dataset.load_from_disk("response_set_path")
            index = Index.create(
                index,
                dataset=response_set,
                tokenizer=self.model.tokenizer,
                model=self.model.model,
                device=device,
                lm_bias=True,
                lm_bias_column="lm_score",
            )
        self.index = index

        # Load diversity strategy
        if isinstance(diversity_strategy, str):
            diversity_strategy = DiversityStrategy.create(diversity_strategy)
        self.diversity_strategy = diversity_strategy

        # A dictionary to store the query to target mapping
        self.query2target = {}
        self.use_posterior = use_posterior

    def get_embedding(self, texts: List[str]):
        query_tokens = self.tokenizer(texts, truncation=True, max_length=64, padding="max_length", return_tensors="pt")
        # To cuda
        query_tokens = {k: v.to(self.model.device) for k, v in query_tokens.items()}
        query_embed = self.model.get_embedding(**query_tokens)

        return query_embed

    @torch.no_grad()
    def batch_act(self, queries: List[str]):

        # Generate initial outputs for reranking
        query_embed = self.get_embedding(queries)

        if self.use_posterior:
            targets = [self.query2target[query] for query in queries]
            target_embed = self.get_embedding(targets)

            # Create new query embed which interpolates between query and target
            query_weight = 0.75
            query_embed = query_embed * query_weight + (1- query_weight) * target_embed

        outputs = self.index.search(
            query_embed, 
            k=self.n,
            output_embeddings=self.diversity_strategy.requires_doc_embeds if self.diversity_strategy is not None else False,
        )
        outputs.query_embed = query_embed

        if self.diversity_strategy is None:
            assert self.n == self.k, "Must use diversity strategy if n != k"
            return outputs

        outputs = self.diversity_strategy.rerank(outputs, self.k)

        return outputs

@Agent.register_subclass("seq2seq")
class Seq2SeqAgent(Agent):
    def __init__(
        self,
        model: T5ForConditionalGeneration,
        tokenizer: T5TokenizerFast,
        device: str = "cuda",
        k: int = 3,
    ):
        super().__init__()
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.k = k

    @torch.no_grad()
    def batch_act(self, queries: List[str]):
        
        inputs = self.tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
    
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            max_length=32,
            do_sample=True,
            num_return_sequences=self.k,
        ).reshape(len(queries), self.k, -1)

        texts = [self.tokenizer.batch_decode(output, skip_special_tokens=True) for output in outputs]

        return AgentBatchOutput(docs=texts)

@Agent.register_subclass("star")
class STARAgent(Agent):
    def __init__(
        self, 
        model: T5ForConditionalGeneration, 
        tokenizer: T5TokenizerFast, 
        id2reply: dict, 
        device: str = "cuda",
        k: int = 3,
    ):
        super().__init__()
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.id2reply = id2reply
        self.device = device
        self.k = k

        # Define function to restrict decoding vocabulary to replies
        INT_TOKEN_IDS = list(id2reply.keys())
        INT_TOKEN_IDS.append(tokenizer.eos_token_id)

        def restrict_decode_vocab(batch_idx, prefix_beam):
            return INT_TOKEN_IDS

        self.restrict_decode_vocab = restrict_decode_vocab

    @torch.no_grad()
    def batch_act(self, queries: List[str]):
       
        queries = ["message: " + query.replace("[SEP]", "") for query in queries]
        inputs = self.tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
    
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_length=4,
            min_length=4,
            do_sample=False,
            no_repeat_ngram_size=1,
        ).cpu().numpy().tolist()

        texts = [[self.id2reply[o] for o in output[1:]] for output in outputs]

        return AgentBatchOutput(docs=texts)
