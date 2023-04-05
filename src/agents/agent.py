import jsonlines
import torch

from abc import ABC, abstractmethod
from datasets import Dataset
from typing import List, Optional, Union
from transformers import pipeline, FeatureExtractionPipeline, T5ForConditionalGeneration, T5TokenizerFast

from ..agents.index import Index
from ..timer import timer
from ..utils import RegistryMixin
from .agent_utils import AgentBatchOutput
from .diversity import DiversityStrategy


class Agent(RegistryMixin, ABC):
    subclasses = {}

    @abstractmethod
    def batch_act(self, queries: List[str]) -> AgentBatchOutput:
        pass


@Agent.register_subclass("retrieval")
class RetrievalAgent(Agent):

    def __init__(
        self,
        model: Optional[FeatureExtractionPipeline] = None,
        model_path: str = "facebook/dpr-question_encoder-multiset-base",
        k: int = 3,
        device: str = "cuda",
        index: Union[str, Index] = "faiss",
        response_set: Optional[Union[str, Dataset]] = None,
        diversity_strategy: Optional[Union[str, DiversityStrategy]] = None,
    ):

        self.k = k  # Must pass to init rather than batch_act as we need to maintain liskov substitution principle.
        
        # Load model which is a transformer pipeline
        if model is None:
            assert model_path is not None, "Must provide model path if model is not provided."
            model = pipeline(
                "feature-extraction", 
                model=model_path,  
                device=0 if device == "cuda" else -1
            )
        self.model = model
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
                embedding_column="dummy"
            )
        self.index = index

        # Load diversity strategy
        if isinstance(diversity_strategy, str):
            diversity_strategy = DiversityStrategy.create(diversity_strategy)
        self.diversity_strategy = diversity_strategy

    def batch_act(self, queries: List[str]):

        # Generate initial outputs for reranking
        with timer.lap("batch_act"):

            query_embed = self.model(
                queries, truncation=True, max_length=64, padding="max_length", return_tensors=True
            )
            query_embed = torch.cat([Q[:, 0, :] for Q in query_embed], dim=0).to(self.model.device) # take CLS token

            outputs = self.index.search(query_embed, k=self.k)

        if self.diversity_strategy is None:
            return outputs

        outputs = self.diversity_strategy.rerank(outputs, self.k)

        return outputs


@Agent.register_subclass("star")
class STARAgent(Agent):
    def __init__(
        self, 
        model, 
        tokenizer, 
        restrict_decode_vocab, 
        id2reply, 
        device,
        k: int = 3,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.restrict_decode_vocab = restrict_decode_vocab
        self.id2reply = id2reply
        self.device = device
        self.k = k

    def batch_act(self, queries: List[str]):
        with timer.lap("batch_act"):
            queries = ["relevance: 0.01 diversity 0.99 message: " + query for query in queries]
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
                max_length=(self.k+1),
                min_length=(self.k+1),
                prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                do_sample=False,
                no_repeat_ngram_size=1,
            ).cpu().numpy().tolist()

            texts = [[self.id2reply[o] for o in output if o in self.id2reply] for output in outputs]

        return AgentBatchOutput(docs=texts)


    @classmethod
    def from_dict(cls, config):

        
        tokenizer = T5TokenizerFast.from_pretrained("t5-small")

        # Build vocabulary of unique replies
        replies = Dataset.load_from_disk("../data/personachat_reply_set")["responses"]

        model = T5ForConditionalGeneration.from_pretrained(config["model_path"]).to(
            config["device"]
        )

        # Create reply2id and id2reply dictionaries
        reply2id = {}
        id2reply = {}
        for i, reply in enumerate(replies):
            idx = i + len(tokenizer)
            reply2id[reply] = idx
            id2reply[idx] = reply

        # Define function to restrict decoding vocabulary to replies
        reply_sets = []
        with jsonlines.open("distillation_pc.jsonl") as reader:
            for i, line in enumerate(reader):
                reply_sets.append(line["action"])
        
        # Get list of unique replies
        reply_sets = list(set([reply for reply_set in reply_sets for reply in reply_set]))
        # Get list of unique reply ids
        reply_ids = [reply2id[reply] for reply in reply_sets]
                
        # Define function to restrict decoding vocabulary to replies
        #INT_TOKEN_IDS = list(reply2id.values())
        INT_TOKEN_IDS = reply_ids
        INT_TOKEN_IDS.append(tokenizer.eos_token_id)

        def restrict_decode_vocab(batch_idx, prefix_beam):
            return INT_TOKEN_IDS

        return cls(
            model=model, 
            tokenizer=tokenizer,
            restrict_decode_vocab=restrict_decode_vocab,
            id2reply=id2reply, 
            device=config["device"]
        )
