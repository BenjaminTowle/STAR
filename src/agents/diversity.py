import copy
import numpy as np
import ray
import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass
from scipy.special import softmax
from scipy.stats import zscore
from transformers import pipeline, FeatureExtractionPipeline

from ..timer import timer
from ..utils import RegistryMixin, compute_f1_matrix_fast
from .agent_utils import AgentBatchOutput


class DiversityStrategy(RegistryMixin, ABC):
    subclasses = {}
    @abstractmethod
    def rerank(self, outputs: AgentBatchOutput, k: int) -> AgentBatchOutput:
        pass

@DiversityStrategy.register_subclass("mmr")
class MMR(DiversityStrategy):

    def rerank(
        self, 
        outputs: AgentBatchOutput, 
        k: int
    ):
        doc_embeds = np.array(outputs.doc_embeds)
        # BNM, BNM -> BNN
        inter_doc_scores = np.einsum('BNM,BOP -> BNP', doc_embeds, np.transpose(doc_embeds, (0, 2, 1)))
        inter_doc_scores = zscore(inter_doc_scores, axis=2)
        doc_scores = np.array(outputs.doc_scores)
        doc_scores = zscore(doc_scores, axis=1)
        l = 0.5

        docs = outputs.docs
        batch_chosen_docs = []
        for i in range(len(docs)):
            chosen_idxs = []
            remaining_idxs = list(range(len(docs[i])))
            for k_ in range(k):
                if k_ == 0:
                    idx = np.argmax(doc_scores[i])
                    chosen_idxs.append(idx)
                    remaining_idxs.remove(idx)
                    continue

                best_idx, best_mmr = None, -99999.
                for idx in remaining_idxs:
                    mmr = (1 - l) * np.argsort(np.max(inter_doc_scores[i, idx])) - l * np.argsort(doc_scores[i, idx])
                    best_idx, best_mmr = max([(best_idx, best_mmr), (idx, mmr)], key=lambda x: x[1])

                chosen_idxs.append(best_idx)
                remaining_idxs.remove(best_idx)

            chosen_docs = [docs[i][idx] for idx in chosen_idxs]
            batch_chosen_docs.append(chosen_docs)

        return AgentBatchOutput(docs=batch_chosen_docs)

@DiversityStrategy.register_subclass("topic")
@dataclass
class Topic(DiversityStrategy):

    roberta = pipeline("text-classification", model="cardiffnlp/tweet-topic-21-multi")
    doc2label = {}   

    def rerank(
        self,
        outputs: AgentBatchOutput,
        k: int
    ):

        docs = [outputs.docs[idx] for idx in reversed(np.argsort(outputs.doc_scores).tolist())]
        
        labels = [None for _ in docs]
        idxs_to_query = []
        docs_to_query = []
        for i, doc in enumerate(docs):
            if doc in self.doc2label:
                labels[i] = self.doc2label[doc]
            else:
                idxs_to_query.append(i)
                docs_to_query.append(doc)

        outputs = self.roberta(docs_to_query)

        labels_to_add = outputs.logits.argmax(-1).cpu().numpy().tolist()
        for idx, label, doc in zip(idxs_to_query, labels_to_add, docs_to_query):
            labels[idx] = label
            self.doc2label[doc] = label

        assert all([l is not None for l in labels])

        chosen_docs = []
        chosen_labels = []
        for doc, label in zip(docs, labels):
            if label not in chosen_labels:
                chosen_docs.append(doc)
                chosen_labels.append(label)
            if len(chosen_docs) == k:
                break

        for doc in chosen_docs:
            if len(chosen_docs) == k:
                break
            chosen_docs.append(doc)

        assert len(chosen_docs) == k

        docs = chosen_docs

        return AgentBatchOutput(docs=docs)


@DiversityStrategy.register_subclass("mcvae")
@dataclass
class MCVAE(DiversityStrategy):
 
    embed_fn: callable 
    n: int 
    s: int 
    device: str
    
    def rerank(
        self, 
        outputs: AgentBatchOutput, 
        k: int
    ):
        embeds = self.embed_fn(
            torch.tensor(np.array(outputs.query_embed)).to(self.device), num_samples=self.s
        )

        scores = torch.bmm(
            embeds, torch.tensor(np.array(outputs.doc_embeds)).to(self.device).transpose(2, 1)
        )

        batch_docs = []
        for i in range(len(outputs.docs)):
            votes = scores[i].argmax(-1).cpu().numpy().tolist()
            vote_sums = [(j, votes.count(j)) for j in range(self.n)]
            vote_sums = list(reversed(sorted(vote_sums, key=lambda x: x[1])))[:k]
            vote_idxs, _ = zip(*vote_sums)
            docs = [outputs.docs[idx] for idx in vote_idxs]
            batch_docs.append(docs)

        return AgentBatchOutput(docs=batch_docs)


class SearcherOutput(list):
    """
    Should behave as a list containing the chosen_idxs to ensure
    compatibility with the rest of the codebase.
    We also want to track the scores of all the considered idxs.
    """

    def __init__(self, iterable, scores):
        super().__init__(iterable)
        self.scores = scores

@dataclass
class SimSR(DiversityStrategy):

    n: int = 15
    s: int = 25
    tau: float = 10.0

    @abstractmethod
    def run_search(self, scores):
        pass

    def _search(self, scores, docs, searcher):
        idxs = self.run_search(scores)
        score = np.mean(np.max(scores[list(idxs)], axis=0), axis=-1).item()
        best_answer = [docs[idx] for idx in idxs]

        return best_answer, score, idxs

    def rerank(self, outputs: AgentBatchOutput, k: int) -> AgentBatchOutput:
        # Check if we have enough documents to rerank
        # must be more than s and more than n to rerank
        if len(outputs.docs) <= self.s or len(outputs.docs) <= self.n:
            raise ValueError("Not enough documents to rerank")

        # Sort outputs
        docs = outputs.docs
        if self.n != self.s:
            argsort = np.argsort(np.array(outputs.doc_scores), axis=-1)
            docs = [[outputs.docs[i][idx] for idx in argsort[i]] for i in range(argsort.shape[0])]
                
        world_docs = [D[-self.s:] for D in docs]
        world_scores = [S[-self.s:] for S in outputs.doc_scores]
        policy_docs = [D[-self.n:] for D in docs]

        if ray.is_initialized():
            scores = [ray.remote(compute_f1_matrix_fast).remote(pd, wd) for pd, wd in zip(policy_docs, world_docs)]
            scores = ray.get(scores)
        else:
            scores = [compute_f1_matrix_fast(pd, wd) for pd, wd in zip(policy_docs, world_docs)]

        tau = 10.0
        probs = [softmax(s / tau) for s in world_scores]
        scores = [s * np.expand_dims(p, axis=0) for s, p in zip(scores, probs)]

        with timer.lap("searching"):
            if not ray.is_initialized():
                best_answer, score, idxs = zip(*[self._search(s, docs) for s, docs in zip(scores, policy_docs)])
            else:
                results = [ray.remote(self._search).remote(s, docs) for s, docs in zip(scores, policy_docs)]
                best_answer, score, idxs = zip(*ray.get(results))
        
        return AgentBatchOutput(
            docs=list(best_answer),
            score=list(score),
            doc_indices=list(idxs),
            topn_doc_indices=[I.tolist() for I in outputs.doc_indices],
            topn_docs=policy_docs,
        )


@DiversityStrategy.register_subclass("sim_sr_greedy")
class SimSRGreedy(SimSR):

    similarity_penalty: float = 0.05

    def run_search(self, scores):
        idxs = list(range(scores.shape[0]))
        chosen_idxs = []
        all_scores = []
        for _ in range(self.k):
            chosen_idxs, idxs, e_scores = self._get_idxs(scores, idxs, chosen_idxs)
            all_scores.append(e_scores)

        # Normalisation
        all_scores = [[np.round(s * 1000, 2) for s in S] for S in all_scores]
        search_outputs = SearcherOutput(chosen_idxs, all_scores)

        return search_outputs

    def _get_idxs(self, scores, idxs, chosen_idxs):
        e_scores = []
        for idx in idxs:
            if idx in chosen_idxs:
                e_scores.append(-1e10)
                continue
            tmp_idxs = chosen_idxs + [idx]
            S = scores[list(tmp_idxs)]
            sim_penalty = np.max(scores[chosen_idxs, idx]) if len(chosen_idxs) > 0 else 0.0
            e_score = self.scorer.score(S) - self.similarity_penalty * sim_penalty
            e_scores.append(e_score)

        best_idx = idxs[np.argmax(e_scores)]  # Note: this is the idx in idxs, not in scores
        chosen_idxs.append(best_idx)

        return chosen_idxs, idxs, e_scores


@DiversityStrategy.register_subclass("sim_sr_ablative")
class SimSRAblative(SimSR):

    def run_search(self, scores):
        idxs = list(range(scores.shape[0]))        

        while len(idxs) > self.k:
            best_score = -1.0
            best_idx = None
            for idx in idxs:
                tmp_idxs = copy.copy(idxs)
                tmp_idxs.remove(idx)

                S = scores[tmp_idxs]
                e_score = self.scorer.score(S)

                best_score, best_idx = max([(best_score, best_idx), (e_score, idx)], key=lambda x: x[0])

            idxs.remove(best_idx)

        e_scores = scores.mean(axis=1)[idxs].tolist()

        # Resort idxs by score in descending order
        idxs = [x for _, x in sorted(zip(e_scores, idxs), key=lambda x: x[0], reverse=True)]

        return idxs
