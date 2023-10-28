import copy
import math
import numpy as np
import torch
import torch.nn.functional as F

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datasets import Dataset
from scipy.special import softmax
from scipy.stats import zscore
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from typing import Optional
from tqdm import tqdm

from ..utils import RegistryMixin, compute_f1_matrix_fast
from .agent_utils import AgentBatchOutput


# Can run without ray installed
is_ray_installed = True
try:
    import ray
except ImportError:
    is_ray_installed = False


@dataclass
class DiversityConfig:
    n: int = 3
    s: int = 3
    tau: float = 10.0
    redundancy_penalty: float = 0.1
    embed_fn: Optional[callable] = None
    device: str = "cuda"


class DiversityStrategy(RegistryMixin, ABC):
    subclasses = {}
    requires_doc_embeds = False  # Child classes should set this to True if they require doc embeds

    def __init__(self, config: DiversityConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def rerank(self, outputs: AgentBatchOutput, k: int) -> AgentBatchOutput:
        pass

@DiversityStrategy.register_subclass("mmr")
class MMR(DiversityStrategy):

    requires_doc_embeds = True

    def rerank(
        self, 
        outputs: AgentBatchOutput, 
        k: int
    ):
        doc_embeds = torch.tensor(outputs.doc_embeds).to(self.config.device)
        # BNM, BNM -> BNN
        inter_doc_scores = torch.bmm(doc_embeds, doc_embeds.transpose(1, 2)).cpu().numpy()
        #inter_doc_scores = np.einsum('BNM,BOP -> BNP', doc_embeds, np.transpose(doc_embeds, (0, 2, 1)))
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
class Topic(DiversityStrategy):

    roberta = pipeline("text-classification", model="cardiffnlp/tweet-topic-21-multi", device=0)
    doc2label = {}

    def precompute(self, dataset: Dataset) -> None:
        labels = []
        for doc in tqdm(self.roberta(KeyDataset(dataset, "text"), batch_size=32, truncation=True, padding="max_length", max_length=64), total=len(dataset)):
            labels.append(doc["label"])
        texts = dataset["text"]

        for text, label in zip(texts, labels):
            self.doc2label[text] = label

    def rerank(
        self,
        outputs: AgentBatchOutput,
        k: int
    ):

        batch_docs = []
        for i in range(len(outputs.docs)):

            docs = [outputs.docs[i][idx] for idx in reversed(np.argsort(outputs.doc_scores[i]).tolist())]
            
            labels = [None for _ in docs]
            idxs_to_query = []
            docs_to_query = []
            for i, doc in enumerate(docs):
                if doc in self.doc2label:
                    labels[i] = self.doc2label[doc]
                else:
                    idxs_to_query.append(i)
                    docs_to_query.append(doc)

            topic_outputs = self.roberta(docs_to_query)

            labels_to_add = [s["label"] for s in topic_outputs]
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

            batch_docs.append(chosen_docs)

        return AgentBatchOutput(docs=batch_docs)


@DiversityStrategy.register_subclass("mcvae")
class MCVAE(DiversityStrategy):

    requires_doc_embeds = True
    
    def rerank(
        self, 
        outputs: AgentBatchOutput, 
        k: int
    ):


        embeds = self.config.embed_fn(
            torch.tensor(outputs.query_embed).to(self.config.device), num_samples=self.config.s
        ) # bsz x s x dim

        scores = torch.bmm(
            embeds, torch.tensor(outputs.doc_embeds).to(self.config.device).transpose(1, 2)
        )

        batch_docs = []
        for i in range(len(outputs.docs)):
            votes = scores[i].argmax(-1).cpu().numpy().tolist()
            vote_sums = [(j, votes.count(j)) for j in range(self.config.n)]
            vote_sums = list(reversed(sorted(vote_sums, key=lambda x: x[1])))[:k]
            vote_idxs, _ = zip(*vote_sums)
            docs = [outputs.docs[i][idx] for idx in vote_idxs]

            batch_docs.append(docs)

        return AgentBatchOutput(docs=batch_docs)


def _search(scores, docs, k, search_fn):
    idxs = search_fn(scores, k)
    score = np.mean(np.max(scores[list(idxs)], axis=0), axis=-1).item()
    best_answer = [docs[idx] for idx in idxs]

    return best_answer, score, idxs


def _rerank(policy_docs, world_docs, world_scores, k, search_fn, tau):
    scores = [compute_f1_matrix_fast(pd, wd) for pd, wd in zip(policy_docs, world_docs)]
    probs = [softmax(np.array(s) / tau) for s in world_scores]
    scores = [s * np.expand_dims(p, axis=0) for s, p in zip(scores, probs)]
    best_answer, score, idxs = zip(*[_search(s, docs, k, search_fn) for s, docs in zip(scores, policy_docs)])

    return best_answer, score, idxs


class SimSR(DiversityStrategy):

    @abstractmethod
    def run_search(self, scores, k):
        pass

    @staticmethod
    def _score_fn(scores):
        return np.mean(np.max(scores, axis=0), axis=-1).item()

    def rerank(self, outputs: AgentBatchOutput, k: int) -> AgentBatchOutput:
        # Check if we have enough documents to rerank
        # must be more than s and more than n to rerank
        if len(outputs.docs[0]) < self.config.s or len(outputs.docs[0]) < self.config.n:
            raise ValueError("Not enough documents to rerank")

        # Sort outputs
        docs = outputs.docs
        if self.config.n != self.config.s:
            argsort = np.argsort(np.array(outputs.doc_scores), axis=-1)
            docs = [[outputs.docs[i][idx] for idx in argsort[i]] for i in range(argsort.shape[0])]
                
        world_docs = [D[-self.config.s:] for D in docs]
        world_scores = [S[-self.config.s:] for S in outputs.doc_scores]
        policy_docs = [D[-self.config.n:] for D in docs]

        if not is_ray_installed or not ray.is_initialized():
            best_answer, score, idxs = _rerank(
                policy_docs, 
                world_docs, 
                world_scores, 
                k, 
                self.run_search, 
                self.config.tau
            )
        else:
            # Get number of workers
            num_workers = ray.cluster_resources().get("CPU", 1)

            # Chunk through inputs to _rerank
            chunk_size = math.ceil(len(docs) / num_workers)
            chunks = [docs[i:i + chunk_size] for i in range(0, len(docs), chunk_size)]
            chunks = [[c, world_docs[i:i + chunk_size], world_scores[i:i + chunk_size], k, self.run_search, self.config.tau] for i, c in enumerate(chunks)]

            # Run _rerank in parallel
            results = [ray.remote(_rerank).remote(*c) for c in chunks]
            results = ray.get(results)

            # Unpack results
            best_answer, score, idxs = zip(*results)

            # Unchunk results
            best_answer = [a for b in best_answer for a in b]
            score = [s for s in score for _ in range(len(s))]
            idxs = [i for i in idxs for _ in range(len(i))]
        
        return AgentBatchOutput(
            docs=list(best_answer),
            score=list(score),
            doc_indices=list(idxs),
            topn_doc_indices=[I.tolist() for I in outputs.doc_indices],
            topn_docs=policy_docs,
        )


@DiversityStrategy.register_subclass("sim_sr_greedy")
class SimSRGreedy(SimSR):

    def run_search(self, scores, k):
        idxs = list(range(scores.shape[0]))
        chosen_idxs = []
        all_scores = []
        for _ in range(k):
            chosen_idxs, idxs, e_scores = self._get_idxs(scores, idxs, chosen_idxs)
            all_scores.append(e_scores)

        # Normalisation
        all_scores = [[np.round(s * 1000, 2) for s in S] for S in all_scores]
        chosen_idxs

        return chosen_idxs

    def _get_idxs(self, scores, idxs, chosen_idxs):
        e_scores = []
        for idx in idxs:
            if idx in chosen_idxs:
                e_scores.append(-1e10)
                continue
            tmp_idxs = chosen_idxs + [idx]
            S = scores[list(tmp_idxs)]
            red_penalty = np.max(scores[chosen_idxs, idx]) if len(chosen_idxs) > 0 else 0.0
            e_score = self._score_fn(S) - self.config.redundancy_penalty * red_penalty
            e_scores.append(e_score)

        best_idx = idxs[np.argmax(e_scores)]  # Note: this is the idx in idxs, not in scores
        chosen_idxs.append(best_idx)

        return chosen_idxs, idxs, e_scores


@DiversityStrategy.register_subclass("sim_sr_ablative")
class SimSRAblative(SimSR):

    def run_search(self, scores, k):
        idxs = list(range(scores.shape[0]))        

        while len(idxs) > k:
            best_score = -1.0
            best_idx = None
            for idx in idxs:
                tmp_idxs = copy.copy(idxs)
                tmp_idxs.remove(idx)

                S = scores[tmp_idxs]
                e_score = self._score_fn(S)

                best_score, best_idx = max([(best_score, best_idx), (e_score, idx)], key=lambda x: x[0])

            idxs.remove(best_idx)

        e_scores = scores.mean(axis=1)[idxs].tolist()

        # Resort idxs by score in descending order
        idxs = [x for _, x in sorted(zip(e_scores, idxs), key=lambda x: x[0], reverse=True)]

        return idxs


@DiversityStrategy.register_subclass("sim_sr_ablative_gpu")
class SimSRAblativeGPU(SimSR):

    """
    A GPU-based implementation of ablative search.
    """

    def rerank(self, outputs: AgentBatchOutput, k: int) -> AgentBatchOutput:
        # Check if we have enough documents to rerank
        # must be more than s and more than n to rerank
        if len(outputs.docs[0]) < self.config.s or len(outputs.docs[0]) < self.config.n:
            raise ValueError("Not enough documents to rerank")

        # Sort outputs
        docs = outputs.docs
        if self.config.n != self.config.s:
            argsort = np.argsort(np.array(outputs.doc_scores), axis=-1)
            docs = [[outputs.docs[i][idx] for idx in argsort[i]] for i in range(argsort.shape[0])]
                
        world_docs = [D[-self.config.s:] for D in docs]
        world_scores = [S[-self.config.s:] for S in outputs.doc_scores]
        policy_docs = [D[-self.config.n:] for D in docs]

        if not is_ray_installed or not ray.is_initialized():
            scores = [compute_f1_matrix_fast(pd, wd) for pd, wd in zip(policy_docs, world_docs)]
        else:
            scores = [ray.remote(compute_f1_matrix_fast).remote(pd, wd) for pd, wd in zip(policy_docs, world_docs)]
            scores = ray.get(scores)

        probs = F.softmax(torch.tensor(world_scores).cuda() / self.config.tau, dim=-1).unsqueeze(1)
        scores = torch.tensor(np.array(scores)).cuda() * probs

        idxs = self.run_search(scores, k)
        best_answer = [[policy_docs[i][x] for x in idx] for i, idx in enumerate(idxs)]

        return AgentBatchOutput(
            docs=best_answer,
            doc_indices=idxs,
            topn_doc_indices=[I.tolist() for I in outputs.doc_indices],
            topn_docs=policy_docs,
        )

    def run_search(self, scores, k):
        """
        scores is 3d tensor of shape (batch size, n, s)
        """
        idxs = [list(range(scores.shape[1])) for _ in range(scores.shape[0])]

        scores = scores.unsqueeze(2).cuda().expand(-1, -1, scores.shape[1], -1) 

        # Create a mask
        mask = torch.ones(scores.size()).cuda()
        for i in range(scores.shape[1]):
            mask[:, i, i, :] = 0.0

        best_idx_mask = torch.ones([scores.shape[0], scores.shape[1]]).cuda()

        # We will incrementally add to the mask
        while len(idxs[0]) > k:
            # Compute scores
            S = scores * mask
            S = S.max(1)[0].mean(-1)

            # The best idx is the one that led to the highest score when it
            # was removed.
            best_idxs = torch.argmax(S * best_idx_mask, dim=-1)
            for i, (best_idx, idx) in enumerate(zip(best_idxs, idxs)):
                idxs[i].remove(best_idx)

                # Update the masks
                mask[i, best_idx, :, :] = 0.0
                best_idx_mask[i, best_idx] = 0.0

        # Check that we have the right number of idxs
        assert len(idxs[0]) == k

        return idxs
