import numpy as np

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AgentOutput:
    docs: List[str]
    score: Optional[float] = None
    doc_embeds: Optional[np.array] = None
    doc_scores: Optional[np.array] = None
    doc_indices: Optional[np.array] = None
    query_embed: Optional[np.array] = None
    contexts: Optional[List[str]] = None
    targets: Optional[List[str]] = None


@dataclass
class AgentBatchOutput:
    docs: List[List[str]]
    score: Optional[List[float]] = None
    doc_embeds: Optional[np.array] = None
    doc_scores: Optional[np.array] = None
    doc_indices: Optional[np.array] = None
    query_embed: Optional[np.array] = None
    contexts: Optional[List[List[str]]] = None
    targets: Optional[List[List[str]]] = None
    logits: Optional[List[List[int]]] = None
    topn_docs: Optional[List[List[str]]] = None
    topn_doc_indices: Optional[List[List[int]]] = None
