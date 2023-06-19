import abc
import numpy as np

from rouge_score import rouge_scorer
from statistics import mean
from typing import List


scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rouge3"], use_stemmer=True)


def _blended_rouge(rouge: dict):
    """Mixes together r1, r2, r3 according to https://arxiv.org/pdf/2106.02017.pdf"""
    r1 = rouge["rouge1"].fmeasure / 6
    r2 = rouge["rouge2"].fmeasure / 3
    r3 = rouge["rouge3"].fmeasure / 2
    return r1 + r2 + r3

def rouge(targets, preds):
    # Computes blended rouge for a batch of samples
    scores = []
    for t, P in zip(targets, preds):
        score = max([_blended_rouge(scorer.score(t, p)) for p in P])
        scores.append(score)

    return scores

def rouge_single(target, preds):
    # Computes blended rouge for a single sample
    return max([_blended_rouge(scorer.score(target, pred)) for pred in preds])

def self_rouge(preds):
    # Convert to two lists of targets and preds
    new_preds = []
    new_targets = []
    for P in preds:
        K = len(P)

        for k in range(K):
            new_preds.append(P[:k] + P[k+1:])
            new_targets.append(P[k])

    return rouge(new_targets, new_preds)


class MetricInput:
    def __init__(self, reward, action, target):
        self.reward = reward
        self.action = action
        self.target = target


class Metric(abc.ABC):

    subclasses = {}

    def __init__(self):
        self.values = []

    @classmethod
    def name(cls):
        # Return the str name of the class
        inv_subclasses = {v: k for k, v in cls.subclasses.items()}
        return inv_subclasses[cls]

    @classmethod
    def register_subclass(cls, metric_type):
        def decorator(subclass):
            cls.subclasses[metric_type] = subclass
            return subclass
        
        return decorator

    @classmethod
    def create(cls, metric_type: str):
        if metric_type not in cls.subclasses:
            raise ValueError(f"Bad metric_type: {metric_type}.")
        
        return cls.subclasses[metric_type]()

    @abc.abstractmethod
    def _update(self, metric_input: MetricInput) -> float:
        raise NotImplementedError()

    def update(self, metric_input: MetricInput):
        value = self._update(metric_input)
        self.values.append(value)

    def mean(self):
        return mean(self.values)

    def __str__(self):
        return f"{self.name()}: {self.mean():.4f}"

@Metric.register_subclass("rouge")
class RougeMetric(Metric):

    def _update(self, metric_input: MetricInput):
        return rouge_single(metric_input.target, metric_input.action)

@Metric.register_subclass("rouge_1")
class RougeMetric(Metric):

    def _update(self, metric_input: MetricInput):
        rouge_scores = [rouge_single(metric_input.target, [pred]) for pred in metric_input.action]
        if np.argmax(np.array(rouge_scores)) == 0:
            return rouge_single(metric_input.target, metric_input.action)
        else:
            return 0.


@Metric.register_subclass("rouge_2")
class RougeMetric(Metric):

    def _update(self, metric_input: MetricInput):
        rouge_scores = [rouge_single(metric_input.target, [pred]) for pred in metric_input.action]
        if np.argmax(np.array(rouge_scores)) == 1:
            return rouge_single(metric_input.target, metric_input.action)
        else:
            return 0.

@Metric.register_subclass("rouge_3")
class RougeMetric(Metric):

    def _update(self, metric_input: MetricInput):
        rouge_scores = [rouge_single(metric_input.target, [pred]) for pred in metric_input.action]
        if np.argmax(np.array(rouge_scores)) == 2:
            return rouge_single(metric_input.target, metric_input.action)
        else:
            return 0.


"""
@Metric.register_subclass("CTR")
class CTRMetric(Metric):

    def _update(self, metric_input: MetricInput):
        return max(metric_input.reward)

@Metric.register_subclass("CTR_1")
class CTRMetric(Metric):

    def _update(self, metric_input: MetricInput):
        return metric_input.reward[0]

@Metric.register_subclass("CTR_2")
class CTRMetric(Metric):

    def _update(self, metric_input: MetricInput):
        return metric_input.reward[1]

@Metric.register_subclass("CTR_3")
class CTRMetric(Metric):

    def _update(self, metric_input: MetricInput):
        return metric_input.reward[2]"""

@Metric.register_subclass("MRR")
class MRRMetric(Metric):

    def _update(self, metric_input: MetricInput):
        return 1 / (np.argmax(np.array(metric_input.reward)) + 1) if max(metric_input.reward) > 0 else 0.

@Metric.register_subclass("self-rouge")
class SelfRougeMetric(Metric):

    def _update(self, metric_input: MetricInput):
        return mean(self_rouge([metric_input.action]))

@Metric.register_subclass("num_words")
class NumWordsMetric(Metric):

    def _update(self, metric_input: MetricInput):
        return mean([len(doc.split()) for doc in metric_input.action])

class MetricLogger:
    def __init__(self, metrics: List[str] = None):
        if metrics is None:
            metrics = Metric.subclasses.keys()
        self.metrics = []
        for metric in metrics:
            self.metrics.append(Metric.create(metric))

    def update(self, metric_input: MetricInput):
        for metric in self.metrics:
            metric.update(metric_input)

    def __str__(self):
        return "; ".join([str(metric) for metric in self.metrics])
