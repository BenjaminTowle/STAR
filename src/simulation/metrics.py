import abc
import numpy as np

from statistics import mean
from typing import List

from eval import self_rouge, rouge_single


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

@Metric.register_subclass("CTR")
class CTRMetric(Metric):

    def _update(self, metric_input: MetricInput):
        return max(metric_input.reward)

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
