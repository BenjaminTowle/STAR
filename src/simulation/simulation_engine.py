# An engine to coordinate entire simulation process
# TODO: Rewrite simulation engine to deal with message, reply pairs, rather than entire dialogues.
import logging
import jsonlines
import re

from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import Optional, List

from .metrics import MetricLogger, MetricInput
from ..timer import timer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class EpisodeResult:
    observation: str
    action: str
    reward: float
    next_observation: str
    scores: Optional[List[List[float]]] = None
    doc_indices: Optional[List[int]] = None

    def to_json(self):
        return {
            "observation": self.observation,
            "action": self.action,
            "reward": self.reward,
            "next_observation": self.next_observation,
            "scores": self.scores,
            "doc_indices": self.doc_indices,
        }


class SimulationEngine:

    def __init__(
        self, 
        agent, 
        env,
        dataset,
        batch_size=1,
        logging_freq=10,
        n_episodes=100,
        json_write_path: Optional[str] = None,
        verbose=True,
    ):

        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        self.agent = agent
        self.env = env  # env is stateless, so we can reuse it for multiple samples in the batch
        self.logging_freq = logging_freq
        self.n_episodes = n_episodes if n_episodes != -1 else len(self.dataset)
        self.json_write_path = json_write_path
        self.verbose = verbose

        self.metric_logger = MetricLogger()

    def _run_batch(self, batch, episode_results):
        """
        Run a single batch.
        """
        with timer.lap("batch_act", batch_size=len(batch["messages"])):
            batch_outputs = self.agent.batch_act(batch["messages"])
        batch_action = batch_outputs.docs
    
        for i in range(len(batch["messages"])):
            next_obs, reward = self.env.step(batch_action[i], batch["responses"][i])
            episode_result = EpisodeResult(
                observation=batch["messages"][i], 
                action=batch_action[i], 
                reward=reward, 
                next_observation=next_obs,
                scores=batch_outputs.doc_indices[i].scores if batch_outputs.doc_indices is not None and getattr(batch_outputs.doc_indices[i], "scores", None) is not None else None,
                doc_indices=batch_outputs.topn_doc_indices[i] if batch_outputs.topn_doc_indices is not None else None,
            )
            #episode_results.append(episode_result)
            if self.json_write_path is not None:
                self.write_to_jsonl(episode_result)

            # remove initial 'reply: ' from action and next obs
            action = [re.sub(r"^reply: ", "", a) for a in batch_action[i]]
            next_obs = re.sub(r"^reply: ", "", next_obs)

            # Update metrics
            metric_input = MetricInput(reward=reward, action=action, target=next_obs)
            self.metric_logger.update(metric_input)

        return episode_results

    def write_to_jsonl(self, episode_result):
        """
        Write episode results to jsonl file.
        """
        with jsonlines.open(self.json_write_path, "a") as f:
            f.write(episode_result.to_json())

    def _warmup(self):
        """
        Warmup the agent by running a single episode.
        """
        batch = next(iter(self.dataloader))
        self.agent.batch_act(batch["messages"])

    def run(self):
        """
        Run simulation for n_episodes.  Each episode consists of a single bandit interaction with the environment.
        """
        self._warmup()
        episode_results = []
        for i, batch in enumerate(self.dataloader):
            episode_results = self._run_batch(batch, episode_results)

            if i % self.logging_freq == 0:
                info = f"Step: {i}; " + str(self.metric_logger)
                logger.info(info)

            if i == self.n_episodes - 1:
                break
        
        logger.info("Final results: " + str(self.metric_logger))

        return episode_results
