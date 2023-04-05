import logging
import ray

from datasets import set_caching_enabled
from dataclasses import dataclass, field
from transformers import HfArgumentParser

from src.agents.agent import Agent
from src.args import parse_args
from src.corpora import Corpus
from src.simulation.env import SmartReplyEnv
from src.simulation.simulation_engine import SimulationEngine
from src.timer import timer
from src.utils import set_random_seed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

set_caching_enabled(False)


@dataclass
class Args:
    # OS args  
    agent_path: str = field(
        default="src/config/t5.json",
        metadata={"help": "Path to agent json config"}
    )

    json_write_path: str = field(
        default=None,
        metadata={"help": "Path to write json file."}
    )

    task: str = field(
        default="personachat", 
        metadata={"choices": ["personachat", "dailydialog", "reddit"], 
        "help": "Task to train on"}
    )

    # Agent args
    agent_type: str = field(
        default="t5", 
        metadata={"choices": [
            "matching", "mmr", "mcvae", "simsr", "topic", "query_cache", "t5"
            ],
            "help": "Type of agent to use"
        }
    )
    
    k: int = field(
        default=3,
        metadata={"help": "Final number of documents to return from agent"}
    )
    
    use_ray: bool = field(
        default=False,
        metadata={"help": "Whether to use Ray for parallelization."}
    )

    mask_ground_truth: bool = field(
        default=False,
        metadata={"help": "Whether to mask the ground-truth in retrieval."}
    )

    # General args
    logging_verbosity: str = field(
        default="info", 
        metadata={
            "choices": ["none", "info", "debug"],
            "help": "Verbosity of logging"
        }
    )

    logging_freq: int = field(
        default=100,
        metadata={"help": "Frequency of logging"}
    )

    n_episodes: int = field(
        default=-1,
        metadata={"help": "Number of dialogues/episodes to run. -1 for all."}
    )

    
    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size"}
    )

    seed: int = field(
        default=0,
        metadata={"help": "Random seed"}
    )


def parse_args():
    parser = HfArgumentParser([Args])
    args, = parser.parse_args_into_dataclasses()
    return args


def run():
    args = parse_args()
    if args.use_ray:
        ray.init(num_cpus=4)
    set_random_seed(args.seed)

    # Construct agent from json file.
    agent = Agent.create_from_json(args.agent_type, args.agent_path)
    env = SmartReplyEnv()
    split = "test"
    corpus = Corpus.create(args.task)
    dataset = corpus.get_dataset(split=split)  # This provides inputs to the agent and user simulator
    simulation_engine = SimulationEngine(
        agent, 
        dataset=dataset,
        env=env,
        n_episodes=args.n_episodes,
        logging_freq=args.logging_freq,
        batch_size=args.batch_size,
        json_write_path=args.json_write_path,
    )

    simulation_engine.run()

    timer.print(logger, len(dataset))

        
if __name__ == "__main__":
    run()
      