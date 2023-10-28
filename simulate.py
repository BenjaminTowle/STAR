import logging
import os
import pickle
import torch

from datasets import set_caching_enabled, Dataset
from dataclasses import dataclass, field
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    AutoTokenizer,
)

from src.agents.agent import RetrievalAgent, STARAgent, Seq2SeqAgent
from src.agents.diversity import DiversityStrategy, DiversityConfig
from src.agents.index import Index
from src.corpora import Corpus
from src.modeling.mcvae import DistilBertMCVAE
from src.modeling.star import STARModel
from src.simulation.env import SmartReplyEnv
from src.simulation.simulation_engine import SimulationEngine
from src.timer import timer
from src.utils import set_random_seed, parse_args

from constants import *

# Can run without ray installed
is_ray_installed = True
try:
    import ray
except ImportError:
    is_ray_installed = False


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

set_caching_enabled(False)
torch.set_grad_enabled(False)


@dataclass
class Args:
    # Agent args  
    agent_type: str = field(
        default="retrieval",
        metadata={"choices": ["star", "seq2seq", "retrieval"],
            "help": "Type of agent to use"}
    )

    diversity_strategy: str = field(
        default="sim_sr_greedy",
        metadata={"choices": [
            "mmr", "topic", "sim_sr_greedy", "sim_sr_ablative", "sim_sr_ablative_gpu", "mcvae"],
            "help": "Reranking strategy."}
    )

    model_path: str = field(
        default=REDDIT_MATCHING,
        metadata={"help": "Path to pretrained model"}
    )

    tokenizer_path: str = field(
        default="distilbert-base-uncased",
        metadata={"help": "Path to pretrained tokenizer"}
    )

    index_path: str = field(
        default="../data/prefix_reddit_index",
        metadata={"help": "Path to index."}
    )

    index_type: str = field(
        default="in_memory",
        metadata={"choices": ["in_memory", "faiss"],
            "help": "Type of index to use."}
    )

    json_write_path: str = field(
        default="../data/reddit-train-5.jsonl",
        metadata={"help": "Path to write json file."}
    )

    task: str = field(
        default="reddit", 
        metadata={"choices": ["personachat", "dailydialog", "reddit"], 
        "help": "Task to train on"}
    )

    split: str = field(
        default="train", 
        metadata={"choices": ["train", "valid", "test"], 
        "help": "Which split to use for inference."}
    )
    
    k: int = field(
        default=3,
        metadata={"help": "Final number of documents to return from agent"}
    )

    n: int = field(
        default=100,
        metadata={"help": "Number of candidates to rerank"}
    )

    s: int = field(
        default=100,
        metadata={"help": "Number of simulations to run."}
    )

    redundancy_penalty: float = field(
        default=0.05,
        metadata={"help": "Penalty for redundant responses."}
    )

    use_posterior: bool = field(
        default=False,
        metadata={"help": "Whether to retrieve with targets or not."}
    )
    
    use_ray: bool = field(
        default=False,
        metadata={"help": "Whether to use Ray for parallelization."}
    )

    num_workers: int = field(
        default=4,
        metadata={"help": "Number of workers to use for parallelization."}
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
        default=32,
        metadata={"help": "Batch size"}
    )

    seed: int = field(
        default=0,
        metadata={"help": "Random seed"}
    )

    device: str = field(
        default="cuda",
        metadata={"choices": ["cuda", "cpu"], "help": "Device to use."}
    )


def build_retrieval_agent(args):

    reply_set = Dataset.load_from_disk(args.index_path)
    index = Index.create(
        args.index_type, 
        dataset=reply_set, 
        device=args.device,
        lm_bias_column="lm_score",
        lm_bias_alpha=0.2,
    )

    # This should work for all baselines, as it is just distilbert
    # with some additional layers on top.
    model = DistilBertMCVAE.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    diversity_strategy = None
    if args.diversity_strategy is not None:
        config = DiversityConfig(
            n=args.n,
            s=args.s,
            redundancy_penalty=args.redundancy_penalty,
            device=args.device,
            embed_fn=model.generate_embedding if args.diversity_strategy == "mcvae" else None,
        )
        diversity_strategy = DiversityStrategy.create(
            args.diversity_strategy, config
        )

        if "sim_sr" in args.diversity_strategy:
            n = max(args.n, args.s)
        else:
            n = args.n
    else:
        n = args.n

    agent = RetrievalAgent(
        model=model,
        tokenizer=tokenizer,
        index=index,
        diversity_strategy=diversity_strategy,
        k=args.k,
        n=n,
        device=args.device,
        use_posterior=args.use_posterior,
    )

    return agent


def build_seq2seq_agent(args):
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")

    agent = Seq2SeqAgent(
        model=model, tokenizer=tokenizer, device=args.device
    )

    return agent


def build_star_agent(args):
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    model = STARModel.from_pretrained(args.model_path, num_masked_tokens=len(tokenizer))

    id2reply = pickle.load(
        open(os.path.join(args.model_path, "id2reply.pkl"), "rb"))

    agent = STARAgent(
        model=model, tokenizer=tokenizer, id2reply=id2reply, device=args.device
    )

    return agent


def init_ray(num_cpus, is_ray_installed):
    if is_ray_installed:
        ray.init(num_cpus=num_cpus)
    else:
        logger.warning(
            "Ray is not installed, but use_ray was set to True. Using single process."
        )

def run():
    args = parse_args(Args)
    if args.use_ray:
        init_ray(args.num_workers, is_ray_installed)
    set_random_seed(args.seed)

    if args.agent_type == "retrieval":
        agent = build_retrieval_agent(args)
    elif args.agent_type == "seq2seq":
        agent = build_seq2seq_agent(args)
    elif args.agent_type == "star":
        agent = build_star_agent(args)
    else:
        raise NotImplementedError

    # Construct agent from json file.
    env = SmartReplyEnv()
    corpus = Corpus.create(args.task)
    dataset = corpus.get_dataset(split=args.split)  # This provides inputs to the agent and user simulator

    if args.use_posterior:
        # Create dictionary mapping messages to responses.
        message2response = {}
        for sample in dataset:
            message2response[sample["messages"]] = sample["responses"]
        agent.query2target = message2response

    if args.diversity_strategy == "topic":
        index_dataset = agent.index.dataset
        agent.diversity_strategy.precompute(index_dataset)

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

    timer.print(logger)

        
if __name__ == "__main__":
    run()
      