from transformers import HfArgumentParser
from dataclasses import dataclass, field


@dataclass
class Args:  
    # Training args
    output_dir: str = "../data/simulation-pc-round1"
    data_dir: str = ""
    dataset_save_path: str = "../data/simulation-dataset"
    dataset_load_path: str = "none"
    task: str = field(default="simulation", 
    metadata={"choices": ["personachat", "simulation", "reddit"]})
    model_type: str = field(default="matching", 
        metadata={"choices": ["matching", "mcvae", "tfidf", "adapter"]})
    debug: bool = False
    bert_model_path: str = "distilbert-base-uncased" # "distilbert-base-uncased"
    base_model_type: str = field(default="bert", metadata={"choices": ["bert", "gpt"]})
    
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    use_symmetric_loss: bool = True
    max_context_length: int = 64
    max_response_length: int = 64
    max_turns: int = 1

    # MCVAE args
    z: int = field(default=256, metadata={"help": "Size of latent vector for MCVAE model"})
    kld_weight: float = 0.05
    use_kld_annealling: bool = False
    kld_annealling_steps: int = -1
    use_message_prior: bool = False

    # Eval Args
    agent_type: str = field(default="matching", metadata={"choices": ["matching", "mmr", "mcvae", "simulation", "topic"]})
    search_strategy: str = field(default="exhaustive", metadata={"choices": 
    ["ablative", "exhaustive", "samplerank", "greedy"]})
    response_set_path: str = "../data/personachat-dataset-lm-tfidf/test" 
    model_load_path: str = "../data/pc-distilbert-biencoder/checkpoint-24645"
    k: int = 3
    n: int = 15
    s: int = 25
    use_valid: bool = False
    use_lm_score: bool = False
    prediction_save_path: str = "preds_test_mc_reddit.txt"
    prediction_load_path: str = "preds_test_mc_reddit.txt"

    comparison_load_path: str = "none"

    # General args
    gpt_tokenizer_path: str = "microsoft/DialoGPT-small"
    bert_tokenizer_path: str = "bert-base-uncased"
    seed: int = 0
    device: str = "cuda"
    
def parse_args():
    parser = HfArgumentParser([Args])
    args, = parser.parse_args_into_dataclasses()
    return args
