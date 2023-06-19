# Testing the new agent refactoring
import pickle

from datasets import Dataset
from transformers import pipeline, FeatureExtractionPipeline, T5ForConditionalGeneration, T5TokenizerFast

from src.agents.agent import RetrievalAgent, STARAgent
from src.agents.diversity import DiversityStrategy
from src.agents.index import Index


def main():
    # Load response set
    response_set = Dataset.load_from_disk("../data/personachat_index")

    model = pipeline("feature-extraction", model="distilbert-base-uncased", device=0)

    # Load index
    index = Index.create("in_memory", dataset=response_set, device="cuda")

    # Load diversity strategy
    diversity_strategy = DiversityStrategy.create("mmr")

    # Load model
    agent = RetrievalAgent(
        model=model,
        k=3,
        device="cuda",
        index=index,
        response_set=response_set,
        diversity_strategy=diversity_strategy
    )

    query = ["Hello, how are you?"]

    output = agent.batch_act(query)
    print(output)

from torch import nn


def main_star():
    model = T5ForConditionalGeneration.from_pretrained("../data/t5-personachat/checkpoint-24000")
    model.config.tie_word_embeddings = False
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")

    id2reply = pickle.load(open("id2reply_pc.pkl", "rb"))

    lm_head = nn.Linear(
        model.config.d_model, len(id2reply), bias=False
    )
    lm_head.weight.data = model.lm_head.weight.data[len(tokenizer):, :]
    model.lm_head = lm_head

    # Adjust id2reply so that each id corresponds to the correct token
    id2reply = {i: id2reply[i + len(tokenizer)] for i in range(len(id2reply))}

    agent = STARAgent(model=model, tokenizer=tokenizer, id2reply=id2reply, device="cuda")

    query = ["Hello, how are you?"]

    output = agent.batch_act(query)
    print(output)

if __name__ == "__main__":
    main_star()
