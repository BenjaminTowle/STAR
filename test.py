# Testing the new agent refactoring
from datasets import Dataset
from transformers import pipeline

from src.agents.agent import RetrievalAgent
from src.agents.diversity import DiversityStrategy
from src.agents.index import Index


def main():
    # Load response set
    response_set = Dataset.load_from_disk("../data/personachat_reply_set")

    model = pipeline("feature-extraction", model="distilbert-base-uncased", device=0)

    # Load index
    index = Index.create("faiss", dataset=response_set, tokenizer=model.tokenizer, model=model.model, device="cuda")

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

if __name__ == "__main__":
    main()
