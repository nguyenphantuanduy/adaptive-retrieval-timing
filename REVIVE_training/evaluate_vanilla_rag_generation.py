"""Generate the single-search top-five Qwen RAG baseline."""
from functools import partial

from REVIVE_training.baseline_generation import run_baseline

RETRIEVAL_TOP_K = 5


def retrieve_for_dataset(dataset):
    if dataset == "musique":
        from shared.ART_Models.ART_REVIVE.musique_retrieval_db import MuSiQueRetrievalDB

        database = MuSiQueRetrievalDB(top_k=RETRIEVAL_TOP_K)
    elif dataset == "hotpotqa":
        from shared.ART_Models.ART_REVIVE.hotpotqa_retrieval_db import HotpotQARetrievalDB

        database = HotpotQARetrievalDB(top_k=RETRIEVAL_TOP_K)
    elif dataset == "2wikimultihopqa":
        from shared.ART_Models.ART_REVIVE.two_wikimultihopqa_db import TwoWikiMultiHopQARetrievalDB

        database = TwoWikiMultiHopQARetrievalDB(top_k=RETRIEVAL_TOP_K)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return partial(database.search_with_metadata, top_k=RETRIEVAL_TOP_K)


if __name__ == "__main__":
    run_baseline("vanilla_rag_top5.jsonl", retrieve_for_dataset)
