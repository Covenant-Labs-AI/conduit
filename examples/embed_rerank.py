from conduit.compute_provider.runpod.runpod_types import GPUS
from conduit.conduit_types import ComputeProvider, LmLiteModelConfig
from conduit.runtime import LMLiteBlock
from conduit.utils.deployment.models import DeploymentConstraint

lm = LMLiteBlock(
    models=[
        LmLiteModelConfig(
            "Qwen/Qwen3-VL-Embedding-8B",
            max_model_len=5000,
            max_model_concurrency=1,
            task="embed",
        ),
        LmLiteModelConfig(
            "Qwen/Qwen3-VL-Reranker-8B",
            max_model_len=5000,
            max_model_concurrency=1,
            task="rerank",
        ),
    ],
    constraints=[DeploymentConstraint.ENTERPRISE],
    compute_provider=ComputeProvider.RUNPOD,
    gpu=GPUS.H100_NVL,
)


def run_embedding_test():
    print("\n=== EMBEDDING TEST ===")
    texts = [
        "Beijing is the capital of China.",
        "Shanghai is a major city in China.",
        "Paris is the capital of France.",
        "The moon is far from Earth.",
    ]

    emb = lm.embed("Qwen/Qwen3-VL-Embedding-8B", texts)
    print(f"embedded {len(emb.data)} texts")
    print(f"embedding dim: {len(emb.data[0].embedding) if emb.data else 0}")

    vecs = [item.embedding for item in emb.data]
    query_vec = (
        lm.embed("Qwen/Qwen3-VL-Embedding-8B", "capital of china").data[0].embedding
    )

    # Basic sanity checks
    assert len(vecs) == len(texts), "wrong number of embeddings returned"
    assert all(len(v) == len(vecs[0]) for v in vecs), "embedding dims are inconsistent"

    # Very simple similarity helper
    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    scored = [(dot(query_vec, v), t) for v, t in zip(vecs, texts)]
    scored.sort(key=lambda x: x[0], reverse=True)

    print("\nquery: capital of china")
    print("top matches by embedding similarity:")
    for score, text in scored:
        print(f"{score:.4f} | {text}")


def run_reranker_test():
    print("\n=== RERANKER TEST ===")

    tests = [
        {
            "name": "capital question",
            "query": "capital of china",
            "documents": [
                "Beijing is the capital of China.",
                "Shanghai is one of the largest cities in China.",
                "Paris is the capital of France.",
                "The moon is far from Earth.",
            ],
        },
        {
            "name": "author question with lexical traps",
            "query": "who wrote pride and prejudice",
            "documents": [
                "Jane Austen wrote Pride and Prejudice.",
                "Pride and Prejudice is a novel about manners, marriage, and class.",
                "Charlotte Bronte wrote Jane Eyre.",
                "The movie adaptation of Pride and Prejudice was widely acclaimed.",
                "J.K. Rowling wrote Harry Potter.",
            ],
        },
        {
            "name": "planet fact question",
            "query": "largest planet in the solar system",
            "documents": [
                "Mars is known as the red planet.",
                "Jupiter is the largest planet in the solar system.",
                "Earth is the only known habitable planet.",
                "Saturn is famous for its rings.",
            ],
        },
    ]

    for test in tests:
        print(f"\n--- {test['name']} ---")
        print(f"query: {test['query']}")

        results = lm.rerank(
            "Qwen/Qwen3-VL-Reranker-8B",
            query=test["query"],
            documents=test["documents"],
            top_n=len(test["documents"]),
            return_documents=True,
        )

        for r in results.results:
            print(f"{r.relevance_score:.4f} | {r.document}")


if lm.ready:
    run_embedding_test()
    run_reranker_test()
else:
    print("Not ready")
