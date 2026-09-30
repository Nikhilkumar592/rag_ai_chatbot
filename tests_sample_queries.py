"""Run benchmark queries and save results.

    python tests_sample_queries.py                       # call the pipeline directly
    python tests_sample_queries.py --api http://127.0.0.1:8000   # call the FastAPI server
"""
import argparse
import json

QUERIES = [
    "What is the core definition of Agentic AI as outlined in the eBook?",
    "What are the main architectural components required to build agentic systems?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    "What is the capital of France?",  # out-of-scope: must be refused
]
REFUSAL = "I cannot answer based on the provided document."


def run(api: str | None):
    if api:
        import requests

        def ask(q):
            r = requests.post(f"{api}/chat", json={"query": q}, timeout=120)
            r.raise_for_status()
            return r.json()
    else:
        from src.graph import ask

    results = []
    for i, q in enumerate(QUERIES, 1):
        out = ask(q)
        results.append(out)
        print(f"\n[{i}] Q: {q}")
        print(f"    A: {out['final_answer']}")
        print(f"    confidence={out['confidence_score']}  chunks={len(out['retrieved_context_chunks'])}")

    with open("sample_outputs.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Sanity checks
    required_keys = {"query", "final_answer", "retrieved_context_chunks", "confidence_score"}
    assert all(required_keys <= set(r) for r in results)
    oos = results[-1]
    assert REFUSAL in oos["final_answer"] and oos["confidence_score"] == 0.0, "Out-of-scope query was not refused!"
    print("\nAll checks passed. Saved sample_outputs.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--api", help="Base URL of the running FastAPI server")
    run(p.parse_args().api)
