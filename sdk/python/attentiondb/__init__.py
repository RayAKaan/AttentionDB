"""AttentionDB Python SDK.

Minimal async HTTP client for the AttentionDB vector database.

Usage:
    from attentiondb import AttentionDB

    client = AttentionDB("http://localhost:8080", api_key="my-key")

    # Create a collection
    client.create_collection(
        "papers",
        dimension=128,
        heads=["semantic", "temporal"],
        head_settings={
            "semantic": {"ef_search": 256},
            "temporal": {"ef_search": 64},
        }
    )

    # Insert documents
    doc_id = client.insert("papers", {
        "title": "Attention Is All You Need",
        "semantic_vector": "0.1,0.2,0.3,0.4",
    })

    # Search
    results = client.search(
        "papers",
        query="0.1,0.2,0.3,0.4",
        heads=["semantic"],
        top_k=10,
    )

    # Hybrid search (BM25 + vector)
    results = client.search(
        "papers",
        query="0.1,0.2,0.3,0.4",
        heads=["semantic"],
        hybrid=True,
        query_text="transformer attention",
    )
"""

from .client import AttentionDB

__all__ = ["AttentionDB"]
