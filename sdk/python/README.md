# AttentionDB Python SDK

Python client for the AttentionDB multi-head attention vector database.

## Installation

```bash
pip install attentiondb
```

Or from source:

```bash
cd sdk/python
pip install .
```

## Usage

```python
from attentiondb import AttentionDB

# Connect to the server
client = AttentionDB("http://localhost:8080", api_key="my-api-key")

# Create a collection with multiple heads
client.create_collection(
    "papers",
    dimension=128,
    heads=["semantic", "temporal"],
    head_settings={
        "semantic": {"ef_search": 256},
        "temporal": {"ef_search": 64},
    },
)

# Insert a document
doc_id = client.insert("papers", {
    "title": "Attention Is All You Need",
    "text": "The transformer architecture...",
    "semantic_vector": "0.1,0.2,0.3,0.4,0.5",
    "temporal_vector": "0.5,0.4,0.3,0.2,0.1",
})

# Search with multi-head attention
results = client.search(
    "papers",
    query="0.1,0.2,0.3,0.4,0.5",
    heads=["semantic", "temporal"],
    top_k=10,
)

for r in results.results:
    print(f"ID: {r.id}, Score: {r.score}, Fields: {r.fields}")

# Hybrid search (BM25 + vector)
results = client.search(
    "papers",
    query="0.1,0.2,0.3,0.4,0.5",
    heads=["semantic"],
    hybrid=True,
    query_text="transformer attention mechanism",
)

# Health checks
print(client.health())
print(client.health_live())
print(client.health_ready())

# Backup
backup = client.create_backup()
print(f"Backup created: {backup.backup_id}")
