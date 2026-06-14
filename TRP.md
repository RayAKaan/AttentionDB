Design Document: Tunable Retrieval Parameters (TRP)
Feature Name: Tunable Retrieval Parameters (TRP)
Status: Design Phase
Target Version: v0.2 (Post-Recall Validation)
Owner: Rayyan Kaan

1. Motivation
Current vector and hybrid databases either:

Hardcode algorithmic parameters (e.g., HNSW ef, ef_construction), or
Only expose limited runtime controls.
AttentionDB aims to differentiate by giving users meaningful control over the core retrieval algorithm. This enables workload-specific optimization (e.g., “fast but approximate” vs “slow but high-recall”) while maintaining good defaults.

2. Goals
Primary Goals (v1)
Allow users to tune key HNSW parameters at the collection level.
Provide sensible, safe defaults so the feature is usable without deep expertise.
Make the tuning experience feel like a natural part of the database (similar to CREATE INDEX ... WITH).
Secondary Goals (Future)
Per-head parameter overrides
Query-level hints
Observability (EXPLAIN, index statistics)
Auto-tuning recommendations
Non-Goals (v1)
Global system-wide tuning
Automatic parameter selection
Per-query overrides
3. Proposed Design (v1)
3.1 Scope
In version 1, we will expose the following five parameters at the collection level:

Parameter	User-Facing Name	Type	Default	Description
ef_search	ef_search	int	64	Controls search quality vs speed during queries
ef_construction	ef_construction	int	400	Controls graph quality during index building
max_nb_connection	max_connections	int	16	Number of connections per node in HNSW graph
similarity_metric	similarity	string	'cosine'	Distance metric (cosine, dot_product, l2)
enable_exact_reranking	exact_rerank	bool	true	Whether to perform exact reranking after HNSW
3.2 Syntax
Creating a Collection with Tunable Parameters
SQL

CREATE COLLECTION papers (
    title       TEXT,
    abstract    TEXT,
    year        INT,
    authors     TEXT[]
)
WITH (
    ef_search = 128,
    ef_construction = 400,
    max_connections = 32,
    similarity = 'cosine',
    exact_rerank = true
);
Altering Parameters Later
SQL

ALTER COLLECTION papers 
SET (
    ef_search = 256,
    max_connections = 48
);
Viewing Current Settings
SQL

SHOW COLLECTION papers SETTINGS;
4. Implementation Considerations
4.1 Storage
Settings will be stored in the collection metadata (alongside schema).
When an HNSW index is created or rebuilt, it will read these settings.
4.2 Validation Rules (v1)
Parameter	Min	Max	Notes
ef_search	1	2048	Must be ≥ k
ef_construction	10	2000	Higher values increase build time
max_connections	4	128	Affects memory usage
similarity	—	—	Only cosine, dot_product, l2 allowed
exact_rerank	—	—	Boolean
4.3 Default Behavior
If a user does not specify any parameters, the following safe defaults will be used:

ef_search = 64
ef_construction = 400
max_connections = 16
similarity = 'cosine'
exact_rerank = true
These defaults should deliver reasonable recall/speed trade-offs for most workloads.

5. Risks & Mitigations
Risk	Mitigation
Users create poor-performing indexes	Strong defaults + validation + warnings
Inconsistent behavior across collections	Clear documentation + SHOW COLLECTION SETTINGS
Increased testing surface	Regression tests with default configuration
Memory blowup from high max_connections	Set reasonable upper limits
6. Future Phases
Phase	Feature	Description
Phase 2	Per-Head Overrides	Allow different settings per attention head
Phase 3	Query Hints	/*+ ef_search(256) */ style hints
Phase 4	Observability	EXPLAIN showing active parameters
Phase 5	Auto-tuning	System suggests good values based on workload
7. Open Questions
Should we allow changing these parameters without rebuilding the index?
How should we handle migration when defaults change in future versions?
Should we expose these parameters via the REST/gRPC API as well?