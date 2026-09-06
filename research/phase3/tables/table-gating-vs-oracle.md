# Table: gating weight allocation vs oracle head — PH3B-COMP-001 (seed-42 model)

Caption draft: *Mean gating weight per head by query type: the gate allocates the largest mean weight to the defining head of each type (title→title, body→body, mixed→full) but remains per-query noisy (oracle-head agreement 0.55–0.62, entropy low) — trained on 280 queries with 1536-dim hashed inputs.* [PH3B-COMP-001]

| Query type | n | w(title) | w(body) | w(full) | oracle-head agreement |
|---|---|---|---|---|---|
| title | 22 | 0.5677 | 0.2444 | 0.1879 | 0.5455 |
| body | 18 | 0.0006 | 0.6377 | 0.3617 | 0.6111 |
| mixed | 21 | 0.0491 | 0.3059 | 0.645 | 0.619 |
