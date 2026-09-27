# Weaviate — Readiness Report

Study `comparative-study-001` · §18 (feasibility).

## Status

**SMOKE-PASS (feasibility: IMPLEMENTED/VALIDATED).** Not BENCHMARK-READY alone.

## Evidence — C2-SMOKE-WEAVIATE-004 (terminal, PASS)

- version `1.39.6` (binary); ready endpoint 200 at 5.5 s
- schema create / object insert / graphql `nearVector` all HTTP 200
- guarded sampler attached: peak RSS 109,656 KB, 12 samples, no abort
- superseded ledger: -002 INVALID-STARTUP (no loopback cluster env, no
  `--scheme`/`--port`), -003 FAILED (`json.loads` on truncated `/v1/meta`;

  server itself was healthy)

## Documented env deltas (startup config, NOT engine changes)

v1.39 raft clustering requires an advertise address; the sandbox has no private
IP, so the smoke pinned:

- `CLUSTER_ADVERTISE_ADDR=127.0.0.1` + gossip/data ports
- `--scheme http --port 8080` (embedded swagger default otherwise demands TLS
  certs)

Every delta is recorded verbatim in the run's `config.yaml`.

## Verdict

Feasible on this host. Quality/latency claims are out of C2 scope.