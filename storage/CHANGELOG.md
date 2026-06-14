# Changelog
All notable changes to AttentionDB Phase 1 will be documented in this file.

## [0.1.0] - 2026-06-13

### Added
- Core Record model with MessagePack serialization
- Write-Ahead Log (WAL) with CRC32 checksums and durability modes
- DocumentStore (memtable + optional WAL)
- ProjectionStore for K/V vectors
- StorageError enum
- Basic CLI (attentiondb) with insert and stats commands
- Unit tests for record, store, and WAL
- Example: basic_usage
- Makefile with common development targets

### Notes
- This is the storage foundation only. Query logic will be added in later phases.
- Focus: durability, correctness, and physical storage layout.
