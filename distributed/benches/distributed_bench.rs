use criterion::{criterion_group, criterion_main, Criterion, black_box};
use attentiondb_distributed::{RaftNode, ShardManager, Shard, ReplicaManager, ReadReplica};

fn raft_benchmark(c: &mut Criterion) {
    let mut raft = RaftNode::new(1, vec![2, 3, 4, 5]);

    c.bench_function("raft_append_entry", |b| {
        b.iter(|| {
            let _ = raft.append_entry("INSERT", black_box(vec![1, 2, 3]));
        });
    });

    c.bench_function("raft_replicate", |b| {
        b.iter(|| {
            let _ = raft.replicate_to_peers();
        });
    });
}

fn shard_benchmark(c: &mut Criterion) {
    let mut sm = ShardManager::new();
    for i in 0..10 {
        sm.add_shard(Shard::new(i, vec![format!("head_{}", i)], "addr"));
    }

    c.bench_function("shard_lookup", |b| {
        b.iter(|| {
            let _ = sm.get_shard_for_head(black_box("head_5"));
        });
    });
}

fn replica_benchmark(c: &mut Criterion) {
    let mut rm = ReplicaManager::new();
    for i in 0..100 {
        rm.add_replica(ReadReplica::new(i, i % 5, "addr"));
    }

    c.bench_function("replica_healthy_lookup", |b| {
        b.iter(|| {
            let _ = rm.get_healthy_replicas(black_box(3));
        });
    });
}

criterion_group!(benches, raft_benchmark, shard_benchmark, replica_benchmark);
criterion_main!(benches);
