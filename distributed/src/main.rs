use attentiondb_distributed::{
    ChaosTester, KubernetesOperator, RaftNode, ReadReplica, ReplicaManager, Shard, ShardManager,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     AttentionDB Phase 7 — Distributed + Replication        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("→ Sharding:");
    let mut shard_manager = ShardManager::new();
    shard_manager.add_shard(Shard::new(
        1,
        vec!["semantic".into(), "temporal".into()],
        "10.0.0.1:7400",
    ));
    shard_manager.add_shard(Shard::new(
        2,
        vec!["structural".into(), "relational".into()],
        "10.0.0.2:7400",
    ));
    shard_manager.add_shard(Shard::new(
        3,
        vec!["field_specific".into()],
        "10.0.0.3:7400",
    ));

    for id in shard_manager.list_shards() {
        let s = shard_manager.get_shard(id).unwrap();
        println!("   Shard {} @ {} — heads: {:?}", s.id, s.address, s.heads);
    }

    println!("\n→ Raft Consensus:");
    let mut raft = RaftNode::new(1, vec![2, 3]);
    let idx = raft.append_entry("INSERT", vec![1, 2, 3]).unwrap();
    raft.commit_up_to(idx);
    println!(
        "   Node {} | Term {} | Log entries: {} | Peers: {:?} | Committed: {}",
        raft.id,
        raft.current_term,
        raft.log_len(),
        raft.peers,
        raft.commit_index
    );

    println!("\n→ Read Replicas:");
    let mut replica_manager = ReplicaManager::new();
    replica_manager.add_replica(ReadReplica::new(1, 1, "10.0.0.11:7400"));
    replica_manager.add_replica(ReadReplica::new(2, 1, "10.0.0.12:7400"));
    replica_manager.add_replica(ReadReplica::new(3, 2, "10.0.1.11:7400"));
    println!("   Total replicas: {}", replica_manager.total_replicas());
    println!(
        "   Healthy replicas for shard 1: {}",
        replica_manager.get_healthy_replicas(1).len()
    );

    println!("\n→ Kubernetes Operator:");
    let mut operator = KubernetesOperator::new("attentiondb", "attentiondb-cluster");
    drop(operator.deploy(3));
    drop(operator.scale(5));
    println!(
        "   Deployed '{}' in '{}' — {} replicas",
        operator.deployment_name,
        operator.namespace,
        operator.get_replicas()
    );

    println!("\n→ Chaos Testing:");
    let chaos = ChaosTester::new(5);
    let killed = chaos.kill_random_node();
    let available = chaos.verify_availability(4);
    println!(
        "   Killed node {} | Cluster available: {}",
        killed, available
    );

    println!("\n✅ Phase 7 demo completed successfully.");
}
