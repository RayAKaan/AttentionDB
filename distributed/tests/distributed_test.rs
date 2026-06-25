use attentiondb_distributed::{
    ChaosTester, KubernetesOperator, RaftNode, ReadReplica, ReplicaManager, Shard, ShardManager,
};
#[test]
fn test_shard_create_and_find() {
    let mut sm = ShardManager::new();
    sm.add_shard(Shard::new(1, vec!["a".into(), "b".into()], "addr1"));
    sm.add_shard(Shard::new(2, vec!["c".into()], "addr2"));
    assert_eq!(sm.get_shard_for_head("b").unwrap().id, 1);
    assert_eq!(sm.get_shard_for_head("c").unwrap().id, 2);
    assert!(sm.get_shard_for_head("nonexistent").is_none());
}

#[test]
fn test_raft_append_and_commit() {
    let mut raft = RaftNode::new(1, vec![2]);
    assert_eq!(raft.append_entry("INSERT", vec![1]).unwrap(), 0);
    assert_eq!(raft.append_entry("DELETE", vec![2]).unwrap(), 1);
    raft.commit_up_to(1);
    assert_eq!(raft.commit_index, 1);
}

#[test]
fn test_replica_health_tracking() {
    let mut rm = ReplicaManager::new();
    let r1 = ReadReplica::new(1, 1, "addr1");
    let mut r2 = ReadReplica::new(2, 1, "addr2");
    r2.mark_unhealthy();
    rm.add_replica(r1);
    rm.add_replica(r2);
    assert_eq!(rm.get_healthy_replicas(1).len(), 1);
}

#[test]
fn test_operator_deploy_scale() {
    let mut op = KubernetesOperator::new("ns", "cluster");
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(op.deploy(3));
    assert_eq!(op.get_replicas(), 3);
    rt.block_on(op.scale(7));
    assert_eq!(op.get_replicas(), 7);
}

#[test]
fn test_chaos_kill_node() {
    let chaos = ChaosTester::new(10);
    for _ in 0..50 {
        let killed = chaos.kill_random_node();
        assert!(killed < 10);
    }
}

#[test]
fn test_chaos_availability() {
    let chaos = ChaosTester::new(6);
    assert!(chaos.verify_availability(3));
    assert!(!chaos.verify_availability(2));
}

#[test]
fn test_multi_shard_assignment() {
    let mut sm = ShardManager::new();
    for i in 0..5 {
        sm.add_shard(Shard::new(
            i,
            vec![],
            &format!("10.0.{}.{}:7400", i / 256, i % 256),
        ));
    }
    assert_eq!(sm.shard_count(), 5);
}

#[test]
fn test_raft_no_peers() {
    let raft = RaftNode::new(1, vec![]);
    let msgs = raft.broadcast_append_entries();
    assert!(msgs.is_empty());
}

#[test]
fn test_replica_apply_log() {
    let mut r = ReadReplica::new(1, 1, "addr");
    r.apply_log(100);
    assert_eq!(r.last_applied_index, 100);
}
