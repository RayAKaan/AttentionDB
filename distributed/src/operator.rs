use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AttentionDBClusterSpec {
    pub replicas: u32,
    pub storage_class: Option<String>,
    pub storage_size: String,
    pub version: String,
    pub enable_gpu_reranking: bool,
    pub enable_auto_reprojection: bool,
}

impl Default for AttentionDBClusterSpec {
    fn default() -> Self {
        Self {
            replicas: 3,
            storage_class: Some("standard".to_string()),
            storage_size: "100Gi".to_string(),
            version: "v0.5.0".to_string(),
            enable_gpu_reranking: true,
            enable_auto_reprojection: true,
        }
    }
}

pub struct KubernetesOperator {
    pub namespace: String,
    pub deployment_name: String,
    pub current_replicas: u32,
    pub active_spec: AttentionDBClusterSpec,
}

impl KubernetesOperator {
    pub fn new(namespace: &str, name: &str) -> Self {
        Self {
            namespace: namespace.to_string(),
            deployment_name: name.to_string(),
            current_replicas: 3,
            active_spec: AttentionDBClusterSpec::default(),
        }
    }

    pub fn with_spec(mut self, spec: AttentionDBClusterSpec) -> Self {
        self.current_replicas = spec.replicas;
        self.active_spec = spec;
        self
    }

    pub async fn deploy(&mut self, replicas: u32) {
        self.current_replicas = replicas;
        self.active_spec.replicas = replicas;
        info!(
            "[Operator] Deploying '{}' to {} replicas",
            self.deployment_name, replicas
        );
    }

    pub async fn scale(&mut self, replicas: u32) {
        self.current_replicas = replicas;
        self.active_spec.replicas = replicas;
        info!(
            "[Operator] Scaling '{}' to {} replicas",
            self.deployment_name, replicas
        );
    }

    pub async fn reconcile(
        &mut self,
        desired_spec: AttentionDBClusterSpec,
    ) -> Result<bool, String> {
        let mut changed = false;
        if self.current_replicas != desired_spec.replicas {
            self.scale(desired_spec.replicas).await;
            changed = true;
        }
        if self.active_spec.version != desired_spec.version {
            self.active_spec.version = desired_spec.version.clone();
            info!(
                "[Operator] Rolling update '{}' to {}",
                self.deployment_name, desired_spec.version
            );
            changed = true;
        }
        self.active_spec = desired_spec;
        self.current_replicas = self.active_spec.replicas;
        Ok(changed)
    }

    pub fn rolling_update(&self) {
        info!("[Operator] Rolling update of '{}'", self.deployment_name);
    }

    pub fn get_replicas(&self) -> u32 {
        self.current_replicas
    }

    pub fn generate_stateful_set_spec(&self) -> serde_json::Value {
        let mut labels = HashMap::new();
        labels.insert("app".to_string(), self.deployment_name.clone());
        let mut resources = serde_json::json!({
            "requests": { "cpu": "2", "memory": "4Gi" },
            "limits": { "cpu": "8", "memory": "16Gi" }
        });
        if self.active_spec.enable_gpu_reranking {
            resources["requests"]["nvidia.com/gpu"] = serde_json::json!("1");
            resources["limits"]["nvidia.com/gpu"] = serde_json::json!("1");
        }
        serde_json::json!({
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": { "name": self.deployment_name, "namespace": self.namespace, "labels": labels },
            "spec": {
                "replicas": self.active_spec.replicas,
                "serviceName": format!("{}-headless", self.deployment_name),
                "selector": { "matchLabels": labels },
                "template": {
                    "metadata": { "labels": labels },
                    "spec": {
                        "containers": [{
                            "name": "attentiondb-node",
                            "image": format!("rayakaan/attentiondb:{}", self.active_spec.version),
                            "imagePullPolicy": "IfNotPresent",
                            "ports": [
                                { "containerPort": 7400, "name": "grpc" },
                                { "containerPort": 8080, "name": "rest" },
                                { "containerPort": 7401, "name": "raft" }
                            ],
                            "env": [
                                { "name": "POD_NAME", "valueFrom": { "fieldRef": { "fieldPath": "metadata.name" } } },
                                { "name": "CLUSTER_SERVICE_NAME", "value": format!("{}-headless", self.deployment_name) }
                            ],
                            "resources": resources,
                            "volumeMounts": [{ "name": "storage", "mountPath": "/storage" }]
                        }]
                    }
                },
                "volumeClaimTemplates": [{
                    "metadata": { "name": "storage" },
                    "spec": {
                        "accessModes": ["ReadWriteOnce"],
                        "storageClassName": self.active_spec.storage_class,
                        "resources": { "requests": { "storage": self.active_spec.storage_size } }
                    }
                }]
            }
        })
    }

    pub fn generate_headless_service_spec(&self) -> serde_json::Value {
        let mut labels = HashMap::new();
        labels.insert("app".to_string(), self.deployment_name.clone());
        serde_json::json!({
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": { "name": format!("{}-headless", self.deployment_name), "namespace": self.namespace, "labels": labels },
            "spec": { "clusterIP": "None", "selector": labels, "ports": [
                { "port": 7400, "targetPort": 7400, "name": "grpc" },
                { "port": 7401, "targetPort": 7401, "name": "raft" }
            ]}
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_creation() {
        let op = KubernetesOperator::new("default", "attentiondb-cluster");
        assert_eq!(op.namespace, "default");
    }

    #[test]
    fn test_generate_stateful_set() {
        let op = KubernetesOperator::new("prod", "adb");
        let s = op.generate_stateful_set_spec();
        assert_eq!(s["kind"], "StatefulSet");
        assert_eq!(s["spec"]["replicas"], 3);
    }

    #[test]
    fn test_generate_headless_service() {
        let op = KubernetesOperator::new("prod", "adb");
        let s = op.generate_headless_service_spec();
        assert_eq!(s["kind"], "Service");
        assert_eq!(s["spec"]["clusterIP"], "None");
    }

    #[test]
    fn test_reconcile_updates_state() {
        let mut op = KubernetesOperator::new("prod", "adb-cluster");
        let desired = AttentionDBClusterSpec {
            replicas: 7,
            storage_class: Some("fast-ssd".to_string()),
            storage_size: "500Gi".to_string(),
            version: "v1.0.0".to_string(),
            enable_gpu_reranking: true,
            enable_auto_reprojection: true,
        };
        let rt = tokio::runtime::Runtime::new().unwrap();
        let changed = rt.block_on(op.reconcile(desired.clone())).unwrap();
        assert!(changed);
        assert_eq!(op.get_replicas(), 7);
        assert_eq!(op.active_spec.version, "v1.0.0");
    }
}
