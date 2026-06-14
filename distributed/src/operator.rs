pub struct KubernetesOperator {
    pub namespace: String,
    pub deployment_name: String,
    pub current_replicas: u32,
}

impl KubernetesOperator {
    pub fn new(namespace: &str, name: &str) -> Self {
        Self {
            namespace: namespace.to_string(),
            deployment_name: name.to_string(),
            current_replicas: 3,
        }
    }

    pub fn deploy(&mut self, replicas: u32) {
        self.current_replicas = replicas;
        println!("[K8s] Deploying '{}' in namespace '{}' with {} replicas",
                 self.deployment_name, self.namespace, replicas);
    }

    pub fn scale(&mut self, replicas: u32) {
        self.current_replicas = replicas;
        println!("[K8s] Scaling '{}' to {} replicas", self.deployment_name, replicas);
    }

    pub fn rolling_update(&self) {
        println!("[K8s] Rolling update of '{}'", self.deployment_name);
    }

    pub fn get_replicas(&self) -> u32 {
        self.current_replicas
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
    fn test_deploy_and_scale() {
        let mut op = KubernetesOperator::new("prod", "attentiondb");
        op.deploy(5);
        assert_eq!(op.get_replicas(), 5);
        op.scale(10);
        assert_eq!(op.get_replicas(), 10);
    }
}
