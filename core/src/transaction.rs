use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::Mutex;
use attentiondb_storage::Record;
use crate::error::CoreError;

#[derive(Debug)]
pub enum TxnOp {
    Insert(Record),
    Delete(uuid::Uuid),
}

#[derive(Debug)]
pub struct Transaction {
    pub collection_name: String,
    pub operations: Vec<TxnOp>,
}

impl Transaction {
    pub fn new(collection_name: String) -> Self {
        Self {
            collection_name,
            operations: Vec::new(),
        }
    }
}

pub struct TransactionManager {
    transactions: Mutex<HashMap<u64, Transaction>>,
    next_id: AtomicU64,
}

impl TransactionManager {
    pub fn new() -> Self {
        Self {
            transactions: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    pub fn begin_transaction(&self, collection_name: &str) -> u64 {
        let txn_id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let mut txns = self.transactions.lock();
        txns.insert(txn_id, Transaction::new(collection_name.to_string()));
        txn_id
    }

    pub fn record_operation(&self, txn_id: u64, op: TxnOp) -> Result<(), CoreError> {
        let mut txns = self.transactions.lock();
        if let Some(txn) = txns.get_mut(&txn_id) {
            txn.operations.push(op);
            Ok(())
        } else {
            Err(CoreError::InvalidOperation(format!("Transaction {} not found", txn_id)))
        }
    }

    pub fn rollback_transaction(&self, txn_id: u64) -> Result<bool, CoreError> {
        let mut txns = self.transactions.lock();
        Ok(txns.remove(&txn_id).is_some())
    }

    pub fn get_staged_transaction(&self, txn_id: u64) -> Option<Transaction> {
        let mut txns = self.transactions.lock();
        txns.remove(&txn_id)
    }
}
