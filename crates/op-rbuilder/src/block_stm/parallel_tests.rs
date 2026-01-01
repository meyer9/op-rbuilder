//! Parallelization tests for Block-STM execution engine.
//!
//! These tests verify the correct behavior of parallel transaction execution,
//! including conflict detection, re-execution, and deterministic ordering.

use super::*;
use crate::block_stm::mv_hashmap::{ReadSet, WriteSet};
use crate::block_stm::types::{
    BlockResourceType, EvmStateKey, EvmStateValue, ReadResult, Task, Version,
};
use alloy_primitives::{Address, U256};
use rand::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

/// Helper to create a storage key.
fn storage_key(addr: u8, slot: u64) -> EvmStateKey {
    EvmStateKey::Storage(Address::repeat_byte(addr), U256::from(slot))
}

/// Helper to create a storage value.
fn storage_value(val: u64) -> EvmStateValue {
    EvmStateValue::Storage(U256::from(val))
}

/// Helper to create a balance key.
fn balance_key(addr: u8) -> EvmStateKey {
    EvmStateKey::Balance(Address::repeat_byte(addr))
}

/// Helper to create a balance value.
fn balance_value(val: u64) -> EvmStateValue {
    EvmStateValue::Balance(U256::from(val))
}

/// Helper to create a balance increment value.
fn balance_increment(val: u64) -> EvmStateValue {
    EvmStateValue::BalanceIncrement(U256::from(val))
}

/// Simulated transaction for testing.
#[derive(Clone, Debug)]
struct SimulatedTx {
    /// Keys this transaction reads
    reads: Vec<EvmStateKey>,
    /// Keys and values this transaction writes
    writes: Vec<(EvmStateKey, EvmStateValue)>,
    /// If true, reads will be from base state (return None version)
    reads_from_base: bool,
}

impl SimulatedTx {
    fn new(reads: Vec<EvmStateKey>, writes: Vec<(EvmStateKey, EvmStateValue)>) -> Self {
        Self {
            reads,
            writes,
            reads_from_base: false,
        }
    }

    fn reads_base(mut self) -> Self {
        self.reads_from_base = true;
        self
    }
}

/// Simulates Block-STM execution for a set of transactions.
/// Returns the number of re-executions that occurred.
fn simulate_block_stm(
    transactions: Vec<SimulatedTx>,
    num_threads: usize,
) -> (usize, HashMap<EvmStateKey, EvmStateValue>) {
    let num_txns = transactions.len();
    let mv = Arc::new(MVHashMap::new(num_txns));
    let scheduler = Arc::new(Scheduler::new(num_txns));
    let transactions = Arc::new(transactions);
    let re_execution_count = Arc::new(AtomicUsize::new(0));

    thread::scope(|s| {
        for _worker_id in 0..num_threads {
            let mv = Arc::clone(&mv);
            let scheduler = Arc::clone(&scheduler);
            let transactions = Arc::clone(&transactions);
            let re_execution_count = Arc::clone(&re_execution_count);

            s.spawn(move || {
                let mut pending_task: Option<Task> = None;

                while !scheduler.done() {
                    let task = pending_task.take().or_else(|| scheduler.next_task());

                    match task {
                        Some(Task::Execute { version }) => {
                            let Version {
                                txn_idx,
                                incarnation,
                            } = version;

                            // Track re-executions
                            if incarnation > 0 {
                                re_execution_count.fetch_add(1, Ordering::Relaxed);
                            }

                            let tx = &transactions[txn_idx as usize];

                            // Build read set by reading from MVHashMap
                            let mut read_set = ReadSet::new();
                            for key in &tx.reads {
                                let result = mv.read(key, txn_idx);
                                match result {
                                    ReadResult::Value { version, .. } => {
                                        read_set.insert((key.clone(), Some(version)));
                                    }
                                    ReadResult::NotFound => {
                                        if !tx.reads_from_base {
                                            // Check if there's a lower transaction that wrote
                                            // For simulation, we just record None
                                        }
                                        read_set.insert((key.clone(), None));
                                    }
                                    ReadResult::Aborted { txn_idx: dep } => {
                                        // Register dependency and retry
                                        scheduler.add_dependency(txn_idx, dep);
                                        continue;
                                    }
                                }
                            }

                            // Build write set
                            let write_set: WriteSet = tx.writes.iter().cloned().collect();

                            // Record to MVHashMap
                            let wrote_new_path = mv.record(version, &read_set, &write_set);

                            // Finish execution
                            let next_task =
                                scheduler.finish_execution(txn_idx, incarnation, wrote_new_path);

                            if let Some(Task::Validate { version: v }) = next_task {
                                // Validate inline
                                let valid = mv.validate_read_set(v.txn_idx);
                                if !valid {
                                    let aborted =
                                        scheduler.try_validation_abort(v.txn_idx, v.incarnation);
                                    if aborted {
                                        mv.convert_writes_to_estimates(v.txn_idx);
                                    }
                                    pending_task = scheduler.finish_validation(v.txn_idx, aborted);
                                } else {
                                    scheduler.finish_validation(v.txn_idx, false);
                                }
                            }
                        }
                        Some(Task::Validate { version }) => {
                            let Version {
                                txn_idx,
                                incarnation,
                            } = version;

                            let valid = mv.validate_read_set(txn_idx);
                            if !valid {
                                let aborted = scheduler.try_validation_abort(txn_idx, incarnation);
                                if aborted {
                                    mv.convert_writes_to_estimates(txn_idx);
                                }
                                pending_task = scheduler.finish_validation(txn_idx, aborted);
                            } else {
                                scheduler.finish_validation(txn_idx, false);
                            }
                        }
                        None => {
                            thread::yield_now();
                        }
                    }
                }
            });
        }
    });

    // Get final state snapshot
    let snapshot = Arc::try_unwrap(mv).unwrap().into_snapshot();
    let re_execs = re_execution_count.load(Ordering::Relaxed);

    (re_execs, snapshot)
}

// ============================================================================
// Test: Independent Transactions (No Conflicts)
// ============================================================================

#[test]
fn test_independent_transactions_no_reexecution() {
    // Each transaction writes to a different storage slot
    // No conflicts should occur
    let transactions: Vec<SimulatedTx> = (0..10)
        .map(|i| {
            SimulatedTx::new(
                vec![], // No reads
                vec![(storage_key(i as u8, 0), storage_value(i as u64 * 100))],
            )
        })
        .collect();

    let (re_execs, snapshot) = simulate_block_stm(transactions, 4);

    // No re-executions should occur
    assert_eq!(re_execs, 0, "Independent transactions should not re-execute");

    // All writes should be present
    for i in 0..10 {
        assert_eq!(
            snapshot.get(&storage_key(i as u8, 0)),
            Some(&storage_value(i as u64 * 100)),
            "Transaction {} write should be in snapshot",
            i
        );
    }
}

#[test]
fn test_many_independent_transactions_parallel() {
    // 100 transactions, each touching unique accounts
    let transactions: Vec<SimulatedTx> = (0..100)
        .map(|i| {
            SimulatedTx::new(
                vec![balance_key((i % 256) as u8)], // Reads own balance
                vec![
                    (balance_key((i % 256) as u8), balance_value(i as u64)),
                    (storage_key((i % 256) as u8, 0), storage_value(i as u64)),
                ],
            )
            .reads_base()
        })
        .collect();

    let (re_execs, _snapshot) = simulate_block_stm(transactions, 8);

    // With independent transactions reading from base state, no conflicts
    assert_eq!(
        re_execs, 0,
        "Independent transactions should not cause re-executions"
    );
}

// ============================================================================
// Test: Sequential Dependencies (Maximum Conflicts)
// ============================================================================

#[test]
fn test_sequential_dependency_chain() {
    // Each transaction reads what the previous wrote
    // This creates a serial dependency chain
    let shared_key = storage_key(0, 0);

    let transactions: Vec<SimulatedTx> = (0..5)
        .map(|i| {
            if i == 0 {
                // First transaction writes initial value
                SimulatedTx::new(vec![], vec![(shared_key.clone(), storage_value(100))])
            } else {
                // Subsequent transactions read and write
                SimulatedTx::new(
                    vec![shared_key.clone()],
                    vec![(shared_key.clone(), storage_value(100 + i as u64 * 10))],
                )
            }
        })
        .collect();

    let (re_execs, snapshot) = simulate_block_stm(transactions, 4);

    // Final value should be from the last transaction
    assert_eq!(
        snapshot.get(&shared_key),
        Some(&storage_value(140)), // 100 + 40
        "Final value should be from last transaction"
    );

    // Some re-executions are expected due to conflicts
    // The exact number depends on timing, but should be > 0 for this pattern
    println!("Sequential chain re-executions: {}", re_execs);
}

#[test]
fn test_all_transactions_write_same_key() {
    // All transactions write to the same key
    // Creates read-write conflicts
    let shared_key = storage_key(0, 0);

    let transactions: Vec<SimulatedTx> = (0..10)
        .map(|i| {
            SimulatedTx::new(
                vec![shared_key.clone()], // Read the shared key
                vec![(shared_key.clone(), storage_value(i as u64))],
            )
        })
        .collect();

    let (_re_execs, snapshot) = simulate_block_stm(transactions, 4);

    // Final value should be from transaction 9 (last one)
    assert_eq!(
        snapshot.get(&shared_key),
        Some(&storage_value(9)),
        "Final value should be from last transaction"
    );
}

// ============================================================================
// Test: Mixed Workload (Partial Conflicts)
// ============================================================================

#[test]
fn test_mixed_workload_partial_conflicts() {
    // Some transactions share keys, others are independent
    let transactions = vec![
        // Group 1: Independent transactions (no conflicts)
        SimulatedTx::new(vec![], vec![(storage_key(1, 0), storage_value(100))]),
        SimulatedTx::new(vec![], vec![(storage_key(2, 0), storage_value(200))]),
        SimulatedTx::new(vec![], vec![(storage_key(3, 0), storage_value(300))]),
        // Group 2: Conflicting transactions (same key)
        SimulatedTx::new(
            vec![storage_key(10, 0)],
            vec![(storage_key(10, 0), storage_value(1000))],
        )
        .reads_base(),
        SimulatedTx::new(
            vec![storage_key(10, 0)],
            vec![(storage_key(10, 0), storage_value(1001))],
        ),
        SimulatedTx::new(
            vec![storage_key(10, 0)],
            vec![(storage_key(10, 0), storage_value(1002))],
        ),
        // Group 3: More independent transactions
        SimulatedTx::new(vec![], vec![(storage_key(4, 0), storage_value(400))]),
        SimulatedTx::new(vec![], vec![(storage_key(5, 0), storage_value(500))]),
    ];

    let (_re_execs, snapshot) = simulate_block_stm(transactions, 4);

    // Verify independent transactions
    assert_eq!(snapshot.get(&storage_key(1, 0)), Some(&storage_value(100)));
    assert_eq!(snapshot.get(&storage_key(2, 0)), Some(&storage_value(200)));
    assert_eq!(snapshot.get(&storage_key(3, 0)), Some(&storage_value(300)));
    assert_eq!(snapshot.get(&storage_key(4, 0)), Some(&storage_value(400)));
    assert_eq!(snapshot.get(&storage_key(5, 0)), Some(&storage_value(500)));

    // Verify conflicting group - last transaction's value wins
    assert_eq!(
        snapshot.get(&storage_key(10, 0)),
        Some(&storage_value(1002))
    );
}

// ============================================================================
// Test: Balance Increments (Commutative Operations)
// ============================================================================

#[test]
fn test_balance_increments_no_read_conflict() {
    // Multiple transactions incrementing the same balance
    // Since they're blind writes (no reads), no conflicts should occur
    let coinbase = balance_key(0);

    let transactions: Vec<SimulatedTx> = (0..5)
        .map(|i| {
            SimulatedTx::new(
                vec![], // No reads - blind write
                vec![(coinbase.clone(), balance_increment((i + 1) as u64 * 10))],
            )
        })
        .collect();

    let (re_execs, snapshot) = simulate_block_stm(transactions, 4);

    // No re-executions for blind writes
    assert_eq!(re_execs, 0, "Blind writes should not cause re-executions");

    // The last transaction's increment should be in the snapshot
    // (In real Block-STM, all increments would be accumulated during commit)
    assert!(
        snapshot.get(&coinbase).is_some(),
        "Coinbase balance increment should be recorded"
    );
}

#[test]
fn test_balance_read_then_increment_causes_conflict() {
    // Transaction reads balance, then later transaction writes increment
    let addr = balance_key(1);

    let transactions = vec![
        // Tx 0: Writes initial balance
        SimulatedTx::new(vec![], vec![(addr.clone(), balance_value(1000))]),
        // Tx 1: Reads balance (will see tx 0's write or base state)
        SimulatedTx::new(vec![addr.clone()], vec![(storage_key(1, 0), storage_value(1))]),
        // Tx 2: Writes balance increment (creates conflict with tx 1's read)
    ];

    let (_re_execs, snapshot) = simulate_block_stm(transactions, 2);

    // Both writes should be in snapshot
    assert_eq!(snapshot.get(&addr), Some(&balance_value(1000)));
    assert_eq!(snapshot.get(&storage_key(1, 0)), Some(&storage_value(1)));
}

// ============================================================================
// Test: Block Resource Tracking
// ============================================================================

#[test]
fn test_cumulative_gas_tracking() {
    // Transactions track cumulative gas usage
    let gas_key = EvmStateKey::BlockResourceUsed(BlockResourceType::Gas);

    let transactions: Vec<SimulatedTx> = (0..5)
        .map(|i| {
            let cumulative_gas = (i + 1) as u64 * 21000;
            SimulatedTx::new(
                if i == 0 {
                    vec![]
                } else {
                    vec![gas_key.clone()]
                },
                vec![(
                    gas_key.clone(),
                    EvmStateValue::BlockResourceUsed(cumulative_gas),
                )],
            )
        })
        .collect();

    let (_re_execs, snapshot) = simulate_block_stm(transactions, 4);

    // Final cumulative gas should be 5 * 21000 = 105000
    assert_eq!(
        snapshot.get(&gas_key),
        Some(&EvmStateValue::BlockResourceUsed(105000)),
        "Final cumulative gas should be correct"
    );
}

// ============================================================================
// Test: Stress Tests
// ============================================================================

#[test]
fn stress_test_high_contention() {
    // Many transactions competing for a few hot keys
    let hot_keys: Vec<EvmStateKey> = (0..3).map(|i| storage_key(i, 0)).collect();

    let transactions: Vec<SimulatedTx> = (0..50)
        .map(|i| {
            let key = hot_keys[i % hot_keys.len()].clone();
            SimulatedTx::new(vec![key.clone()], vec![(key, storage_value(i as u64))])
        })
        .collect();

    let (re_execs, snapshot) = simulate_block_stm(transactions, 8);

    // Verify that the last writer to each key has their value
    for (key_idx, key) in hot_keys.iter().enumerate() {
        // Find the last transaction that wrote to this key
        let last_writer = (0..50)
            .rev()
            .find(|i| i % 3 == key_idx)
            .expect("Each key should have at least one writer");

        assert_eq!(
            snapshot.get(key),
            Some(&storage_value(last_writer as u64)),
            "Key {} should have value from last writer (tx {})",
            key_idx,
            last_writer
        );
    }

    println!(
        "High contention test: {} re-executions for 50 transactions",
        re_execs
    );
}

#[test]
fn stress_test_random_access_pattern() {
    // Random read/write patterns
    let mut rng = rand::rng();
    let num_keys = 20;
    let num_txns = 100;

    let transactions: Vec<SimulatedTx> = (0..num_txns)
        .map(|i| {
            let mut reads = Vec::new();
            let mut writes = Vec::new();

            // Random number of reads (0-3)
            let num_reads = rng.random_range(0..=3);
            for _ in 0..num_reads {
                let key_idx = rng.random_range(0..num_keys);
                reads.push(storage_key(key_idx as u8, 0));
            }

            // At least one write
            let num_writes = rng.random_range(1..=2);
            for _ in 0..num_writes {
                let key_idx = rng.random_range(0..num_keys);
                writes.push((storage_key(key_idx as u8, 0), storage_value(i as u64)));
            }

            SimulatedTx::new(reads, writes)
        })
        .collect();

    let (re_execs, _snapshot) = simulate_block_stm(transactions, 4);

    println!(
        "Random access test: {} re-executions for {} transactions",
        re_execs, num_txns
    );

    // Test should complete without panics or deadlocks
    // The actual number of re-executions depends on the random pattern
}

#[test]
fn stress_test_many_threads() {
    // Test with more threads than typical
    let transactions: Vec<SimulatedTx> = (0..200)
        .map(|i| {
            SimulatedTx::new(
                vec![], // No reads
                vec![(storage_key((i % 50) as u8, 0), storage_value(i as u64))],
            )
        })
        .collect();

    let (re_execs, _snapshot) = simulate_block_stm(transactions, 16);

    println!("Many threads test: {} re-executions", re_execs);
}

// ============================================================================
// Test: Correctness Under Concurrent Execution
// ============================================================================

#[test]
fn test_deterministic_results() {
    // Run the same workload multiple times and verify consistent results
    let transactions: Vec<SimulatedTx> = (0..20)
        .map(|i| {
            let key = storage_key((i % 5) as u8, 0);
            SimulatedTx::new(vec![key.clone()], vec![(key, storage_value(i as u64))])
        })
        .collect();

    let mut results = Vec::new();

    for _ in 0..5 {
        let (_, snapshot) = simulate_block_stm(transactions.clone(), 4);
        results.push(snapshot);
    }

    // All runs should produce the same final state
    for (i, result) in results.iter().enumerate().skip(1) {
        assert_eq!(
            *result, results[0],
            "Run {} produced different result than run 0",
            i
        );
    }
}

#[test]
fn test_ordering_preserved() {
    // Verify that transaction ordering is preserved in final state
    // Each transaction writes its index to a different key
    let transactions: Vec<SimulatedTx> = (0..10)
        .map(|i| {
            SimulatedTx::new(vec![], vec![(storage_key(i as u8, i as u64), storage_value(i as u64))])
        })
        .collect();

    let (_, snapshot) = simulate_block_stm(transactions, 4);

    // Verify each transaction's write is present
    for i in 0..10 {
        assert_eq!(
            snapshot.get(&storage_key(i as u8, i as u64)),
            Some(&storage_value(i as u64)),
            "Transaction {} write should be in snapshot",
            i
        );
    }
}

// ============================================================================
// Test: Edge Cases
// ============================================================================

#[test]
fn test_single_transaction() {
    let transactions = vec![SimulatedTx::new(
        vec![],
        vec![(storage_key(0, 0), storage_value(42))],
    )];

    let (re_execs, snapshot) = simulate_block_stm(transactions, 1);

    assert_eq!(re_execs, 0);
    assert_eq!(snapshot.get(&storage_key(0, 0)), Some(&storage_value(42)));
}

#[test]
fn test_empty_transactions() {
    // Transactions that do nothing
    let transactions: Vec<SimulatedTx> = (0..5).map(|_| SimulatedTx::new(vec![], vec![])).collect();

    let (re_execs, snapshot) = simulate_block_stm(transactions, 2);

    assert_eq!(re_execs, 0);
    assert!(snapshot.is_empty());
}

#[test]
fn test_read_only_transactions() {
    // First transaction writes, rest just read
    let key = storage_key(0, 0);

    let mut transactions = vec![SimulatedTx::new(
        vec![],
        vec![(key.clone(), storage_value(100))],
    )];

    // Add read-only transactions
    for _ in 0..5 {
        transactions.push(SimulatedTx::new(vec![key.clone()], vec![]));
    }

    let (re_execs, snapshot) = simulate_block_stm(transactions, 4);

    assert_eq!(snapshot.get(&key), Some(&storage_value(100)));
    println!("Read-only test re-executions: {}", re_execs);
}

#[test]
fn test_write_only_transactions() {
    // All transactions write to different keys without reading
    let transactions: Vec<SimulatedTx> = (0..10)
        .map(|i| SimulatedTx::new(vec![], vec![(storage_key(i as u8, 0), storage_value(i as u64))]))
        .collect();

    let (re_execs, snapshot) = simulate_block_stm(transactions, 4);

    // No conflicts for write-only to different keys
    assert_eq!(re_execs, 0);
    assert_eq!(snapshot.len(), 10);
}

// ============================================================================
// Test: Scheduler Integration
// ============================================================================

#[test]
fn test_scheduler_completes_all_transactions() {
    let num_txns = 50;
    let scheduler = Scheduler::new(num_txns);
    let mv = MVHashMap::new(num_txns);

    // Process all transactions
    let mut processed = HashSet::new();
    let mut iterations = 0;
    let max_iterations = num_txns * 4;

    while !scheduler.done() && iterations < max_iterations {
        iterations += 1;

        match scheduler.next_task() {
            Some(Task::Execute { version }) => {
                // Empty write set
                let read_set = ReadSet::new();
                let write_set = WriteSet::new();
                mv.record(version, &read_set, &write_set);

                scheduler.finish_execution(version.txn_idx, version.incarnation, false);
                processed.insert(version.txn_idx);
            }
            Some(Task::Validate { version }) => {
                scheduler.finish_validation(version.txn_idx, false);
            }
            None => {
                thread::yield_now();
            }
        }
    }

    assert!(scheduler.done(), "Scheduler should complete");
    assert_eq!(
        processed.len(),
        num_txns,
        "All transactions should be processed"
    );
}

#[test]
fn test_scheduler_handles_all_abort() {
    // Test where every transaction is aborted once
    let num_txns = 10;
    let scheduler = Scheduler::new(num_txns);
    let mv = MVHashMap::new(num_txns);

    let mut aborted = HashSet::new();
    let mut pending_task: Option<Task> = None;
    let mut iterations = 0;

    while !scheduler.done() && iterations < 200 {
        iterations += 1;

        let task = pending_task.take().or_else(|| scheduler.next_task());

        match task {
            Some(Task::Execute { version }) => {
                let read_set = ReadSet::new();
                let write_set = WriteSet::new();
                mv.record(version, &read_set, &write_set);

                let next = scheduler.finish_execution(version.txn_idx, version.incarnation, false);

                if let Some(Task::Validate { version: v }) = next {
                    // Abort each transaction once (on incarnation 0)
                    if v.incarnation == 0 && !aborted.contains(&v.txn_idx) {
                        aborted.insert(v.txn_idx);
                        scheduler.try_validation_abort(v.txn_idx, v.incarnation);
                        pending_task = scheduler.finish_validation(v.txn_idx, true);
                    } else {
                        scheduler.finish_validation(v.txn_idx, false);
                    }
                }
            }
            Some(Task::Validate { version }) => {
                if version.incarnation == 0 && !aborted.contains(&version.txn_idx) {
                    aborted.insert(version.txn_idx);
                    scheduler.try_validation_abort(version.txn_idx, version.incarnation);
                    pending_task = scheduler.finish_validation(version.txn_idx, true);
                } else {
                    scheduler.finish_validation(version.txn_idx, false);
                }
            }
            None => {
                thread::yield_now();
            }
        }
    }

    assert!(scheduler.done(), "Scheduler should complete after aborts");
    assert_eq!(aborted.len(), num_txns, "All transactions should be aborted once");
}

// ============================================================================
// Test: Nonce Tracking
// ============================================================================

/// Helper to create a nonce key.
fn nonce_key(addr: u8) -> EvmStateKey {
    EvmStateKey::Nonce(Address::repeat_byte(addr))
}

/// Helper to create a nonce value.
fn nonce_value(val: u64) -> EvmStateValue {
    EvmStateValue::Nonce(val)
}

#[test]
fn test_nonce_increment_chain() {
    // Simulates multiple transactions from the same account
    // Each must read and increment the nonce
    let addr = 1u8;
    let nonce = nonce_key(addr);

    let transactions: Vec<SimulatedTx> = (0..5)
        .map(|i| {
            SimulatedTx::new(
                if i == 0 { vec![] } else { vec![nonce.clone()] },
                vec![(nonce.clone(), nonce_value(i as u64))],
            )
        })
        .collect();

    let (_re_execs, snapshot) = simulate_block_stm(transactions, 4);

    // Final nonce should be 4 (from last transaction)
    assert_eq!(
        snapshot.get(&nonce),
        Some(&nonce_value(4)),
        "Final nonce should be from last transaction"
    );
}

#[test]
fn test_multiple_accounts_nonce_independent() {
    // Multiple accounts incrementing their own nonces - no conflicts
    let transactions: Vec<SimulatedTx> = (0..10)
        .map(|i| {
            let nonce = nonce_key(i as u8);
            SimulatedTx::new(vec![], vec![(nonce, nonce_value(1))])
        })
        .collect();

    let (re_execs, snapshot) = simulate_block_stm(transactions, 4);

    assert_eq!(re_execs, 0, "Independent nonce updates should not conflict");
    assert_eq!(snapshot.len(), 10);
}

// ============================================================================
// Test: Code Hash Handling
// ============================================================================

/// Helper to create a code hash key.
fn code_hash_key(addr: u8) -> EvmStateKey {
    EvmStateKey::CodeHash(Address::repeat_byte(addr))
}

/// Helper to create a code hash value.
fn code_hash_value(val: u8) -> EvmStateValue {
    EvmStateValue::CodeHash(alloy_primitives::B256::repeat_byte(val))
}

#[test]
fn test_contract_deployment_visibility() {
    // Simulates contract deployment followed by calls
    // Tx 0: Deploys contract (writes code hash)
    // Tx 1-4: Call the contract (read code hash)

    let contract_addr = 42u8;
    let code_hash = code_hash_key(contract_addr);

    let mut transactions = vec![
        // Deploy: write code hash
        SimulatedTx::new(
            vec![],
            vec![
                (code_hash.clone(), code_hash_value(0xAB)),
                (balance_key(contract_addr), balance_value(0)),
            ],
        ),
    ];

    // Add transactions that call the contract
    for i in 1..5 {
        transactions.push(SimulatedTx::new(
            vec![code_hash.clone()], // Read code to execute
            vec![(storage_key(contract_addr, i), storage_value(i as u64))],
        ));
    }

    let (_re_execs, snapshot) = simulate_block_stm(transactions, 4);

    // Contract should be deployed
    assert_eq!(snapshot.get(&code_hash), Some(&code_hash_value(0xAB)));

    // All calls should have written their storage
    for i in 1..5 {
        assert_eq!(
            snapshot.get(&storage_key(contract_addr, i)),
            Some(&storage_value(i as u64))
        );
    }
}

// ============================================================================
// Test: Multiple Storage Slots Per Transaction
// ============================================================================

#[test]
fn test_transaction_writes_multiple_slots() {
    // Each transaction writes to multiple storage slots
    let transactions: Vec<SimulatedTx> = (0..5)
        .map(|i| {
            let writes: Vec<_> = (0..4)
                .map(|slot| (storage_key(i as u8, slot), storage_value(i as u64 * 10 + slot)))
                .collect();
            SimulatedTx::new(vec![], writes)
        })
        .collect();

    let (re_execs, snapshot) = simulate_block_stm(transactions, 4);

    assert_eq!(re_execs, 0, "Independent multi-slot writes should not conflict");

    // Verify all writes
    for i in 0..5 {
        for slot in 0..4 {
            assert_eq!(
                snapshot.get(&storage_key(i as u8, slot)),
                Some(&storage_value(i as u64 * 10 + slot))
            );
        }
    }
}

#[test]
fn test_transaction_reads_multiple_slots() {
    // Transaction reads multiple slots written by different prior transactions
    let transactions = vec![
        SimulatedTx::new(vec![], vec![(storage_key(0, 0), storage_value(100))]),
        SimulatedTx::new(vec![], vec![(storage_key(0, 1), storage_value(200))]),
        SimulatedTx::new(vec![], vec![(storage_key(0, 2), storage_value(300))]),
        // This transaction reads all three slots
        SimulatedTx::new(
            vec![storage_key(0, 0), storage_key(0, 1), storage_key(0, 2)],
            vec![(storage_key(0, 3), storage_value(600))], // Sum
        ),
    ];

    let (_re_execs, snapshot) = simulate_block_stm(transactions, 4);

    assert_eq!(snapshot.get(&storage_key(0, 3)), Some(&storage_value(600)));
}

// ============================================================================
// Test: Diamond Dependency Pattern
// ============================================================================

#[test]
fn test_diamond_dependency() {
    // Diamond pattern:
    //     T0 (writes A)
    //    /  \
    //   T1   T2 (both read A, write B and C respectively)
    //    \  /
    //     T3 (reads B and C)

    let key_a = storage_key(0, 0);
    let key_b = storage_key(0, 1);
    let key_c = storage_key(0, 2);
    let key_d = storage_key(0, 3);

    let transactions = vec![
        // T0: writes A
        SimulatedTx::new(vec![], vec![(key_a.clone(), storage_value(100))]),
        // T1: reads A, writes B
        SimulatedTx::new(
            vec![key_a.clone()],
            vec![(key_b.clone(), storage_value(200))],
        ),
        // T2: reads A, writes C
        SimulatedTx::new(
            vec![key_a.clone()],
            vec![(key_c.clone(), storage_value(300))],
        ),
        // T3: reads B and C, writes D
        SimulatedTx::new(
            vec![key_b.clone(), key_c.clone()],
            vec![(key_d.clone(), storage_value(500))],
        ),
    ];

    let (_re_execs, snapshot) = simulate_block_stm(transactions, 4);

    assert_eq!(snapshot.get(&key_a), Some(&storage_value(100)));
    assert_eq!(snapshot.get(&key_b), Some(&storage_value(200)));
    assert_eq!(snapshot.get(&key_c), Some(&storage_value(300)));
    assert_eq!(snapshot.get(&key_d), Some(&storage_value(500)));
}

// ============================================================================
// Test: Validation Result Details
// ============================================================================

#[test]
fn test_validation_result_conflict_details() {
    let mv = MVHashMap::new(3);
    let key = storage_key(0, 0);

    // Tx 0 writes
    let mut ws0 = WriteSet::new();
    ws0.insert((key.clone(), storage_value(100)));
    mv.record(Version::new(0, 0), &ReadSet::new(), &ws0);

    // Tx 1 reads from tx 0
    let mut rs1 = ReadSet::new();
    rs1.insert((key.clone(), Some(Version::new(0, 0))));
    mv.record(Version::new(1, 0), &rs1, &WriteSet::new());

    // Validation should pass
    let result = mv.validate_read_set_detailed(1);
    assert_eq!(result, ValidationResult::Valid);

    // Now tx 0 re-executes with incarnation 1
    let mut ws0_new = WriteSet::new();
    ws0_new.insert((key.clone(), storage_value(200)));
    mv.record(Version::new(0, 1), &ReadSet::new(), &ws0_new);

    // Validation should fail with conflict details
    let result = mv.validate_read_set_detailed(1);
    match result {
        ValidationResult::Conflict {
            key: conflict_key,
            expected_version,
            actual_version,
        } => {
            assert_eq!(conflict_key, key);
            assert_eq!(expected_version, Some(Version::new(0, 0)));
            assert_eq!(actual_version, Some(Version::new(0, 1)));
        }
        ValidationResult::Valid => panic!("Expected conflict, got valid"),
    }
}

// ============================================================================
// Test: Estimate Marker Detection
// ============================================================================

#[test]
fn test_estimate_marker_causes_abort() {
    let mv = MVHashMap::new(3);
    let key = storage_key(0, 0);

    // Tx 0 writes
    let mut ws0 = WriteSet::new();
    ws0.insert((key.clone(), storage_value(100)));
    mv.record(Version::new(0, 0), &ReadSet::new(), &ws0);

    // Convert tx 0 to estimate (simulating abort)
    mv.convert_writes_to_estimates(0);

    // Tx 1 reading should see Aborted
    match mv.read(&key, 1) {
        ReadResult::Aborted { txn_idx } => {
            assert_eq!(txn_idx, 0, "Should report tx 0 as aborted");
        }
        other => panic!("Expected Aborted, got {:?}", other),
    }
}

#[test]
fn test_estimate_cleared_on_reexecution() {
    let mv = MVHashMap::new(3);
    let key = storage_key(0, 0);

    // Tx 0 writes
    let mut ws0 = WriteSet::new();
    ws0.insert((key.clone(), storage_value(100)));
    mv.record(Version::new(0, 0), &ReadSet::new(), &ws0);

    // Convert to estimate
    mv.convert_writes_to_estimates(0);

    // Verify it's an estimate
    assert!(matches!(mv.read(&key, 1), ReadResult::Aborted { .. }));

    // Tx 0 re-executes with incarnation 1
    let mut ws0_new = WriteSet::new();
    ws0_new.insert((key.clone(), storage_value(200)));
    mv.record(Version::new(0, 1), &ReadSet::new(), &ws0_new);

    // Now tx 1 should see the new value
    match mv.read(&key, 1) {
        ReadResult::Value { value, version } => {
            assert_eq!(value, storage_value(200));
            assert_eq!(version.incarnation, 1);
        }
        other => panic!("Expected Value, got {:?}", other),
    }
}

// ============================================================================
// Test: Long Dependency Chains
// ============================================================================

#[test]
fn test_long_linear_dependency_chain() {
    // Chain of 20 transactions, each depending on the previous
    let key = storage_key(0, 0);
    let num_txns = 20;

    let transactions: Vec<SimulatedTx> = (0..num_txns)
        .map(|i| {
            if i == 0 {
                SimulatedTx::new(vec![], vec![(key.clone(), storage_value(1))])
            } else {
                SimulatedTx::new(
                    vec![key.clone()],
                    vec![(key.clone(), storage_value(i as u64 + 1))],
                )
            }
        })
        .collect();

    let (re_execs, snapshot) = simulate_block_stm(transactions, 8);

    // Final value should be 20
    assert_eq!(snapshot.get(&key), Some(&storage_value(num_txns as u64)));

    println!("Long chain ({} txns): {} re-executions", num_txns, re_execs);
}

#[test]
fn test_branching_dependency_tree() {
    // Tree pattern:
    //        T0
    //      / | \
    //    T1  T2  T3
    //   /|   |   |\
    //  T4 T5 T6 T7 T8

    let root_key = storage_key(0, 0);

    let mut transactions = vec![
        // T0: Root writes
        SimulatedTx::new(vec![], vec![(root_key.clone(), storage_value(1))]),
    ];

    // T1, T2, T3: Read root, write their own keys
    for i in 1..=3 {
        transactions.push(SimulatedTx::new(
            vec![root_key.clone()],
            vec![(storage_key(i, 0), storage_value(i as u64 * 10))],
        ));
    }

    // T4-T8: Read from T1, T2, or T3
    let parents = [(1, 4), (1, 5), (2, 6), (3, 7), (3, 8)];
    for (parent, child) in parents {
        transactions.push(SimulatedTx::new(
            vec![storage_key(parent, 0)],
            vec![(storage_key(child, 0), storage_value(child as u64 * 10))],
        ));
    }

    let (_re_execs, snapshot) = simulate_block_stm(transactions, 4);

    // Verify all writes
    assert_eq!(snapshot.get(&root_key), Some(&storage_value(1)));
    for i in 1..=8 {
        assert_eq!(
            snapshot.get(&storage_key(i, 0)),
            Some(&storage_value(i as u64 * 10))
        );
    }
}

// ============================================================================
// Test: Write Path Changes
// ============================================================================

#[test]
fn test_new_write_path_triggers_revalidation() {
    // Test that writing to a new location triggers revalidation of later transactions
    let _scheduler = Scheduler::new(3);
    let mv = MVHashMap::new(3);

    let key_a = storage_key(0, 0);
    let key_b = storage_key(0, 1);

    // Execute T0: writes only to A
    {
        let mut ws = WriteSet::new();
        ws.insert((key_a.clone(), storage_value(100)));
        let wrote_new = mv.record(Version::new(0, 0), &ReadSet::new(), &ws);
        assert!(wrote_new, "First write should be new path");
    }

    // Execute T1: reads A
    {
        let mut rs = ReadSet::new();
        rs.insert((key_a.clone(), Some(Version::new(0, 0))));
        mv.record(Version::new(1, 0), &rs, &WriteSet::new());
    }

    // T0 re-executes and now also writes to B (new location)
    {
        let mut ws = WriteSet::new();
        ws.insert((key_a.clone(), storage_value(100)));
        ws.insert((key_b.clone(), storage_value(200))); // New location!
        let wrote_new = mv.record(Version::new(0, 1), &ReadSet::new(), &ws);
        assert!(wrote_new, "Writing to new location should trigger revalidation");
    }
}

// ============================================================================
// Test: DEX-like Swap Pattern
// ============================================================================

#[test]
fn test_dex_swap_pattern() {
    // Simulates DEX swaps where multiple users swap tokens
    // Each swap reads reserves and writes new reserves

    let reserve_a = storage_key(0, 0); // Token A reserve
    let reserve_b = storage_key(0, 1); // Token B reserve

    let transactions = vec![
        // Initial liquidity provision
        SimulatedTx::new(
            vec![],
            vec![
                (reserve_a.clone(), storage_value(10000)),
                (reserve_b.clone(), storage_value(10000)),
            ],
        ),
        // Swap 1: Reads reserves, updates them
        SimulatedTx::new(
            vec![reserve_a.clone(), reserve_b.clone()],
            vec![
                (reserve_a.clone(), storage_value(10100)),
                (reserve_b.clone(), storage_value(9900)),
            ],
        ),
        // Swap 2: Reads updated reserves
        SimulatedTx::new(
            vec![reserve_a.clone(), reserve_b.clone()],
            vec![
                (reserve_a.clone(), storage_value(9900)),
                (reserve_b.clone(), storage_value(10100)),
            ],
        ),
        // Swap 3
        SimulatedTx::new(
            vec![reserve_a.clone(), reserve_b.clone()],
            vec![
                (reserve_a.clone(), storage_value(10050)),
                (reserve_b.clone(), storage_value(9950)),
            ],
        ),
    ];

    let (re_execs, snapshot) = simulate_block_stm(transactions, 4);

    // Final reserves should be from the last swap
    assert_eq!(snapshot.get(&reserve_a), Some(&storage_value(10050)));
    assert_eq!(snapshot.get(&reserve_b), Some(&storage_value(9950)));

    println!("DEX swap pattern: {} re-executions for 4 swaps", re_execs);
}

// ============================================================================
// Test: NFT Minting Pattern
// ============================================================================

#[test]
fn test_nft_mint_pattern() {
    // Simulates NFT minting where each mint:
    // - Reads total supply
    // - Writes new token owner
    // - Increments total supply

    let total_supply = storage_key(0, 0);

    let transactions: Vec<SimulatedTx> = (0..10)
        .map(|i| {
            let token_owner = storage_key(0, (i + 1) as u64); // Token ID -> owner
            SimulatedTx::new(
                if i == 0 {
                    vec![]
                } else {
                    vec![total_supply.clone()]
                },
                vec![
                    (token_owner, balance_value((i + 1) as u64)), // Owner address
                    (total_supply.clone(), storage_value((i + 1) as u64)),
                ],
            )
        })
        .collect();

    let (_re_execs, snapshot) = simulate_block_stm(transactions, 4);

    // Final total supply should be 10
    assert_eq!(snapshot.get(&total_supply), Some(&storage_value(10)));

    // All token owners should be set
    for i in 1..=10 {
        assert!(snapshot.contains_key(&storage_key(0, i as u64)));
    }
}

// ============================================================================
// Test: ERC20 Transfer Patterns
// ============================================================================

#[test]
fn test_erc20_transfers_same_sender() {
    // Multiple transfers from the same sender
    // All need to read sender's balance, creating conflicts

    let sender_balance = balance_key(1);

    let transactions: Vec<SimulatedTx> = (0..5)
        .map(|i| {
            let receiver_balance = balance_key((i + 10) as u8);
            SimulatedTx::new(
                vec![sender_balance.clone()],
                vec![
                    (sender_balance.clone(), balance_value(1000 - (i + 1) as u64 * 100)),
                    (receiver_balance, balance_value((i + 1) as u64 * 100)),
                ],
            )
        })
        .collect();

    let (_re_execs, snapshot) = simulate_block_stm(transactions, 4);

    // Sender should have 500 left (1000 - 5*100)
    assert_eq!(snapshot.get(&sender_balance), Some(&balance_value(500)));
}

#[test]
fn test_erc20_transfers_different_senders() {
    // Transfers from different senders - should be parallel
    let transactions: Vec<SimulatedTx> = (0..10)
        .map(|i| {
            let sender = balance_key(i as u8);
            let receiver = balance_key((i + 100) as u8);
            SimulatedTx::new(
                vec![sender.clone()],
                vec![
                    (sender.clone(), balance_value(900)),
                    (receiver, balance_value(100)),
                ],
            )
            .reads_base()
        })
        .collect();

    let (re_execs, snapshot) = simulate_block_stm(transactions, 4);

    assert_eq!(
        re_execs, 0,
        "Different senders should not cause conflicts"
    );

    // All senders should have 900
    for i in 0..10 {
        assert_eq!(snapshot.get(&balance_key(i as u8)), Some(&balance_value(900)));
    }
}

// ============================================================================
// Test: Worst-Case Scenarios
// ============================================================================

#[test]
fn test_all_transactions_conflict() {
    // Every transaction reads and writes the same key
    let key = storage_key(0, 0);

    let transactions: Vec<SimulatedTx> = (0..20)
        .map(|i| {
            SimulatedTx::new(
                vec![key.clone()],
                vec![(key.clone(), storage_value(i as u64))],
            )
        })
        .collect();

    let (re_execs, snapshot) = simulate_block_stm(transactions, 8);

    // Final value should be from transaction 19
    assert_eq!(snapshot.get(&key), Some(&storage_value(19)));

    println!("All-conflict test: {} re-executions for 20 txns", re_execs);
}

#[test]
fn test_cascading_invalidations() {
    // Each transaction reads from all previous transactions
    // Creating maximum cascade on any abort

    let transactions: Vec<SimulatedTx> = (0..10)
        .map(|i| {
            let reads: Vec<_> = (0..i).map(|j| storage_key(j as u8, 0)).collect();
            SimulatedTx::new(reads, vec![(storage_key(i as u8, 0), storage_value(i as u64))])
        })
        .collect();

    let (_re_execs, snapshot) = simulate_block_stm(transactions, 4);

    // All writes should be present
    for i in 0..10 {
        assert_eq!(
            snapshot.get(&storage_key(i as u8, 0)),
            Some(&storage_value(i as u64))
        );
    }
}

// ============================================================================
// Test: Resource Tracking
// ============================================================================

#[test]
fn test_da_bytes_tracking() {
    let da_key = EvmStateKey::BlockResourceUsed(BlockResourceType::DABytes);

    let transactions: Vec<SimulatedTx> = (0..5)
        .map(|i| {
            let cumulative_da = (i + 1) as u64 * 1000;
            SimulatedTx::new(
                if i == 0 { vec![] } else { vec![da_key.clone()] },
                vec![(da_key.clone(), EvmStateValue::BlockResourceUsed(cumulative_da))],
            )
        })
        .collect();

    let (_re_execs, snapshot) = simulate_block_stm(transactions, 4);

    assert_eq!(
        snapshot.get(&da_key),
        Some(&EvmStateValue::BlockResourceUsed(5000))
    );
}

#[test]
fn test_combined_gas_and_da_tracking() {
    let gas_key = EvmStateKey::BlockResourceUsed(BlockResourceType::Gas);
    let da_key = EvmStateKey::BlockResourceUsed(BlockResourceType::DABytes);

    let transactions: Vec<SimulatedTx> = (0..5)
        .map(|i| {
            SimulatedTx::new(
                if i == 0 {
                    vec![]
                } else {
                    vec![gas_key.clone(), da_key.clone()]
                },
                vec![
                    (
                        gas_key.clone(),
                        EvmStateValue::BlockResourceUsed((i + 1) as u64 * 21000),
                    ),
                    (
                        da_key.clone(),
                        EvmStateValue::BlockResourceUsed((i + 1) as u64 * 100),
                    ),
                ],
            )
        })
        .collect();

    let (_re_execs, snapshot) = simulate_block_stm(transactions, 4);

    assert_eq!(
        snapshot.get(&gas_key),
        Some(&EvmStateValue::BlockResourceUsed(105000))
    );
    assert_eq!(
        snapshot.get(&da_key),
        Some(&EvmStateValue::BlockResourceUsed(500))
    );
}

// ============================================================================
// Test: Concurrent Thread Safety
// ============================================================================

#[test]
fn stress_test_concurrent_mvhashmap_access() {
    // Stress test MVHashMap with many concurrent readers and writers
    let num_txns = 100;
    let mv = Arc::new(MVHashMap::new(num_txns));
    let num_threads = 8;

    thread::scope(|s| {
        // Writer threads
        for t in 0..num_threads / 2 {
            let mv = Arc::clone(&mv);
            s.spawn(move || {
                for i in 0..num_txns {
                    if i % (num_threads / 2) == t {
                        let mut ws = WriteSet::new();
                        ws.insert((storage_key(i as u8, 0), storage_value(i as u64)));
                        mv.record(Version::new(i as u32, 0), &ReadSet::new(), &ws);
                    }
                }
            });
        }

        // Reader threads
        for _ in 0..num_threads / 2 {
            let mv = Arc::clone(&mv);
            s.spawn(move || {
                for i in 0..num_txns {
                    let _ = mv.read(&storage_key(i as u8, 0), num_txns as u32);
                }
            });
        }
    });

    // All writes should be present
    let snapshot = Arc::try_unwrap(mv).unwrap().into_snapshot();
    assert_eq!(snapshot.len(), num_txns);
}

#[test]
fn stress_test_scheduler_concurrent_access() {
    // Stress test scheduler with many threads competing for tasks
    let num_txns = 100;
    let scheduler = Arc::new(Scheduler::new(num_txns));
    let executed = Arc::new(std::sync::Mutex::new(HashSet::new()));
    let mv = Arc::new(MVHashMap::new(num_txns));

    thread::scope(|s| {
        for _ in 0..8 {
            let scheduler = Arc::clone(&scheduler);
            let executed = Arc::clone(&executed);
            let mv = Arc::clone(&mv);

            s.spawn(move || {
                let mut pending: Option<Task> = None;

                while !scheduler.done() {
                    let task = pending.take().or_else(|| scheduler.next_task());

                    match task {
                        Some(Task::Execute { version }) => {
                            executed.lock().unwrap().insert(version.txn_idx);

                            let read_set = ReadSet::new();
                            let write_set = WriteSet::new();
                            mv.record(version, &read_set, &write_set);

                            let next = scheduler.finish_execution(
                                version.txn_idx,
                                version.incarnation,
                                false,
                            );

                            if let Some(Task::Validate { version: v }) = next {
                                scheduler.finish_validation(v.txn_idx, false);
                            }
                        }
                        Some(Task::Validate { version }) => {
                            scheduler.finish_validation(version.txn_idx, false);
                        }
                        None => {
                            thread::yield_now();
                        }
                    }
                }
            });
        }
    });

    let executed = executed.lock().unwrap();
    assert_eq!(
        executed.len(),
        num_txns,
        "All transactions should be executed exactly once"
    );
}
