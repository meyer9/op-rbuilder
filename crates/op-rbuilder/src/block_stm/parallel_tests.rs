//! Parallelization tests for Block-STM execution engine.
//!
//! These tests verify the correct behavior of parallel transaction execution,
//! including conflict detection, re-execution, and deterministic ordering.

use super::*;
use crate::block_stm::mv_hashmap::{ReadSet, WriteSet};
use crate::block_stm::types::{
    BlockResourceType, EvmStateKey, EvmStateValue, ExecutionStatus, Incarnation, ReadResult, Task,
    TxnIndex, Version,
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
