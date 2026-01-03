# Block-STM: Parallel Transaction Execution for EVM

This document provides a comprehensive explanation of the Block-STM parallel execution engine implemented in this crate. Block-STM is an algorithm originally developed for the Aptos/Diem blockchain and adapted here for EVM transaction execution on Optimism.

## Table of Contents

1. [Overview](#overview)
2. [The Problem: Sequential EVM Execution](#the-problem-sequential-evm-execution)
3. [Block-STM Solution](#block-stm-solution)
4. [Core Components](#core-components)
5. [Algorithm Flow](#algorithm-flow)
6. [Conflict Detection and Resolution](#conflict-detection-and-resolution)
7. [Special Handling](#special-handling)
8. [Performance Characteristics](#performance-characteristics)
9. [Example Scenarios](#example-scenarios)

---

## Overview

Block-STM (Software Transactional Memory) enables parallel execution of EVM transactions while maintaining the same deterministic output as sequential execution. The key insight is that most transactions in a block don't conflict with each other - they touch different accounts, storage slots, or contracts. Block-STM exploits this parallelism while correctly handling the cases where conflicts do occur.

**Key Properties:**
- **Deterministic**: Produces identical results to sequential execution
- **Optimistic**: Executes transactions speculatively in parallel
- **Lock-free reads**: No blocking on read operations
- **Push-based invalidation**: Conflicts trigger re-execution automatically

## The Problem: Sequential EVM Execution

Traditional EVM execution processes transactions sequentially:

```
Block with transactions: [T0, T1, T2, T3, T4]

Sequential execution:
  T0 executes → state S0
  T1 executes on S0 → state S1
  T2 executes on S1 → state S2
  T3 executes on S2 → state S3
  T4 executes on S3 → state S4 (final)
```

This is slow because each transaction must wait for all previous transactions to complete, even if they touch completely different state.

## Block-STM Solution

Block-STM introduces a multi-version data structure (MVHashMap) that allows all transactions to execute in parallel against versioned state:

```
Parallel execution:
  All transactions start executing simultaneously
  Each sees writes from lower-indexed transactions via MVHashMap

  T0 reads base state → writes version (0, 0)
  T1 reads base state + T0's writes → writes version (1, 0)
  T2 reads base state + T0,T1's writes → writes version (2, 0)
  ...

  Validation phase checks if reads are still valid
  Re-execute if a dependency changed
```

### The Versioned World View

Each transaction sees a consistent snapshot:
- **Transaction Ti reads from**: highest version written by Tj where j < i
- **If no prior transaction wrote**: read from base state
- **Write visibility**: Transaction Ti's writes are visible to all Tj where j > i

## Core Components

### 1. MVHashMap (`mv_hashmap.rs`)

The Multi-Version Hash Map is the central data structure:

```rust
struct MVHashMap {
    // Map from state key → (transaction index → versioned value)
    data: DashMap<EvmStateKey, RwLock<HashMap<TxnIndex, MVHashMapValue>>>,

    // Track what each transaction wrote (for abort handling)
    last_written_locations: Vec<RwLock<HashSet<EvmStateKey>>>,

    // Track what each transaction read (for validation)
    last_read_set: Vec<RwLock<ReadSet>>,
}
```

**Key Operations:**
- `read(key, reader_idx)`: Returns highest version < reader_idx, or NotFound
- `apply_write_set(txn_idx, incarnation, writes)`: Record transaction's writes
- `validate_read_set(txn_idx)`: Check if all reads are still at expected versions
- `convert_writes_to_estimates(txn_idx)`: Mark aborted transaction's writes as invalid

### 2. Scheduler (`scheduler.rs`)

Coordinates parallel execution and validation:

```rust
struct Scheduler {
    execution_idx: AtomicUsize,    // Next transaction to execute
    validation_idx: AtomicUsize,   // Next transaction to validate
    txn_status: Vec<Mutex<(Incarnation, ExecutionStatus)>>,
    txn_dependency: Vec<Mutex<HashSet<TxnIndex>>>,
}
```

**Task Types:**
- `Execute { version }`: Execute transaction at (txn_idx, incarnation)
- `Validate { version }`: Validate transaction's read set

**Status Transitions:**
```
ReadyToExecute → Executing → Executed
       ↑                        ↓
       └──────── Aborting ←─────┘
                 (on conflict)
```

### 3. Executor (`executor.rs`)

Orchestrates the parallel execution:

```rust
impl Executor {
    fn execute_transactions_parallel(...) {
        thread::scope(|s| {
            for worker_id in 0..num_threads {
                s.spawn(|| {
                    while !scheduler.done() {
                        match scheduler.next_task() {
                            Task::Execute { version } => {
                                // Execute transaction
                                // Record reads/writes to MVHashMap
                                scheduler.finish_execution(...)
                            }
                            Task::Validate { version } => {
                                // Check if read set is still valid
                                // Abort and re-execute if not
                                scheduler.finish_validation(...)
                            }
                        }
                    }
                });
            }
        });
    }
}
```

### 4. VersionedDatabase (`db_adapter.rs`)

Implements revm's `Database` trait with versioned reads:

```rust
impl Database for VersionedDatabase<'_, DB> {
    fn basic(&mut self, address: Address) -> Result<Option<AccountInfo>> {
        // Read balance, nonce, code_hash from MVHashMap or base state
        // Track reads for later validation
    }

    fn storage(&mut self, address: Address, slot: U256) -> Result<U256> {
        // Read storage slot from MVHashMap or base state
    }
}
```

### 5. State Keys and Values (`types.rs`)

Abstracts EVM state for the versioned storage:

```rust
enum EvmStateKey {
    Balance(Address),
    Nonce(Address),
    CodeHash(Address),
    Code(Address),
    Storage(Address, U256),
    BlockResourceUsed(BlockResourceType),  // Gas, DA bytes
}

enum EvmStateValue {
    Balance(U256),
    BalanceIncrement(U256),  // For commutative fee payments
    Nonce(u64),
    CodeHash(B256),
    Code(Bytes),
    Storage(U256),
    BlockResourceUsed(u64),
}
```

## Algorithm Flow

### Phase 1: Speculative Execution

```
1. Worker threads grab Execute tasks from scheduler
2. Each transaction executes against VersionedDatabase:
   - Reads go through MVHashMap (sees prior transactions' writes)
   - Falls back to base state if no prior write
   - Records read set (key → version read)
3. After execution:
   - Record write set to MVHashMap
   - Signal scheduler that execution is complete
```

### Phase 2: Validation

```
1. Worker threads grab Validate tasks from scheduler
2. For each read in the transaction's read set:
   - Check current version in MVHashMap
   - If version changed → conflict detected
3. If conflict:
   - Abort transaction
   - Convert writes to "estimate" markers
   - Trigger re-execution with incremented incarnation
4. If valid:
   - Transaction is ready to commit
```

### Phase 3: Commit

```
1. Find safe commit point:
   - Longest prefix of transactions that are:
     - Status == Executed
     - Passed validation (validation_idx > txn_idx)
2. Commit results in order (T0, T1, T2, ...)
3. Apply state changes to database
```

## Conflict Detection and Resolution

### What Causes a Conflict?

A conflict occurs when transaction Ti read a value that was later modified by a re-execution of Tj (where j < i):

```
Initial state: slot[0] = 100

Parallel execution:
  T0: reads slot[0] → 100, writes slot[0] = 200
  T1: reads slot[0] → 100 (from base state, T0 not visible yet)

After T0 completes:
  T1 validation fails: expected version=None, actual version=(0,0)
  T1 must re-execute

T1 re-executes:
  T1: reads slot[0] → 200 (sees T0's write now)
```

### Resolution: Re-execution

When a transaction is aborted:

1. **Mark writes as estimates**: Other transactions see `Aborted` when reading
2. **Increment incarnation**: Version becomes (txn_idx, incarnation + 1)
3. **Re-execute**: Transaction runs again with updated inputs
4. **Re-validate**: New read set is checked for conflicts

### Estimate Markers

When transaction Ti aborts, its writes become "estimates":

```rust
enum MVHashMapValue {
    Write(Incarnation, EvmStateValue),  // Valid write
    Estimate,                            // Aborted, value uncertain
}
```

If Tj (j > i) reads an estimate, it receives `ReadResult::Aborted` and must wait or register a dependency.

## Special Handling

### Balance Increments (Fee Payments)

Fee payments to coinbase are handled specially to avoid false conflicts:

```rust
// Instead of: coinbase_balance = coinbase_balance + fee
// Block-STM uses: BalanceIncrement(fee)
```

**Why?** Multiple transactions paying fees to the same coinbase don't truly conflict - the increments are commutative. Using `BalanceIncrement` records the intent without creating read-write conflicts.

```
T0: BalanceIncrement(coinbase, 100)
T1: BalanceIncrement(coinbase, 150)
T2: BalanceIncrement(coinbase, 75)

Final: coinbase_balance += 100 + 150 + 75 = 325
Order doesn't matter!
```

### Block Resource Tracking

Cumulative resources (gas used, DA bytes) are tracked via `BlockResourceUsed`:

```rust
EvmStateKey::BlockResourceUsed(BlockResourceType::Gas)
EvmStateKey::BlockResourceUsed(BlockResourceType::DABytes)
```

Each transaction reads the previous cumulative value and writes the new cumulative:
- T0: writes gas_used = 21000
- T1: reads gas_used = 21000, writes gas_used = 45000
- T2: reads gas_used = 45000, writes gas_used = 67000

This creates intentional dependencies for gas limit enforcement.

### Code Cache

Newly deployed contracts must be visible to subsequent transactions:

```rust
shared_code_cache: Arc<DashMap<B256, Bytecode>>
```

When a contract is deployed, its bytecode is stored in the shared cache so later transactions can call it via `code_by_hash`.

## Performance Characteristics

### Best Case: No Conflicts

When transactions are independent (different accounts/slots):
- **Speedup**: Near-linear with thread count
- **Re-executions**: None
- **Validation**: All pass on first try

### Worst Case: Serial Dependencies

When every transaction depends on the previous:
- **Speedup**: None (effectively sequential)
- **Re-executions**: O(n) re-executions
- **Example**: All transactions transfer from same account

### Typical Case: Mixed Workload

Real-world blocks have some conflicts but not total:
- **Speedup**: 2-4x with 4 threads (depends on workload)
- **Re-executions**: Some transactions re-execute
- **Hot spots**: Popular contracts may cause more conflicts

### Conflict Key Optimization

When re-executing, we track which keys actually caused the conflict:

```rust
let conflicting_keys: HashSet<EvmStateKey> = /* keys where version changed */;
```

If only `BlockResourceUsed` changed (not account state), we can potentially reuse the previous execution result.

## Example Scenarios

### Scenario 1: Independent Transactions

```
T0: Alice → Bob (100 ETH)
T1: Charlie → Dave (50 ETH)
T2: Eve → Frank (75 ETH)

No conflicts! All execute in parallel:
- T0 touches: Alice.balance, Bob.balance
- T1 touches: Charlie.balance, Dave.balance
- T2 touches: Eve.balance, Frank.balance

All validations pass. Final order maintained.
```

### Scenario 2: Read-Write Conflict

```
T0: writes storage[0] = 100
T1: reads storage[0], writes storage[1] = storage[0] * 2

Parallel execution:
  T0: writes storage[0] = 100
  T1: reads storage[0] → NotFound (T0 not visible yet)
      writes storage[1] = 0 (wrong!)

T0 completes → T1 validation:
  T1 expected storage[0] from base state
  But now storage[0] has version (0, 0)
  CONFLICT! T1 aborts.

T1 re-executes (incarnation 1):
  T1: reads storage[0] = 100 (sees T0's write)
      writes storage[1] = 200 (correct!)

T1 validation passes.
Final: storage[0] = 100, storage[1] = 200
```

### Scenario 3: Chain of Dependencies

```
T0: account.balance = 1000
T1: account.balance = account.balance + 100
T2: account.balance = account.balance + 200

This creates serial dependency - each must wait for previous.
Block-STM handles this correctly through re-execution:

Round 1:
  T0: balance = 1000
  T1: reads balance → NotFound, writes balance = 100 (wrong)
  T2: reads balance → 100 (from T1), writes balance = 300 (wrong)

Validation:
  T1 conflict (T0 wrote balance)
  T2 conflict (T1 will re-execute)

Round 2:
  T1: reads balance = 1000, writes balance = 1100
  T2: reads balance → 100 (still T1's old write), conflict again

Round 3:
  T2: reads balance = 1100, writes balance = 1300

Final: balance = 1300 (correct!)
```

### Scenario 4: Contract Deployment and Call

```
T0: Deploy contract C at address 0x123
T1: Call C.foo()

Block-STM handles this via shared code cache:

T0 execution:
  - Deploys contract
  - Stores bytecode in shared_code_cache[C.code_hash]

T1 execution:
  - Reads C.code_hash from MVHashMap
  - Looks up bytecode in shared_code_cache
  - Executes C.foo()

If T1 executes before T0 completes:
  - C.code_hash not found → base state lookup
  - Contract doesn't exist → transaction reverts (speculatively)
  - When T0 completes, T1 validation fails (code_hash changed)
  - T1 re-executes, now sees the deployed contract
```

---

## References

- [Block-STM Paper](https://arxiv.org/abs/2203.06871) - Original algorithm description
- [Aptos Block-STM Implementation](https://github.com/aptos-labs/aptos-core/tree/main/aptos-move/block-executor)
- [Move VM Parallel Execution](https://medium.com/aptoslabs/block-stm-how-we-execute-over-160k-transactions-per-second-on-the-aptos-blockchain-3b003657e4ba)
