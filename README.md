# oxide-journal

Write-ahead journal for GPU state mutations with ternary integrity.

## Why This Exists

GPU kernels mutate state: memory buffers, tensor shapes, execution contexts. When a kernel crashes mid-execution or a node dies, you need to know exactly what state was durable and what was in-flight. Database people solved this decades ago with write-ahead logs. This is the same idea, adapted for GPU compute.

The ternary integrity model is the key innovation. Every journal entry is one of three states: **Committed** (+1, durable and verified), **Pending** (0, written but not yet confirmed), or **Corrupted** (-1, checksum mismatch). After a crash, you replay committed entries, discard pending ones, and flag corrupted ones. No ambiguity, no heuristic recovery.

## Architecture

```
┌────────────────────────────────────────────┐
│              OxideJournal                   │
│                                            │
│  next_lsn: 4                               │
│  committed_lsn: 2                          │
│  corrupted_count: 0                        │
│                                            │
│  ┌──────────────────────────────────────┐  │
│  │ LSN 1: op="alloc"   [Committed]  ✓  │  │
│  │ LSN 2: op="write"   [Committed]  ✓  │  │
│  │ LSN 3: op="launch"  [Pending]       │  │
│  │ LSN 4: op="copy"    [Pending]       │  │
│  └──────────────────────────────────────┘  │
│                                            │
│  append(op, payload) → lsn                 │
│  commit(lsn) → bool                        │
│  verify() → (ok, bad)                      │
│  replay(|entry| { ... }) → count           │
│  compact(watermark) → removed              │
└────────────────────────────────────────────┘
```

**Key types:**

- `EntryState` — `Committed(+1)`, `Pending(0)`, `Corrupted(-1)`
- `JournalEntry` — LSN, operation name, payload bytes, FNV-1a checksum, state
- `OxideJournal` — the journal itself, backed by a `VecDeque`

**Checksum:** FNV-1a variant seeded with the LSN, so each entry's checksum is position-dependent. You can't swap entries without detection.

## Usage

```rust
use oxide_journal::OxideJournal;

let mut journal = OxideJournal::new();

// Append operations (all start as Pending)
let lsn1 = journal.append("alloc_buffer", b"gpu_mem_0: 2GiB");
let lsn2 = journal.append("write_tensor", b"tensor_A: shape=[3,224,224]");
let lsn3 = journal.append("launch_kernel", b"conv2d: filters=64");

// Commit completed operations
journal.commit(lsn1); // alloc done
journal.commit(lsn2); // write done
// lsn3 stays Pending — kernel still running

// Verify integrity after crash
let (ok, bad) = journal.verify();
assert_eq!(ok, 3);
assert_eq!(bad, 0);

// Replay only committed entries for recovery
let mut recovered_ops = Vec::new();
journal.replay(|entry| {
    recovered_ops.push((entry.op.clone(), entry.payload.clone()));
});
assert_eq!(recovered_ops.len(), 2); // only committed

// Compact: remove committed entries below watermark
let removed = journal.compact(2); // remove committed entries with LSN ≤ 2
```

## API Reference

### `EntryState`

```rust
pub enum EntryState {
    Committed = 1,   // Durable and verified
    Pending = 0,     // Written but unconfirmed
    Corrupted = -1,  // Checksum mismatch
}
```

### `JournalEntry`

```rust
pub struct JournalEntry {
    pub lsn: u64,           // Log sequence number (monotonic)
    pub op: String,         // Operation name
    pub payload: Vec<u8>,   // Operation data
    pub checksum: u64,      // FNV-1a hash of payload + LSN
    pub state: EntryState,
}
```

### `OxideJournal`

- `new() -> Self`
- `append(op: &str, payload: &[u8]) -> u64` — write a pending entry, returns LSN
- `commit(lsn: u64) -> bool` — mark entry as committed, fails if already committed or not found
- `verify() -> (u64, u64)` — checksum all entries, mark corrupted ones, returns (ok_count, bad_count)
- `replay(f: FnMut(&JournalEntry)) -> u64` — iterate committed entries in LSN order, returns count
- `compact(watermark_lsn: u64) -> usize` — remove committed entries with LSN ≤ watermark
- `corrupt(lsn: u64)` — inject corruption for testing (zeroes checksum)
- `entry_count() -> usize` / `committed_lsn() -> u64` / `corrupted_count() -> u64`

## The Deeper Idea

The journal is the **durability layer** in the oxide stack's state management architecture. It sits between the pipeline execution layer (oxide-pipeline) and the checkpoint layer (oxide-checkpoint). The journal captures every mutation; checkpoints are periodic snapshots. Together, they give you both point-in-time recovery (checkpoints) and complete audit trails (journal replay).

The ternary entry state (Committed/Pending/Corrupted) mirrors the {-1, 0, +1} pattern used throughout the ecosystem. A corrupted journal entry triggers the same kind of degradation signal as a failed GPU node or an overloaded capacity signal — the system knows something is wrong and can take corrective action without human interpretation.

## Related Crates

- **oxide-checkpoint** — periodic state snapshots that complement journal replay
- **oxide-pipeline** — execution pipeline that generates journal entries
- **oxide-health-monitor** — GPU health signals that trigger journal verification
- **oxide-compile-cache** — cached compilation results stored as journal entries
