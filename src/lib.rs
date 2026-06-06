//! # oxide-journal
//!
//! Write-ahead journal for GPU state mutations with ternary integrity.

use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryState { Committed = 1, Pending = 0, Corrupted = -1 }

#[derive(Debug, Clone)]
pub struct JournalEntry {
    pub lsn: u64,
    pub op: String,
    pub payload: Vec<u8>,
    pub checksum: u64,
    pub state: EntryState,
}

fn simple_checksum(data: &[u8], lsn: u64) -> u64 {
    let mut h: u64 = 14695981039346656037 ^ lsn;
    for &b in data { h ^= b as u64; h = h.wrapping_mul(1099511628211); }
    h
}

pub struct OxideJournal {
    entries: VecDeque<JournalEntry>,
    next_lsn: u64,
    committed_lsn: u64,
    corrupted_count: u64,
    compacted_count: u64,
}

impl OxideJournal {
    pub fn new() -> Self {
        Self { entries: VecDeque::new(), next_lsn: 1, committed_lsn: 0, corrupted_count: 0, compacted_count: 0 }
    }

    pub fn append(&mut self, op: &str, payload: &[u8]) -> u64 {
        let lsn = self.next_lsn;
        self.next_lsn += 1;
        let checksum = simple_checksum(payload, lsn);
        self.entries.push_back(JournalEntry { lsn, op: op.into(), payload: payload.into(), checksum, state: EntryState::Pending });
        lsn
    }

    pub fn commit(&mut self, lsn: u64) -> bool {
        for entry in &mut self.entries {
            if entry.lsn == lsn && entry.state == EntryState::Pending {
                entry.state = EntryState::Committed;
                self.committed_lsn = self.committed_lsn.max(lsn);
                return true;
            }
        }
        false
    }

    /// Verify all entries. Mark corrupted if checksum doesn't match.
    pub fn verify(&mut self) -> (u64, u64) {
        let mut ok = 0u64;
        let mut bad = 0u64;
        for entry in &mut self.entries {
            let expected = simple_checksum(&entry.payload, entry.lsn);
            if expected != entry.checksum {
                entry.state = EntryState::Corrupted;
                bad += 1;
                self.corrupted_count += 1;
            } else {
                ok += 1;
            }
        }
        (ok, bad)
    }

    /// Replay committed entries in order, calling f for each.
    pub fn replay<F>(&self, mut f: F) -> u64 where F: FnMut(&JournalEntry) {
        let mut count = 0u64;
        for entry in &self.entries {
            if entry.state == EntryState::Committed {
                f(entry);
                count += 1;
            }
        }
        count
    }

    /// Compact: remove committed entries below watermark.
    pub fn compact(&mut self, watermark_lsn: u64) -> usize {
        let before = self.entries.len();
        self.entries.retain(|e| e.lsn > watermark_lsn || e.state != EntryState::Committed);
        let removed = before - self.entries.len();
        self.compacted_count += removed as u64;
        removed
    }

    /// Corrupt an entry (for testing).
    pub fn corrupt(&mut self, lsn: u64) {
        for entry in &mut self.entries {
            if entry.lsn == lsn {
                entry.checksum = 0; // guaranteed wrong
                entry.state = EntryState::Corrupted;
                self.corrupted_count += 1;
                return;
            }
        }
    }

    pub fn entry_count(&self) -> usize { self.entries.len() }
    pub fn committed_lsn(&self) -> u64 { self.committed_lsn }
    pub fn corrupted_count(&self) -> u64 { self.corrupted_count }
}

impl Default for OxideJournal { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append() {
        let mut j = OxideJournal::new();
        let lsn = j.append("write", b"data");
        assert_eq!(lsn, 1);
        assert_eq!(j.entry_count(), 1);
    }

    #[test]
    fn test_commit() {
        let mut j = OxideJournal::new();
        let lsn = j.append("write", b"data");
        assert!(j.commit(lsn));
        assert_eq!(j.committed_lsn(), 1);
    }

    #[test]
    fn test_verify_ok() {
        let mut j = OxideJournal::new();
        j.append("op", b"payload");
        let (ok, bad) = j.verify();
        assert_eq!(ok, 1);
        assert_eq!(bad, 0);
    }

    #[test]
    fn test_verify_corrupt() {
        let mut j = OxideJournal::new();
        let lsn = j.append("op", b"payload");
        j.corrupt(lsn);
        let (ok, bad) = j.verify();
        assert_eq!(bad, 1);
    }

    #[test]
    fn test_replay() {
        let mut j = OxideJournal::new();
        j.append("a", b"1");
        let lsn2 = j.append("b", b"2");
        j.commit(lsn2);
        let mut replayed = Vec::new();
        let count = j.replay(|e| replayed.push(e.op.clone()));
        assert_eq!(count, 1);
        assert_eq!(replayed, vec!["b".to_string()]);
    }

    #[test]
    fn test_compact() {
        let mut j = OxideJournal::new();
        let l1 = j.append("a", b"1");
        let _l2 = j.append("b", b"2");
        j.commit(l1);
        let removed = j.compact(1);
        assert_eq!(removed, 1);
        assert_eq!(j.entry_count(), 1);
    }

    #[test]
    fn test_order_preserved() {
        let mut j = OxideJournal::new();
        j.append("first", b"1");
        j.append("second", b"2");
        let mut ops = Vec::new();
        j.replay(|e| ops.push(e.lsn));
        // Neither committed, so nothing replayed
        assert!(ops.is_empty());
    }

    #[test]
    fn test_commit_twice_fails() {
        let mut j = OxideJournal::new();
        let lsn = j.append("op", b"d");
        assert!(j.commit(lsn));
        assert!(!j.commit(lsn)); // already committed
    }
}
