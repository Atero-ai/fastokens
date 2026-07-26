use std::{
    cell::RefCell,
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    fmt,
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder};
use serde::Deserialize;
use serde_json::Value;

use super::Result;
use crate::pre_tokenizers::BYTE_TO_CHAR;

type TokenId = u32;
type ParsedMergeMap = HashMap<(u32, u32), (u32, u32)>;
type Vocab = HashMap<String, u32>;

const INVALID_TOKEN: u32 = u32::MAX;

/// Pretokens with at most this many initial (per-byte) symbols use the
/// stack-resident linear-scan merge instead of the heap. Sized so the stack
/// arrays fit in registers/L1 and `u8` linked-list indices stay in range.
const SMALL_MERGE_MAX: usize = 32;

/// Open-addressing hash table for merge lookups.
#[derive(Clone, PartialEq)]
struct MergeMap {
    mask: usize,
    keys: Vec<u64>,
    vals: Vec<u32>,
}

const EMPTY_KEY: u64 = u64::MAX;

impl MergeMap {
    fn new() -> Self {
        Self {
            mask: 0,
            keys: Vec::new(),
            vals: Vec::new(),
        }
    }

    fn from_parsed(parsed: &ParsedMergeMap) -> Self {
        if parsed.is_empty() {
            return Self::new();
        }
        // ~50% load factor.
        let capacity = (parsed.len() * 2).next_power_of_two();
        let mask = capacity - 1;
        let mut keys = vec![EMPTY_KEY; capacity];
        let mut vals = vec![0u32; capacity];

        for (&(t1, t2), &(_rank, merged_id)) in parsed {
            let key = pack_pair(t1, t2);
            let mut idx = fx_hash(key) as usize & mask;
            loop {
                if keys[idx] == EMPTY_KEY {
                    keys[idx] = key;
                    vals[idx] = merged_id;
                    break;
                }
                idx = (idx + 1) & mask;
            }
        }

        Self { mask, keys, vals }
    }

    /// Look up the merged token ID for a pair.
    #[inline(always)]
    fn get(&self, t1: u32, t2: u32) -> Option<u32> {
        if self.keys.is_empty() {
            return None;
        }
        let key = pack_pair(t1, t2);
        let mut idx = fx_hash(key) as usize & self.mask;
        loop {
            let k = unsafe { *self.keys.get_unchecked(idx) };
            if k == key {
                return Some(unsafe { *self.vals.get_unchecked(idx) });
            }
            if k == EMPTY_KEY {
                return None;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    fn len(&self) -> usize {
        self.keys.iter().filter(|&&k| k != EMPTY_KEY).count()
    }
}

#[inline(always)]
fn pack_pair(t1: u32, t2: u32) -> u64 {
    (t1 as u64) << 32 | t2 as u64
}

#[inline(always)]
fn fx_hash(key: u64) -> u64 {
    key.wrapping_mul(0x517cc1b727220a95)
}

/// FxHash-based [`BuildHasher`] for the token cache.
#[derive(Clone, Default)]
struct FxBuildHasher;

impl std::hash::BuildHasher for FxBuildHasher {
    type Hasher = FxStrHasher;
    fn build_hasher(&self) -> FxStrHasher {
        FxStrHasher(0)
    }
}

struct FxStrHasher(u64);

impl std::hash::Hasher for FxStrHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut state = self.0;
        let mut i = 0;
        while i + 8 <= bytes.len() {
            let word = u64::from_ne_bytes(bytes[i..i + 8].try_into().unwrap());
            state = state.wrapping_add(word).wrapping_mul(0x517cc1b727220a95);
            i += 8;
        }
        while i < bytes.len() {
            state = state
                .wrapping_add(bytes[i] as u64)
                .wrapping_mul(0x517cc1b727220a95);
            i += 1;
        }
        self.0 = state;
    }
}

type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;

const FLAT_CACHE_BITS: usize = 16;
const EMPTY_SLOT: u64 = 0;

#[derive(Clone, Copy)]
#[repr(C)]
struct CacheSlot {
    hash: u64,
    offset: u32,
    len: u16,
    key_len: u16,
    key_offset: u32,
}

struct FlatCache {
    bpe_id: usize,
    mask: usize,
    max_load: usize,
    slots: Vec<CacheSlot>,
    pool: Vec<u32>,
    key_pool: Vec<u8>,
    count: usize,
}

impl FlatCache {
    fn new() -> Self {
        Self::with_bits(FLAT_CACHE_BITS)
    }

    /// A flat cache with `1 << bits` slots. The thread-local L1 uses
    /// [`FLAT_CACHE_BITS`]; the shared-cache shards use fewer bits each.
    fn with_bits(bits: usize) -> Self {
        let size = 1usize << bits;
        Self {
            bpe_id: 0,
            mask: size - 1,
            max_load: size * 3 / 4,
            slots: vec![
                CacheSlot {
                    hash: EMPTY_SLOT,
                    offset: 0,
                    len: 0,
                    key_len: 0,
                    key_offset: 0,
                };
                size
            ],
            pool: Vec::new(),
            key_pool: Vec::new(),
            count: 0,
        }
    }

    fn clear(&mut self) {
        for slot in &mut self.slots {
            slot.hash = EMPTY_SLOT;
        }
        self.pool.clear();
        self.key_pool.clear();
        self.count = 0;
    }

    #[inline(always)]
    fn hash_str(s: &str) -> u64 {
        let bytes = s.as_bytes();
        let mut h: u64 = bytes.len() as u64;
        let mut i = 0;
        while i + 8 <= bytes.len() {
            let word = u64::from_ne_bytes(bytes[i..i + 8].try_into().unwrap());
            h = h.wrapping_add(word).wrapping_mul(0x517cc1b727220a95);
            i += 8;
        }
        while i < bytes.len() {
            h = h
                .wrapping_add(bytes[i] as u64)
                .wrapping_mul(0x517cc1b727220a95);
            i += 1;
        }
        if h == EMPTY_SLOT {
            h = 1;
        }
        h
    }

    #[inline(always)]
    fn get(&self, key: &str, out: &mut Vec<u32>) -> bool {
        let hash = Self::hash_str(key);
        let key_bytes = key.as_bytes();
        let mut idx = hash as usize & self.mask;
        loop {
            let slot = unsafe { self.slots.get_unchecked(idx) };
            if slot.hash == hash {
                let ks = slot.key_offset as usize;
                let ke = ks + slot.key_len as usize;
                if unsafe { self.key_pool.get_unchecked(ks..ke) } == key_bytes {
                    let start = slot.offset as usize;
                    let end = start + slot.len as usize;
                    out.extend_from_slice(unsafe { self.pool.get_unchecked(start..end) });
                    return true;
                }
            }
            if slot.hash == EMPTY_SLOT {
                return false;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    #[inline(always)]
    fn insert(&mut self, key: &str, ids: &[u32]) {
        if self.count >= self.max_load {
            self.clear();
        }
        let hash = Self::hash_str(key);
        let key_bytes = key.as_bytes();
        let mut idx = hash as usize & self.mask;
        loop {
            let slot = unsafe { self.slots.get_unchecked(idx) };
            let h = slot.hash;
            if h == EMPTY_SLOT {
                let Ok(len) = u16::try_from(ids.len()) else {
                    return;
                };
                let Ok(key_len) = u16::try_from(key_bytes.len()) else {
                    return;
                };
                self.count += 1;
                let offset = self.pool.len() as u32;
                self.pool.extend_from_slice(ids);
                let key_offset = self.key_pool.len() as u32;
                self.key_pool.extend_from_slice(key_bytes);
                let slot = unsafe { self.slots.get_unchecked_mut(idx) };
                slot.hash = hash;
                slot.offset = offset;
                slot.len = len;
                slot.key_offset = key_offset;
                slot.key_len = key_len;
                return;
            }
            if h == hash {
                let ks = slot.key_offset as usize;
                let ke = ks + slot.key_len as usize;
                if unsafe { self.key_pool.get_unchecked(ks..ke) } == key_bytes {
                    let Ok(len) = u16::try_from(ids.len()) else {
                        return;
                    };
                    let offset = self.pool.len() as u32;
                    self.pool.extend_from_slice(ids);
                    let slot = unsafe { self.slots.get_unchecked_mut(idx) };
                    slot.offset = offset;
                    slot.len = len;
                    return;
                }
            }
            idx = (idx + 1) & self.mask;
        }
    }
}

thread_local! {
    static TL_BPE_CACHE: RefCell<FlatCache> = RefCell::new(FlatCache::new());
    static TL_FUSED_CACHE: RefCell<FlatCache> = RefCell::new(FlatCache::new());
}

const CACHE_SHARDS: usize = 64;

/// Slot bits per shared-cache shard: 64 shards x 4096 slots = 256k entries.
const SHARED_SHARD_BITS: usize = 12;

/// Cross-thread token cache: [`CACHE_SHARDS`] mutex-guarded [`FlatCache`]
/// shards. Compared to the previous `HashMap<String, Vec<u32>>` shards this
/// is allocation-free per insert (keys and ids are copied into per-shard
/// pools that retain capacity across clears) and bounded (a shard clears at
/// 3/4 load instead of growing forever), which both removes the two heap
/// allocations per cold pretoken and fixes unbounded memory growth on
/// diverse long-running traffic.
struct SharedCache {
    shards: Vec<Mutex<FlatCache>>,
}

impl SharedCache {
    fn new() -> Self {
        Self {
            shards: (0..CACHE_SHARDS)
                .map(|_| Mutex::new(FlatCache::with_bits(SHARED_SHARD_BITS)))
                .collect(),
        }
    }

    /// Shard selector. Uses the TOP bits of the same hash `FlatCache`
    /// indexes slots with (low bits), so the two are independent.
    #[inline]
    fn shard_index(key: &str) -> usize {
        (FlatCache::hash_str(key) >> (64 - 6)) as usize & (CACHE_SHARDS - 1)
    }

    #[inline]
    fn get_into(&self, key: &str, out: &mut Vec<u32>) -> bool {
        self.shards[Self::shard_index(key)]
            .lock()
            .unwrap()
            .get(key, out)
    }

    #[inline]
    fn insert(&self, key: &str, ids: &[u32]) {
        self.shards[Self::shard_index(key)]
            .lock()
            .unwrap()
            .insert(key, ids);
    }
}

/// Raw deserialization helper.
#[derive(Deserialize)]
struct RawBpe {
    #[serde(default)]
    vocab: Vocab,
    #[serde(default)]
    merges: Vec<Value>,
    #[allow(dead_code)]
    dropout: Option<f64>,
    #[allow(dead_code)]
    unk_token: Option<String>,
    #[allow(dead_code)]
    continuing_subword_prefix: Option<String>,
    #[allow(dead_code)]
    end_of_word_suffix: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    fuse_unk: bool,
    #[serde(default)]
    #[allow(dead_code)]
    byte_fallback: bool,
    #[serde(default)]
    ignore_merges: bool,
}

/// Monotonic counter for unique Bpe instance IDs.
static BPE_ID_COUNTER: AtomicUsize = AtomicUsize::new(1);

/// Entry in the BPE merge priority queue.
/// `key = (rank << 32) | pos`, `val = (left_c << 32) | right_c`.
#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(C)]
struct MergeEntry {
    key: u64,
    val: u64,
}

impl MergeEntry {
    #[inline(always)]
    fn new(rank: u32, pos: u32, left_c: u32, right_c: u32) -> Self {
        Self {
            key: (rank as u64) << 32 | pos as u64,
            val: (left_c as u64) << 32 | right_c as u64,
        }
    }

    #[inline(always)]
    fn pos(&self) -> u32 {
        self.key as u32
    }

    #[inline(always)]
    fn left_c(&self) -> u32 {
        (self.val >> 32) as u32
    }

    #[inline(always)]
    fn right_c(&self) -> u32 {
        self.val as u32
    }
}

impl Ord for MergeEntry {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

impl PartialOrd for MergeEntry {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Symbol in the merge linked list.
#[derive(Clone, Copy)]
struct MergeSymbol {
    c: u32,
    prev: i32,
    next: i32,
}

struct MergeScratch {
    symbols: Vec<MergeSymbol>,
    heap: BinaryHeap<Reverse<MergeEntry>>,
    heap_buf: Vec<Reverse<MergeEntry>>,
}

impl MergeScratch {
    fn new() -> Self {
        Self {
            symbols: Vec::new(),
            heap: BinaryHeap::new(),
            heap_buf: Vec::new(),
        }
    }
}

thread_local! {
    static TL_MERGE_SCRATCH: RefCell<MergeScratch> = RefCell::new(MergeScratch::new());
}

/// Interleaved slot for the ranked merge map (16 bytes).
#[derive(Clone, Copy)]
#[repr(C)]
struct RankedMergeSlot {
    key: u64,
    rank: u32,
    id: u32,
}

/// Open-addressing hash table storing `(left_id, right_id) → (rank, merged_id)`.
#[derive(Clone)]
struct RankedMergeMap {
    mask: usize,
    slots: Vec<RankedMergeSlot>,
}

impl RankedMergeMap {
    fn from_parsed(parsed: &ParsedMergeMap) -> Self {
        if parsed.is_empty() {
            return Self {
                mask: 0,
                slots: Vec::new(),
            };
        }
        let capacity = (parsed.len() * 2).next_power_of_two();
        let mask = capacity - 1;
        let mut slots = vec![
            RankedMergeSlot {
                key: EMPTY_KEY,
                rank: 0,
                id: 0
            };
            capacity
        ];

        for (&(t1, t2), &(rank, merged_id)) in parsed {
            let key = pack_pair(t1, t2);
            let mut idx = fx_hash(key) as usize & mask;
            loop {
                if slots[idx].key == EMPTY_KEY {
                    slots[idx] = RankedMergeSlot {
                        key,
                        rank,
                        id: merged_id,
                    };
                    break;
                }
                idx = (idx + 1) & mask;
            }
        }

        Self { mask, slots }
    }

    /// Look up the rank and merged token ID for a pair.
    #[inline(always)]
    fn get(&self, t1: u32, t2: u32) -> Option<(u32, u32)> {
        if self.slots.is_empty() {
            return None;
        }
        let key = pack_pair(t1, t2);
        let mut idx = fx_hash(key) as usize & self.mask;
        loop {
            let slot = unsafe { self.slots.get_unchecked(idx) };
            if slot.key == key {
                return Some((slot.rank, slot.id));
            }
            if slot.key == EMPTY_KEY {
                return None;
            }
            idx = (idx + 1) & self.mask;
        }
    }
}

/// CSR adjacency structure for merge pair discovery.
#[derive(Clone)]
struct MergeAdjacency {
    offsets: Vec<u32>,
    data: Vec<(u32, u32, u32)>, // (neighbor, rank, new_id)
}

impl MergeAdjacency {
    fn from_parsed(parsed: &ParsedMergeMap, vocab_size: usize) -> Self {
        let mut counts = vec![0u32; vocab_size];
        for &(left, _right) in parsed.keys() {
            counts[left as usize] += 1;
        }

        let mut offsets = Vec::with_capacity(vocab_size + 1);
        offsets.push(0u32);
        let mut running = 0u32;
        for &c in &counts {
            running += c;
            offsets.push(running);
        }

        let mut data = vec![(0u32, 0u32, 0u32); running as usize];
        let mut write_pos = offsets[..vocab_size].to_vec();
        for (&(left, right), &(rank, merged_id)) in parsed {
            let idx = write_pos[left as usize] as usize;
            data[idx] = (right, rank, merged_id);
            write_pos[left as usize] += 1;
        }

        for i in 0..vocab_size {
            let start = offsets[i] as usize;
            let end = offsets[i + 1] as usize;
            data[start..end].sort_unstable_by_key(|&(neighbor, _, _)| neighbor);
        }

        Self { offsets, data }
    }

    #[inline(always)]
    fn get(&self, left: u32, right: u32) -> Option<(u32, u32)> {
        let start = unsafe { *self.offsets.get_unchecked(left as usize) } as usize;
        let end = unsafe { *self.offsets.get_unchecked(left as usize + 1) } as usize;
        let slice = unsafe { self.data.get_unchecked(start..end) };
        match slice.binary_search_by_key(&right, |&(n, _, _)| n) {
            Ok(idx) => {
                let entry = unsafe { slice.get_unchecked(idx) };
                Some((entry.1, entry.2))
            }
            Err(_) => None,
        }
    }
}

#[derive(Deserialize)]
#[serde(try_from = "RawBpe")]
pub struct Bpe {
    #[serde(skip)]
    id: usize,
    #[serde(skip)]
    content_affinity_available: AtomicBool,
    daac: DoubleArrayAhoCorasick<TokenId>,
    merge_map: MergeMap,
    unmerge_map: Vec<(TokenId, TokenId)>,
    next_prefix_map: Vec<TokenId>,
    token_lens: Vec<u16>,
    shared_cache: SharedCache,
    fused_shared_cache: SharedCache,
    id_to_token: Vec<String>,
    token_to_id: FxHashMap<String, u32>,
    byte_to_initial_token: [u32; 256],
    ranked_merge_map: RankedMergeMap,
    byte_pair_initial: Vec<(u32, u32)>,
    merge_adj: MergeAdjacency,
    ignore_merges: bool,
}

impl TryFrom<RawBpe> for Bpe {
    type Error = String;

    fn try_from(raw: RawBpe) -> Result<Self> {
        let merge_map = parse_merges(&raw.vocab, &raw.merges)?;
        let mut bpe = Self::new(&raw.vocab, merge_map)?;
        bpe.ignore_merges = raw.ignore_merges;
        Ok(bpe)
    }
}

enum Decomposition {
    Pair(TokenId, TokenId),
    CharsNotInVocab,
    Stuck,
}

fn encoding_decomposition(text: &str, vocab: &Vocab, merge_map: &ParsedMergeMap) -> Decomposition {
    let mut tokens: Vec<TokenId> = Vec::new();
    for ch in text.chars() {
        let mut buf = [0u8; 4];
        let s = ch.encode_utf8(&mut buf);
        match vocab.get(s) {
            Some(&tid) => tokens.push(tid),
            None => return Decomposition::CharsNotInVocab,
        }
    }

    if tokens.len() < 2 {
        return Decomposition::CharsNotInVocab;
    }

    while tokens.len() > 2 {
        let mut best_rank = u32::MAX;
        let mut best_pos = usize::MAX;
        let mut best_new = 0;
        for i in 0..tokens.len() - 1 {
            let pair = (tokens[i], tokens[i + 1]);
            if let Some(&(rank, new_id)) = merge_map.get(&pair)
                && rank < best_rank
            {
                best_rank = rank;
                best_pos = i;
                best_new = new_id;
            }
        }
        if best_pos == usize::MAX {
            return Decomposition::Stuck;
        }
        tokens[best_pos] = best_new;
        tokens.remove(best_pos + 1);
    }

    Decomposition::Pair(tokens[0], tokens[1])
}

fn parse_merges(vocab: &Vocab, merges: &[Value]) -> Result<ParsedMergeMap> {
    let mut merge_map = ParsedMergeMap::new();
    for (rank, entry) in merges.iter().enumerate() {
        let (left, right) = parse_merge_entry(entry)?;
        let &left_id = vocab
            .get(left)
            .ok_or_else(|| format!("merge token not in vocab: {left:?}"))?;
        let &right_id = vocab
            .get(right)
            .ok_or_else(|| format!("merge token not in vocab: {right:?}"))?;
        let merged = format!("{left}{right}");
        let &merged_id = vocab
            .get(&merged)
            .ok_or_else(|| format!("merged token not in vocab: {merged:?}"))?;
        merge_map.insert((left_id, right_id), (rank as u32, merged_id));
    }
    Ok(merge_map)
}

fn parse_merge_entry(entry: &Value) -> Result<(&str, &str)> {
    match entry {
        Value::String(s) => {
            let (left, right) = s
                .split_once(' ')
                .ok_or_else(|| format!("invalid merge entry (no space): {s:?}"))?;
            Ok((left, right))
        }
        Value::Array(arr) if arr.len() == 2 => {
            let left = arr[0]
                .as_str()
                .ok_or_else(|| format!("merge element not a string: {:?}", arr[0]))?;
            let right = arr[1]
                .as_str()
                .ok_or_else(|| format!("merge element not a string: {:?}", arr[1]))?;
            Ok((left, right))
        }
        _ => Err(format!("unrecognized merge entry format: {entry:?}")),
    }
}

impl Bpe {
    pub fn new(vocab: &Vocab, merge_map: ParsedMergeMap) -> Result<Self> {
        if vocab.is_empty() {
            return Err("cannot build Bpe with empty vocabulary".into());
        }

        let vocab_r: std::collections::BTreeMap<u32, &str> =
            vocab.iter().map(|(s, &id)| (id, s.as_str())).collect();

        let id_to_token: Vec<String> = (0..=*vocab_r.keys().max().unwrap())
            .map(|t| {
                vocab_r
                    .get(&t)
                    .ok_or_else(|| format!("non-contiguous tokens - token {t} is missing"))
                    .map(|s| s.to_string())
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let max_token = vocab_r.keys().max().copied().unwrap();

        let mut unmerge_map = (0..=max_token).map(|t| (t, t)).collect::<Vec<_>>();
        let mut is_orphan = vec![false; (max_token + 1) as usize];
        for (&tid, text) in &vocab_r {
            if text.chars().count() < 2 {
                continue;
            }
            match encoding_decomposition(text, vocab, &merge_map) {
                Decomposition::Pair(left, right) => {
                    unmerge_map[tid as usize] = (left, right);
                }
                Decomposition::Stuck => {
                    is_orphan[tid as usize] = true;
                }
                Decomposition::CharsNotInVocab => {}
            }
        }

        let daac = DoubleArrayAhoCorasickBuilder::new()
            .match_kind(daachorse::MatchKind::LeftmostLongest)
            .build_with_values(vocab_r.iter().filter_map(|(&token, pattern)| {
                (!is_orphan[token as usize]).then_some((pattern, token))
            }))
            .map_err(|e| format!("error building DAAC: {e}"))?;

        let token_lens: Vec<u16> = (0..=max_token)
            .map(|t| {
                u16::try_from(vocab_r[&t].len())
                    .map_err(|_| format!("token {t} length {} exceeds u16::MAX", vocab_r[&t].len()))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let next_prefix_map: Vec<TokenId> = (0..=max_token)
            .map(|token| {
                let token_str = &vocab_r[&token];
                let Some((last_char_start, _)) = token_str.char_indices().next_back() else {
                    return INVALID_TOKEN;
                };
                if last_char_start == 0 {
                    return INVALID_TOKEN;
                }
                daac.leftmost_find_iter(&token_str[..last_char_start])
                    .next()
                    .map_or(INVALID_TOKEN, |m| m.value())
            })
            .collect();

        let flat_merge_map = MergeMap::from_parsed(&merge_map);
        let ranked_merge_map = RankedMergeMap::from_parsed(&merge_map);

        let mut byte_to_initial_token = [INVALID_TOKEN; 256];
        for byte_val in 0u16..256 {
            let ch = BYTE_TO_CHAR[byte_val as usize];
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            if let Some(&id) = vocab.get(s) {
                byte_to_initial_token[byte_val as usize] = id;
            }
        }

        // Pre-compute initial byte-pair merges (256×256 table).
        let mut byte_pair_initial = vec![(u32::MAX, 0u32); 65536];
        for b1 in 0u16..256 {
            let t1 = byte_to_initial_token[b1 as usize];
            if t1 == INVALID_TOKEN {
                continue;
            }
            for b2 in 0u16..256 {
                let t2 = byte_to_initial_token[b2 as usize];
                if t2 == INVALID_TOKEN {
                    continue;
                }
                if let Some((rank, new_id)) = ranked_merge_map.get(t1, t2) {
                    byte_pair_initial[b1 as usize * 256 + b2 as usize] = (rank, new_id);
                }
            }
        }

        let vocab_size = id_to_token.len();
        let merge_adj = MergeAdjacency::from_parsed(&merge_map, vocab_size);

        Ok(Self {
            id: BPE_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            content_affinity_available: AtomicBool::new(true),
            daac,
            merge_map: flat_merge_map,
            unmerge_map,
            next_prefix_map,
            token_lens,
            shared_cache: SharedCache::new(),
            fused_shared_cache: SharedCache::new(),
            id_to_token,
            token_to_id: {
                let mut m = HashMap::with_capacity_and_hasher(vocab.len(), FxBuildHasher);
                m.extend(vocab.iter().map(|(k, v)| (k.clone(), *v)));
                m
            },
            byte_to_initial_token,
            ranked_merge_map,
            byte_pair_initial,
            merge_adj,
            ignore_merges: false,
        })
    }

    pub fn is_compatible_token_pair(&self, mut t1: TokenId, mut t2: TokenId) -> bool {
        if t1 == INVALID_TOKEN {
            return false;
        }

        let mut limit = u32::MAX;
        loop {
            if let Some(t) = self.merge_map.get(t1, t2)
                && t < limit
            {
                return false;
            }

            if t1 > t2 {
                limit = t1;
                t1 = self.unmerge_map[t1 as usize].1;
                if t1 == limit {
                    limit = t2 + 1;
                    t2 = self.unmerge_map[t2 as usize].0;
                    if t2 + 1 == limit {
                        return true;
                    }
                }
            } else {
                limit = t2 + 1;
                t2 = self.unmerge_map[t2 as usize].0;
                if t2 + 1 == limit {
                    limit = t1;
                    t1 = self.unmerge_map[t1 as usize].1;
                    if t1 == limit {
                        return true;
                    }
                }
            }
        }
    }

    fn next_match(&self, input: &str) -> Option<TokenId> {
        let m = self.daac.leftmost_find_iter(input).next()?;
        (m.start() == 0).then(|| m.value())
    }

    pub fn tokenize(&self, input: &str) -> Result<Vec<TokenId>> {
        let mut out = Vec::new();
        self.tokenize_into(input, &mut out)?;
        Ok(out)
    }

    #[inline(always)]
    pub fn tokenize_into(&self, input: &str, out: &mut Vec<u32>) -> Result<()> {
        if input.is_empty() {
            return Ok(());
        }

        if let Some(token) = self.next_match(input)
            && self.token_lens[token as usize] as usize == input.len()
        {
            out.push(token);
            return Ok(());
        }

        let bpe_id = self.id;
        let hit = TL_BPE_CACHE.with(|c| {
            let c = c.borrow();
            if c.bpe_id != bpe_id {
                return false;
            }
            c.get(input, out)
        });
        if hit {
            return Ok(());
        }

        let start = out.len();
        if self.shared_cache.get_into(input, out) {
            TL_BPE_CACHE.with(|c| {
                let mut c = c.borrow_mut();
                if c.bpe_id != bpe_id {
                    c.bpe_id = bpe_id;
                    c.clear();
                }
                c.insert(input, &out[start..]);
            });
            return Ok(());
        }

        self.merge_all_encoded_into(input, out)?;

        let ids = &out[start..];
        TL_BPE_CACHE.with(|c| {
            let mut c = c.borrow_mut();
            if c.bpe_id != bpe_id {
                c.bpe_id = bpe_id;
                c.clear();
            }
            c.insert(input, ids);
        });
        self.shared_cache.insert(input, ids);

        Ok(())
    }

    /// Priority-queue BPE merge on already-encoded (ByteLevel) text.
    fn merge_all_encoded_into(&self, input: &str, out: &mut Vec<u32>) -> Result<()> {
        if input.is_empty() {
            return Ok(());
        }

        TL_MERGE_SCRATCH.with(|s| {
            let mut scratch = s.borrow_mut();
            scratch.symbols.clear();
            scratch.heap.clear();

            let mut n = 0usize;
            for ch in input.chars() {
                let mut buf = [0u8; 4];
                let s = ch.encode_utf8(&mut buf);
                let id = self
                    .token_to_id
                    .get(s)
                    .copied()
                    .ok_or_else(|| format!("character {ch:?} not in vocabulary"))?;
                scratch.symbols.push(MergeSymbol {
                    c: id,
                    prev: if n == 0 { -1 } else { (n - 1) as i32 },
                    next: -1,
                });
                if n > 0 {
                    scratch.symbols[n - 1].next = n as i32;
                }
                n += 1;
            }

            if n == 1 {
                out.push(scratch.symbols[0].c);
                return Ok(());
            }

            self.init_merge_heap(&mut scratch, n);
            self.run_merge_loop(&mut scratch, out);
            Ok(())
        })
    }

    /// Linear-scan BPE merge for short pretokens (`n <= SMALL_MERGE_MAX`
    /// initial symbols). Avoids the `BinaryHeap` entirely: a stack-resident
    /// doubly-linked list plus a per-position rank array, find-min by a short
    /// scan over stack `u32`s, merge (O(1) pointer update), then refresh only
    /// the two neighbor pairs. At these sizes this beats the heap's
    /// sift/stale-entry traffic and does zero heap allocation.
    ///
    /// Produces the identical token sequence as [`run_merge_loop`]: both
    /// process the globally lowest-`(rank, pos)` active pair each step (the
    /// heap's `MergeEntry` key is `(rank << 32) | pos`; the scan's strict
    /// `<` keeps the leftmost/lowest-`pos` position on ties). Enforced by
    /// `merge_small_matches_heap` differential test. `ids[..n]` are the
    /// per-byte initial token ids.
    fn merge_small_raw(
        &self,
        bytes: &[u8],
        ids: &mut [u32; SMALL_MERGE_MAX],
        n: usize,
        out: &mut Vec<u32>,
    ) {
        let mut next = [0u8; SMALL_MERGE_MAX];
        let mut prev = [0u8; SMALL_MERGE_MAX];
        let mut ranks = [u32::MAX; SMALL_MERGE_MAX];
        let mut new_ids = [0u32; SMALL_MERGE_MAX];
        for i in 0..n {
            next[i] = (i + 1) as u8;
            prev[i] = (i as u8).wrapping_sub(1); // prev[0] = 255 (>= n): sentinel
        }
        // Round-1 ranks via the dense byte-pair table: one direct-indexed
        // load per pair instead of a CSR neighbor scan.
        for i in 0..n - 1 {
            let (rank, new_id) =
                self.byte_pair_initial[bytes[i] as usize * 256 + bytes[i + 1] as usize];
            if rank != u32::MAX {
                ranks[i] = rank;
                new_ids[i] = new_id;
            }
        }
        loop {
            let mut best = u32::MAX;
            let mut best_i = 0usize;
            for (i, &rank) in ranks[..n - 1].iter().enumerate() {
                if rank < best {
                    best = rank;
                    best_i = i;
                }
            }
            if best == u32::MAX {
                break;
            }
            let i = best_i;
            ids[i] = new_ids[i];
            let dead = next[i] as usize;
            let new_right = next[dead] as usize;
            next[i] = new_right as u8;
            ranks[dead] = u32::MAX;
            if new_right < n {
                prev[new_right] = i as u8;
                match self.merge_adj.get(ids[i], ids[new_right]) {
                    Some((rank, new_id)) => {
                        ranks[i] = rank;
                        new_ids[i] = new_id;
                    }
                    None => ranks[i] = u32::MAX,
                }
            } else {
                ranks[i] = u32::MAX;
            }
            let left = prev[i] as usize;
            if left < n {
                match self.merge_adj.get(ids[left], ids[i]) {
                    Some((rank, new_id)) => {
                        ranks[left] = rank;
                        new_ids[left] = new_id;
                    }
                    None => ranks[left] = u32::MAX,
                }
            }
        }
        let mut i = 0usize;
        while i < n {
            out.push(ids[i]);
            i = next[i] as usize;
        }
    }

    /// `ignore_merges` whole-pretoken lookup: is the ByteLevel-encoded form
    /// of the entire pretoken a single vocab token? Encodes into a stack
    /// buffer for short pretokens (`BYTE_TO_CHAR` codepoints are <= U+0143,
    /// so <= 2 UTF-8 bytes per input byte) instead of allocating a `String`
    /// per cold pretoken; falls back to a heap `String` only for long ones.
    #[inline]
    fn whole_pretoken_id(&self, raw: &str) -> Option<u32> {
        const STACK: usize = 128;
        let bytes = raw.as_bytes();
        if bytes.len() * 2 <= STACK {
            let mut buf = [0u8; STACK];
            let mut n = 0;
            for &b in bytes {
                n += BYTE_TO_CHAR[b as usize].encode_utf8(&mut buf[n..]).len();
            }
            // SAFETY: buf[..n] is a concatenation of `char::encode_utf8`
            // outputs, hence valid UTF-8.
            let encoded = unsafe { std::str::from_utf8_unchecked(&buf[..n]) };
            self.token_to_id.get(encoded).copied()
        } else {
            let mut encoded = String::with_capacity(bytes.len() * 2);
            for &b in bytes {
                encoded.push(BYTE_TO_CHAR[b as usize]);
            }
            self.token_to_id.get(encoded.as_str()).copied()
        }
    }

    /// Priority-queue BPE merge on raw (pre-ByteLevel) bytes.
    fn merge_all_raw_into(&self, raw_input: &str, out: &mut Vec<u32>) -> Result<()> {
        if raw_input.is_empty() {
            return Ok(());
        }

        // Short pretokens (the common case) skip the heap + thread-local
        // scratch and use the stack-resident linear-scan merge. `n` initial
        // symbols == byte length, so the tier is known up front.
        let bytes = raw_input.as_bytes();
        if bytes.len() <= SMALL_MERGE_MAX {
            let n = bytes.len();
            let mut ids = [0u32; SMALL_MERGE_MAX];
            for (i, &byte) in bytes.iter().enumerate() {
                let id = self.byte_to_initial_token[byte as usize];
                if id == INVALID_TOKEN {
                    return Err(format!("byte 0x{byte:02x} has no token in vocabulary"));
                }
                ids[i] = id;
            }
            if n == 1 {
                out.push(ids[0]);
            } else {
                self.merge_small_raw(bytes, &mut ids, n, out);
            }
            return Ok(());
        }

        self.merge_all_raw_heap_into(raw_input, out)
    }

    /// Reference heap implementation used for long pretokens and as the
    /// correctness oracle for the short linear-scan tier.
    fn merge_all_raw_heap_into(&self, raw_input: &str, out: &mut Vec<u32>) -> Result<()> {
        TL_MERGE_SCRATCH.with(|s| {
            let mut scratch = s.borrow_mut();
            scratch.symbols.clear();
            scratch.heap.clear();
            scratch.heap_buf.clear();

            let bytes = raw_input.as_bytes();
            let n = bytes.len();
            let mut prev_byte = 0u8;
            for (i, &byte) in bytes.iter().enumerate() {
                let id = self.byte_to_initial_token[byte as usize];
                if id == INVALID_TOKEN {
                    return Err(format!("byte 0x{byte:02x} has no token in vocabulary"));
                }
                scratch.symbols.push(MergeSymbol {
                    c: id,
                    prev: if i == 0 { -1 } else { (i - 1) as i32 },
                    next: if i == n - 1 { -1 } else { (i + 1) as i32 },
                });
                // Check pair with previous byte via pre-computed table.
                if i > 0 {
                    let (rank, _new_id) =
                        self.byte_pair_initial[prev_byte as usize * 256 + byte as usize];
                    if rank != u32::MAX {
                        scratch.heap_buf.push(Reverse(MergeEntry::new(
                            rank,
                            (i - 1) as u32,
                            self.byte_to_initial_token[prev_byte as usize],
                            id,
                        )));
                    }
                }
                prev_byte = byte;
            }

            if n == 1 {
                out.push(scratch.symbols[0].c);
                return Ok(());
            }

            // Bulk heapify.
            let mut tmp = std::mem::take(&mut scratch.heap_buf);
            scratch.heap.extend(tmp.drain(..));
            scratch.heap_buf = tmp;

            self.run_merge_loop(&mut scratch, out);

            Ok(())
        })
    }

    /// Seed the priority queue with all initial adjacent pairs.
    #[inline(always)]
    fn init_merge_heap(&self, scratch: &mut MergeScratch, n: usize) {
        let symbols = &scratch.symbols;
        scratch.heap.extend((0..n - 1).filter_map(|i| {
            let left = symbols[i].c;
            let right = symbols[i + 1].c;
            self.merge_adj
                .get(left, right)
                .map(|(rank, _new_id)| Reverse(MergeEntry::new(rank, i as u32, left, right)))
        }));
    }

    #[inline(always)]
    fn run_merge_loop(&self, scratch: &mut MergeScratch, out: &mut Vec<u32>) {
        let symbols = &mut scratch.symbols;
        let heap = &mut scratch.heap;

        while let Some(Reverse(entry)) = heap.pop() {
            let pos = entry.pos() as usize;
            let sym = symbols[pos];

            // Stale-entry check.
            let left_c = entry.left_c();
            let right_c = entry.right_c();
            if sym.c != left_c {
                continue;
            }
            let next_idx = sym.next;
            if next_idx < 0 {
                continue;
            }
            let next_idx = next_idx as usize;
            let next_sym = symbols[next_idx];
            if next_sym.c != right_c {
                continue;
            }

            // Derive new_id from adjacency list.
            let new_id = match self.merge_adj.get(left_c, right_c) {
                Some((_, nid)) => nid,
                None => continue,
            };

            // Merge: left symbol absorbs right.
            symbols[pos].c = new_id;
            symbols[pos].next = next_sym.next;
            if next_sym.next >= 0 {
                symbols[next_sym.next as usize].prev = pos as i32;
            }
            symbols[next_idx].c = INVALID_TOKEN;

            // Discover new adjacent pairs.
            if sym.prev >= 0 {
                let prev_c = symbols[sym.prev as usize].c;
                if let Some((rank, _)) = self.merge_adj.get(prev_c, new_id) {
                    heap.push(Reverse(MergeEntry::new(
                        rank,
                        sym.prev as u32,
                        prev_c,
                        new_id,
                    )));
                }
            }
            let new_next = symbols[pos].next;
            if new_next >= 0 {
                let next_c = symbols[new_next as usize].c;
                if let Some((rank, _)) = self.merge_adj.get(new_id, next_c) {
                    heap.push(Reverse(MergeEntry::new(rank, pos as u32, new_id, next_c)));
                }
            }
        }

        let mut i: i32 = 0;
        while i >= 0 {
            let sym = symbols[i as usize];
            out.push(sym.c);
            i = sym.next;
        }
    }

    #[inline(always)]
    pub fn tokenize_into_fused(&self, raw_input: &str, out: &mut Vec<u32>) -> Result<()> {
        if raw_input.is_empty() {
            return Ok(());
        }

        let bpe_id = self.id;
        let hit = TL_FUSED_CACHE.with(|c| {
            let c = c.borrow();
            if c.bpe_id != bpe_id {
                return false;
            }
            c.get(raw_input, out)
        });
        if hit {
            return Ok(());
        }

        let start = out.len();
        if self.fused_shared_cache.get_into(raw_input, out) {
            TL_FUSED_CACHE.with(|c| {
                let mut c = c.borrow_mut();
                if c.bpe_id != bpe_id {
                    c.bpe_id = bpe_id;
                    c.clear();
                }
                c.insert(raw_input, &out[start..]);
            });
            return Ok(());
        }

        if self.ignore_merges
            && let Some(id) = self.whole_pretoken_id(raw_input)
        {
            out.push(id);
            return Ok(());
        }

        self.merge_all_raw_into(raw_input, out)?;

        let ids = &out[start..];
        TL_FUSED_CACHE.with(|c| {
            let mut c = c.borrow_mut();
            if c.bpe_id != bpe_id {
                c.bpe_id = bpe_id;
                c.clear();
            }
            c.insert(raw_input, ids);
        });
        self.fused_shared_cache.insert(raw_input, ids);

        Ok(())
    }

    pub fn tokenize_batch_fused(
        &self,
        buffer: &str,
        splits: &[crate::pre_tokenized::Split],
        out: &mut Vec<u32>,
    ) -> Result<()> {
        self.tokenize_batch_fused_impl::<false>(buffer, splits, out, &mut Vec::new())
    }

    pub fn tokenize_batch_fused_indexed(
        &self,
        buffer: &str,
        splits: &[crate::pre_tokenized::Split],
        out: &mut Vec<u32>,
        ends: &mut Vec<usize>,
    ) -> Result<()> {
        self.tokenize_batch_fused_impl::<true>(buffer, splits, out, ends)
    }

    fn tokenize_batch_fused_impl<const RECORD_ENDS: bool>(
        &self,
        buffer: &str,
        splits: &[crate::pre_tokenized::Split],
        out: &mut Vec<u32>,
        ends: &mut Vec<usize>,
    ) -> Result<()> {
        let bpe_id = self.id;
        TL_FUSED_CACHE.with(|c| {
            let mut cache = c.borrow_mut();
            if cache.bpe_id != bpe_id {
                cache.bpe_id = bpe_id;
                cache.clear();
            }

            for split in splits {
                if let Some(id) = split.token_id {
                    out.push(id);
                } else if !split.range.is_empty() {
                    let text = &buffer[split.range.clone()];
                    if text.is_empty() {
                        if RECORD_ENDS {
                            ends.push(out.len());
                        }
                        continue;
                    }

                    if cache.get(text, out) {
                        if RECORD_ENDS {
                            ends.push(out.len());
                        }
                        continue;
                    }

                    let start = out.len();
                    if self.fused_shared_cache.get_into(text, out) {
                        cache.insert(text, &out[start..]);
                        if RECORD_ENDS {
                            ends.push(out.len());
                        }
                        continue;
                    }

                    if self.ignore_merges
                        && let Some(id) = self.whole_pretoken_id(text)
                    {
                        out.push(id);
                        cache.insert(text, &out[start..]);
                        if RECORD_ENDS {
                            ends.push(out.len());
                        }
                        continue;
                    }

                    self.merge_all_raw_into(text, out)?;

                    cache.insert(text, &out[start..]);
                    self.fused_shared_cache.insert(text, &out[start..]);
                }
                if RECORD_ENDS {
                    ends.push(out.len());
                }
            }
            Ok(())
        })
    }

    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(String::as_str)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub(crate) fn take_content_affinity(&self) -> bool {
        self.content_affinity_available
            .swap(false, Ordering::Relaxed)
    }
}

impl Clone for Bpe {
    fn clone(&self) -> Self {
        Self {
            id: BPE_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            content_affinity_available: AtomicBool::new(true),
            daac: self.daac.clone(),
            merge_map: self.merge_map.clone(),
            unmerge_map: self.unmerge_map.clone(),
            next_prefix_map: self.next_prefix_map.clone(),
            token_lens: self.token_lens.clone(),
            shared_cache: SharedCache::new(),
            fused_shared_cache: SharedCache::new(),
            id_to_token: self.id_to_token.clone(),
            token_to_id: self.token_to_id.clone(),
            byte_to_initial_token: self.byte_to_initial_token,
            ranked_merge_map: self.ranked_merge_map.clone(),
            byte_pair_initial: self.byte_pair_initial.clone(),
            merge_adj: self.merge_adj.clone(),
            ignore_merges: self.ignore_merges,
        }
    }
}

impl fmt::Debug for Bpe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Bpe")
            .field("vocab_size", &self.token_lens.len())
            .field("merges", &self.merge_map.len())
            .finish()
    }
}

impl PartialEq for Bpe {
    fn eq(&self, other: &Self) -> bool {
        self.daac == other.daac
            && self.merge_map == other.merge_map
            && self.unmerge_map == other.unmerge_map
            && self.next_prefix_map == other.next_prefix_map
            && self.token_lens == other.token_lens
            && self.ignore_merges == other.ignore_merges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json_structs::ModelConfig;

    fn test_bpe() -> Bpe {
        let vocab: Vocab = [
            ("a", 0),
            ("b", 1),
            ("c", 2),
            ("d", 3),
            ("ab", 4),
            ("cd", 5),
            ("abcd", 6),
        ]
        .into_iter()
        .map(|(s, id)| (s.to_string(), id))
        .collect();

        let merges: Vec<Value> = vec![
            Value::String("a b".into()),
            Value::String("c d".into()),
            Value::String("ab cd".into()),
        ];

        let merge_map = parse_merges(&vocab, &merges).unwrap();
        Bpe::new(&vocab, merge_map).unwrap()
    }

    #[test]
    fn empty_input() {
        let bpe = test_bpe();
        assert_eq!(bpe.tokenize("").unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn single_char() {
        let bpe = test_bpe();
        assert_eq!(bpe.tokenize("a").unwrap(), vec![0]);
        assert_eq!(bpe.tokenize("d").unwrap(), vec![3]);
    }

    #[test]
    fn simple_merge() {
        let bpe = test_bpe();
        assert_eq!(bpe.tokenize("ab").unwrap(), vec![4]);
        assert_eq!(bpe.tokenize("cd").unwrap(), vec![5]);
    }

    #[test]
    fn chained_merge() {
        let bpe = test_bpe();
        assert_eq!(bpe.tokenize("abcd").unwrap(), vec![6]);
    }

    #[test]
    fn partial_merge() {
        let bpe = test_bpe();
        assert_eq!(bpe.tokenize("abc").unwrap(), vec![4, 2]);
    }

    #[test]
    fn repeated_merge() {
        let bpe = test_bpe();
        assert_eq!(bpe.tokenize("abab").unwrap(), vec![4, 4]);
    }

    #[test]
    fn content_affinity_is_available_once_per_bpe() {
        let bpe = test_bpe();
        assert!(bpe.take_content_affinity());
        assert!(!bpe.take_content_affinity());
    }

    #[test]
    fn merge_small_matches_heap() {
        let bpe = test_bpe();
        let alphabet = [b'a', b'b', b'c', b'd'];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;

        for len in 1..=SMALL_MERGE_MAX {
            for _ in 0..512 {
                let mut bytes = Vec::with_capacity(len);
                for _ in 0..len {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    bytes.push(alphabet[state as usize & 3]);
                }
                let input = std::str::from_utf8(&bytes).unwrap();
                let mut small = Vec::new();
                let mut heap = Vec::new();

                bpe.merge_all_raw_into(input, &mut small).unwrap();
                bpe.merge_all_raw_heap_into(input, &mut heap).unwrap();
                assert_eq!(small, heap, "short merge mismatch for {input:?}");
            }
        }
    }

    #[test]
    fn deserialize_from_json() {
        let json = serde_json::json!({
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "ab": 2},
            "merges": ["a b"]
        });
        let config: ModelConfig = serde_json::from_value(json).unwrap();
        assert!(matches!(config, ModelConfig::Bpe(_)));
    }

    #[test]
    fn deserialize_array_merges() {
        let json = serde_json::json!({
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "ab": 2},
            "merges": [["a", "b"]]
        });
        let config: ModelConfig = serde_json::from_value(json).unwrap();
        let ModelConfig::Bpe(bpe) = config else {
            panic!("expected Bpe variant");
        };
        assert_eq!(bpe.tokenize("ab").unwrap(), vec![2]);
    }

    #[test]
    fn cache_returns_same_result() {
        let vocab: Vocab = [("a", 0), ("b", 1), ("ab", 2)]
            .into_iter()
            .map(|(s, id)| (s.to_string(), id))
            .collect();
        let merges = vec![Value::String("a b".into())];
        let merge_map = parse_merges(&vocab, &merges).unwrap();
        let bpe = Bpe::new(&vocab, merge_map).unwrap();

        let first = bpe.tokenize("ab").unwrap();
        let second = bpe.tokenize("ab").unwrap();
        assert_eq!(first, second);
        assert_eq!(first, vec![2]);
    }
}
