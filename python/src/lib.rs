use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, LazyLock, RwLock},
};

use numpy::IntoPyArray;
use pyo3::exceptions::{PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyString};
use pyo3_async_runtimes::tokio::future_into_py;
use rayon::prelude::*;
use serde_json::Value;
use tokio::runtime::Runtime;

static TOKIO_RUNTIME: LazyLock<Arc<Runtime>> = LazyLock::new(|| {
    Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("failed to create global Tokio runtime"),
    )
});

fn extract_string_vec(value: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<String>> {
    if value.downcast::<PyString>().is_ok() {
        return Err(PyTypeError::new_err(format!(
            "{name} must be a sequence or set of strings, not a single string"
        )));
    }
    if let Ok(values) = value.extract::<Vec<String>>() {
        return Ok(values);
    }
    if let Ok(values) = value.extract::<HashSet<String>>() {
        return Ok(values.into_iter().collect());
    }
    Err(PyTypeError::new_err(format!(
        "{name} must be a sequence or set of strings"
    )))
}

// ---------------------------------------------------------------------------
// PyEncoding
// ---------------------------------------------------------------------------

/// Minimal stand-in for `tokenizers.Encoding`.
///
/// Returned directly by `Tokenizer.encode` and `Tokenizer.encode_batch` so
/// no Python-side wrapping is needed.  Fields that `fastokens` does not
/// track (`tokens`, `offsets`, `sequence_ids`, `word_ids`) have getters that
/// raise `NotImplementedError` to match the HuggingFace API surface.
#[pyclass(name = "Encoding")]
pub struct PyEncoding {
    #[pyo3(get, set)]
    pub ids: Vec<u32>,
    #[pyo3(get, set)]
    pub attention_mask: Vec<u32>,
    #[pyo3(get, set)]
    pub type_ids: Vec<u32>,
    #[pyo3(get, set)]
    pub special_tokens_mask: Vec<u32>,
    #[pyo3(get, set)]
    pub n_sequences: usize,
    // Backing storage for set-only properties.
    _sequence_ids: Vec<Option<i64>>,
    _word_ids: Vec<Option<i64>>,
}

impl PyEncoding {
    pub fn make(ids: Vec<u32>, attention_mask: Vec<u32>) -> Self {
        let n = ids.len();
        Self {
            type_ids: vec![0u32; n],
            special_tokens_mask: vec![0u32; n],
            n_sequences: 1,
            _sequence_ids: vec![Some(0); n],
            _word_ids: vec![None; n],
            ids,
            attention_mask,
        }
    }

    fn apply_slice(&mut self, start: usize, end: usize) {
        self.ids = self.ids[start..end].to_vec();
        self.attention_mask = self.attention_mask[start..end].to_vec();
        self.type_ids = self.type_ids[start..end].to_vec();
        self.special_tokens_mask = self.special_tokens_mask[start..end].to_vec();
        self._sequence_ids = self._sequence_ids[start..end].to_vec();
        self._word_ids = self._word_ids[start..end].to_vec();
    }

    fn extend_right(&mut self, pad_id: u32, pad_type_id: u32, count: usize) {
        self.ids.extend(vec![pad_id; count]);
        self.attention_mask.extend(vec![0u32; count]);
        self.type_ids.extend(vec![pad_type_id; count]);
        self.special_tokens_mask.extend(vec![0u32; count]);
        self._sequence_ids.extend(vec![None; count]);
        self._word_ids.extend(vec![None; count]);
    }

    fn extend_left(&mut self, pad_id: u32, pad_type_id: u32, count: usize) {
        let mut ids = vec![pad_id; count];
        ids.extend_from_slice(&self.ids);
        let mut mask = vec![0u32; count];
        mask.extend_from_slice(&self.attention_mask);
        let mut type_ids = vec![pad_type_id; count];
        type_ids.extend_from_slice(&self.type_ids);
        let mut special = vec![0u32; count];
        special.extend_from_slice(&self.special_tokens_mask);
        let mut seq_ids = vec![None; count];
        seq_ids.extend_from_slice(&self._sequence_ids);
        let mut word_ids = vec![None; count];
        word_ids.extend_from_slice(&self._word_ids);
        self.ids = ids;
        self.attention_mask = mask;
        self.type_ids = type_ids;
        self.special_tokens_mask = special;
        self._sequence_ids = seq_ids;
        self._word_ids = word_ids;
    }
}

#[pymethods]
impl PyEncoding {
    #[new]
    #[pyo3(signature = (ids, attention_mask = None))]
    fn new(ids: Vec<u32>, attention_mask: Option<Vec<u32>>) -> Self {
        let n = ids.len();
        let mask = attention_mask.unwrap_or_else(|| vec![1u32; n]);
        Self::make(ids, mask)
    }

    fn __len__(&self) -> usize {
        self.ids.len()
    }

    fn __repr__(&self) -> String {
        format!("Encoding(num_tokens={})", self.ids.len())
    }

    /// Move selected fields into NumPy uint32 arrays.
    ///
    /// This consumes/drains the encoding's per-token fields. The returned dict
    /// contains only requested arrays; unrequested fields are cleared.
    #[pyo3(signature = (
        ids = true,
        attention_mask = false,
        type_ids = false,
        special_tokens_mask = false
    ))]
    fn into_numpy<'py>(
        &mut self,
        py: Python<'py>,
        ids: bool,
        attention_mask: bool,
        type_ids: bool,
        special_tokens_mask: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        if !(ids || attention_mask || type_ids || special_tokens_mask) {
            return Err(PyValueError::new_err(
                "at least one field must be selected for into_numpy()",
            ));
        }

        let out = PyDict::new(py);
        let ids_vec = std::mem::take(&mut self.ids);
        let attention_mask_vec = std::mem::take(&mut self.attention_mask);
        let type_ids_vec = std::mem::take(&mut self.type_ids);
        let special_tokens_mask_vec = std::mem::take(&mut self.special_tokens_mask);
        std::mem::take(&mut self._sequence_ids);
        std::mem::take(&mut self._word_ids);
        self.n_sequences = 0;

        if ids {
            out.set_item("ids", ids_vec.into_pyarray(py))?;
        }
        if attention_mask {
            out.set_item("attention_mask", attention_mask_vec.into_pyarray(py))?;
        }
        if type_ids {
            out.set_item("type_ids", type_ids_vec.into_pyarray(py))?;
        }
        if special_tokens_mask {
            out.set_item(
                "special_tokens_mask",
                special_tokens_mask_vec.into_pyarray(py),
            )?;
        }

        Ok(out)
    }

    // -- Properties that raise NotImplementedError ----------------------

    #[getter]
    fn tokens(&self) -> PyResult<Vec<String>> {
        Err(PyNotImplementedError::new_err(
            "fastokens does not track token strings; \
             use Tokenizer.id_to_token() to convert individual IDs",
        ))
    }
    #[setter]
    fn set_tokens(&mut self, _v: &Bound<'_, PyAny>) {}

    #[getter]
    fn offsets(&self) -> PyResult<Vec<(usize, usize)>> {
        Err(PyNotImplementedError::new_err(
            "fastokens does not track character offsets",
        ))
    }
    #[setter]
    fn set_offsets(&mut self, _v: &Bound<'_, PyAny>) {}

    #[getter]
    fn sequence_ids(&self) -> PyResult<Vec<Option<i64>>> {
        Err(PyNotImplementedError::new_err(
            "fastokens does not track sequence IDs",
        ))
    }
    #[setter]
    fn set_sequence_ids(&mut self, value: Vec<Option<i64>>) {
        self._sequence_ids = value;
    }

    #[getter]
    fn word_ids(&self) -> PyResult<Vec<Option<i64>>> {
        Err(PyNotImplementedError::new_err(
            "fastokens does not track word IDs",
        ))
    }
    #[setter]
    fn set_word_ids(&mut self, value: Vec<Option<i64>>) {
        self._word_ids = value;
    }

    #[getter]
    fn words(&self) -> PyResult<Vec<Option<i64>>> {
        Err(PyNotImplementedError::new_err(
            "fastokens does not track word IDs",
        ))
    }
    #[setter]
    fn set_words(&mut self, value: Vec<Option<i64>>) {
        self._word_ids = value;
    }

    /// Always empty — fastokens does not produce overflowing sequences.
    #[getter]
    fn overflowing<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        PyList::empty(py)
    }
    #[setter]
    fn set_overflowing(&mut self, _v: &Bound<'_, PyAny>) {}

    // -- Sequence ID helper ---------------------------------------------

    fn set_sequence_id(&mut self, sequence_id: i64) {
        let n = self.ids.len();
        self._sequence_ids = vec![Some(sequence_id); n];
    }

    // -- Positional mapping (all raise NotImplementedError) -------------

    #[pyo3(signature = (char_pos, sequence_index = 0))]
    fn char_to_token(&self, char_pos: usize, sequence_index: usize) -> PyResult<Option<usize>> {
        let _ = (char_pos, sequence_index);
        Err(PyNotImplementedError::new_err(
            "fastokens does not track character offsets",
        ))
    }

    #[pyo3(signature = (char_pos, sequence_index = 0))]
    fn char_to_word(&self, char_pos: usize, sequence_index: usize) -> PyResult<Option<usize>> {
        let _ = (char_pos, sequence_index);
        Err(PyNotImplementedError::new_err(
            "fastokens does not track word IDs",
        ))
    }

    fn token_to_chars(&self, token_index: usize) -> PyResult<Option<(usize, usize)>> {
        let _ = token_index;
        Err(PyNotImplementedError::new_err(
            "fastokens does not track character offsets",
        ))
    }

    fn token_to_sequence(&self, token_index: usize) -> PyResult<Option<usize>> {
        let _ = token_index;
        Err(PyNotImplementedError::new_err(
            "fastokens does not track sequence IDs",
        ))
    }

    fn token_to_word(&self, token_index: usize) -> PyResult<Option<usize>> {
        let _ = token_index;
        Err(PyNotImplementedError::new_err(
            "fastokens does not track word IDs",
        ))
    }

    #[pyo3(signature = (word_index, sequence_index = 0))]
    fn word_to_chars(
        &self,
        word_index: usize,
        sequence_index: usize,
    ) -> PyResult<Option<(usize, usize)>> {
        let _ = (word_index, sequence_index);
        Err(PyNotImplementedError::new_err(
            "fastokens does not track character offsets",
        ))
    }

    #[pyo3(signature = (word_index, sequence_index = 0))]
    fn word_to_tokens(
        &self,
        word_index: usize,
        sequence_index: usize,
    ) -> PyResult<Option<(usize, usize)>> {
        let _ = (word_index, sequence_index);
        Err(PyNotImplementedError::new_err(
            "fastokens does not track word IDs",
        ))
    }

    // -- Truncate / pad -------------------------------------------------

    #[pyo3(signature = (max_length, stride = 0, direction = "right"))]
    fn truncate(&mut self, max_length: usize, stride: usize, direction: &str) {
        let _ = stride;
        let n = self.ids.len();
        if n <= max_length {
            return;
        }
        if direction == "left" {
            self.apply_slice(n - max_length, n);
        } else {
            self.apply_slice(0, max_length);
        }
    }

    #[pyo3(signature = (length, direction = "right", pad_id = 0, pad_type_id = 0, pad_token = "[PAD]"))]
    fn pad(
        &mut self,
        length: usize,
        direction: &str,
        pad_id: u32,
        pad_type_id: u32,
        pad_token: &str,
    ) {
        let _ = pad_token;
        let n = self.ids.len();
        if length <= n {
            return;
        }
        let deficit = length - n;
        if direction == "left" {
            self.extend_left(pad_id, pad_type_id, deficit);
        } else {
            self.extend_right(pad_id, pad_type_id, deficit);
        }
    }

    // -- Merge ----------------------------------------------------------

    #[staticmethod]
    #[pyo3(signature = (encodings, growing_offsets = true))]
    fn merge(py: Python<'_>, encodings: Vec<Py<PyEncoding>>, growing_offsets: bool) -> PyEncoding {
        let _ = growing_offsets;
        let mut ids: Vec<u32> = vec![];
        let mut attention_mask: Vec<u32> = vec![];
        let mut type_ids: Vec<u32> = vec![];
        let mut special_tokens_mask: Vec<u32> = vec![];
        let mut n_sequences: usize = 0;
        let mut seq_ids: Vec<Option<i64>> = vec![];
        let mut word_ids: Vec<Option<i64>> = vec![];

        for enc_py in &encodings {
            let enc = enc_py.borrow(py);
            ids.extend_from_slice(&enc.ids);
            attention_mask.extend_from_slice(&enc.attention_mask);
            type_ids.extend_from_slice(&enc.type_ids);
            special_tokens_mask.extend_from_slice(&enc.special_tokens_mask);
            n_sequences += enc.n_sequences;
            seq_ids.extend_from_slice(&enc._sequence_ids);
            word_ids.extend_from_slice(&enc._word_ids);
        }

        PyEncoding {
            ids,
            attention_mask,
            type_ids,
            special_tokens_mask,
            n_sequences,
            _sequence_ids: seq_ids,
            _word_ids: word_ids,
        }
    }
}

// ---------------------------------------------------------------------------
// TruncationParams / PaddingParams
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct TruncationParams {
    max_length: usize,
    stride: usize,
    strategy: String,
    direction: String,
}

#[derive(Clone)]
struct PaddingParams {
    direction: String,
    pad_id: u32,
    pad_type_id: u32,
    pad_token: String,
    length: Option<usize>,
    pad_to_multiple_of: Option<usize>,
}

fn build_encoding(ids: Vec<u32>, pad: Option<&PaddingParams>, target: usize) -> PyEncoding {
    let n = ids.len();
    let mut enc = PyEncoding::make(ids, vec![1u32; n]);
    if let Some(p) = pad {
        enc.pad(target, &p.direction, p.pad_id, p.pad_type_id, &p.pad_token);
    }
    enc
}

// ---------------------------------------------------------------------------
// PyPostProcessor
// ---------------------------------------------------------------------------

/// Python-facing post-processor object — mirrors `tokenizers.processors.*`.
///
/// Holds the JSON representation of the post-processor so that:
/// - `str(pp)` returns JSON (the setter calls `str()` on whatever it receives)
/// - the object round-trips correctly through the getter/setter pair
#[pyclass(name = "PostProcessor")]
#[derive(Clone)]
struct PyPostProcessor {
    json: String,
}

#[pymethods]
impl PyPostProcessor {
    fn __str__(&self) -> &str {
        &self.json
    }
    fn __repr__(&self) -> &str {
        &self.json
    }
}

// ---------------------------------------------------------------------------
// PyTokenizer
// ---------------------------------------------------------------------------

/// Mutable state guarded by `PyTokenizer::state`.
///
/// All read paths (encode/decode/getters) hold a read lock; mutators
/// (`enable_truncation`, `set_post_processor`, …) hold a write lock so they
/// cannot race with concurrent reads when the GIL is released.
struct TokenizerState {
    inner: fastokens::Tokenizer,
    trunc: Option<TruncationParams>,
    pad: Option<PaddingParams>,
    /// Cached JSON of the current post-processor (for the getter).
    post_processor_json: Option<String>,
}

impl TokenizerState {
    fn do_truncate(&self, ids: &mut Vec<u32>) {
        let Some(ref t) = self.trunc else { return };
        if ids.len() <= t.max_length {
            return;
        }
        if t.direction == "left" {
            ids.drain(..ids.len() - t.max_length);
        } else {
            ids.truncate(t.max_length);
        }
    }

    fn single_pad_target(&self, n: usize) -> usize {
        let Some(ref p) = self.pad else { return n };
        let base = p.length.unwrap_or(n).max(n);
        match p.pad_to_multiple_of {
            Some(m) if m > 0 => (base + m - 1) / m * m,
            _ => base,
        }
    }

    fn encode_batch_encodings(
        &self,
        inputs: &[String],
        add_special_tokens: bool,
        split_special_tokens: bool,
    ) -> Result<Vec<PyEncoding>, String> {
        let mut batch: Vec<Vec<u32>> = inputs
            .par_iter()
            .map(|s| {
                self.inner
                    .encode_with_options(s.as_str(), add_special_tokens, split_special_tokens)
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;

        for ids in &mut batch {
            self.do_truncate(ids);
        }

        let pad_target: Option<usize> = self.pad.as_ref().map(|p| {
            let max_len = batch.iter().map(|ids| ids.len()).max().unwrap_or(0);
            let base = p.length.unwrap_or(max_len).max(max_len);
            match p.pad_to_multiple_of {
                Some(m) if m > 0 => (base + m - 1) / m * m,
                _ => base,
            }
        });

        Ok(batch
            .into_iter()
            .map(|ids| {
                let target = pad_target.unwrap_or(ids.len());
                build_encoding(ids, self.pad.as_ref(), target)
            })
            .collect())
    }

    /// Parse `json`, update the Rust post-processor in place, and cache the JSON.
    fn update_post_processor_json(&mut self, json: &str) -> PyResult<()> {
        use fastokens::json_structs::PostProcessorConfig;
        use fastokens::post_processors::PostProcessor;

        let value: Value = serde_json::from_str(json)
            .map_err(|e| PyValueError::new_err(format!("invalid post-processor JSON: {e}")))?;
        let config: PostProcessorConfig = serde_json::from_value(value)
            .map_err(|e| PyValueError::new_err(format!("cannot parse post-processor: {e}")))?;
        let pp =
            PostProcessor::from_config(config).map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner.set_post_processor(Some(pp));
        self.post_processor_json = Some(json.to_string());
        Ok(())
    }
}

#[pyclass(name = "StructuralTokenConfig")]
struct PyStructuralTokenConfig {
    inner: Arc<fastokens::StructuralTokenConfig>,
}

#[pymethods]
impl PyStructuralTokenConfig {
    /// Build constant structural-token state for rendered-prompt encoding.
    ///
    /// Include every token string that the rendered template may use as a
    /// structural boundary, including tag-like non-special added tokens.
    #[new]
    #[pyo3(signature = (structural_tokens, non_special_added_tokens = None))]
    fn new(
        structural_tokens: &Bound<'_, PyAny>,
        non_special_added_tokens: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let structural_tokens = extract_string_vec(structural_tokens, "structural_tokens")?;
        let non_special_added_tokens = match non_special_added_tokens {
            Some(value) => extract_string_vec(value, "non_special_added_tokens")?
                .into_iter()
                .collect(),
            None => HashSet::new(),
        };
        let inner =
            fastokens::StructuralTokenConfig::new(&structural_tokens, &non_special_added_tokens)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
}

/// An LLM tokenizer backed by `tokenizer.json`.
#[pyclass(name = "Tokenizer")]
struct PyTokenizer {
    state: Arc<RwLock<TokenizerState>>,
}

impl PyTokenizer {
    fn read(&self) -> std::sync::RwLockReadGuard<'_, TokenizerState> {
        self.state.read().expect("PyTokenizer state lock poisoned")
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, TokenizerState> {
        self.state.write().expect("PyTokenizer state lock poisoned")
    }

    /// Build from a raw JSON string, extracting the post-processor field so
    /// the getter can return it without needing to re-serialize.
    fn build_from_str(json: &str, py: Python<'_>) -> PyResult<Self> {
        let value: Value =
            serde_json::from_str(json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let post_processor_json = value
            .get("post_processor")
            .filter(|v| !v.is_null())
            .map(|v| v.to_string());
        let inner = py
            .allow_threads(|| fastokens::Tokenizer::from_json(value).map_err(|e| e.to_string()))
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            state: Arc::new(RwLock::new(TokenizerState {
                inner,
                trunc: None,
                pad: None,
                post_processor_json,
            })),
        })
    }
}

#[pymethods]
impl PyTokenizer {
    /// Download `tokenizer.json` from HuggingFace Hub for the given model
    /// (e.g. `"meta-llama/Llama-3.1-8B"`) and create a tokenizer with it.
    ///
    /// (This is an alias for Tokenizer.from_model)
    #[new]
    fn new(model: &str, py: Python<'_>) -> PyResult<Self> {
        Self::from_model(model, py)
    }

    /// Create a tokenizer from a `tokenizer.json` file.
    #[staticmethod]
    fn from_file(path: &str, py: Python<'_>) -> PyResult<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| PyValueError::new_err(format!("cannot read {path}: {e}")))?;
        Self::build_from_str(&json, py)
    }

    /// Create a tokenizer from a raw JSON string for `tokenizer.json`.
    #[staticmethod]
    fn from_json_str(json: &str, py: Python<'_>) -> PyResult<Self> {
        Self::build_from_str(json, py)
    }

    /// Download `tokenizer.json` from HuggingFace Hub for the given model
    /// (e.g. `"meta-llama/Llama-3.1-8B"`) and create a tokenizer with it.
    #[staticmethod]
    fn from_model(model: &str, py: Python<'_>) -> PyResult<Self> {
        let json = py
            .allow_threads(|| {
                fastokens::Tokenizer::download_tokenizer_json(model).map_err(|e| e.to_string())
            })
            .map_err(PyValueError::new_err)?;
        Self::build_from_str(&json, py)
    }

    // ── Post-processor ────────────────────────────────────────────────

    /// The current post-processor, or ``None`` if none is configured.
    ///
    /// The returned object's ``__str__`` yields its JSON representation,
    /// so ``str(tokenizer.post_processor)`` round-trips through the setter.
    #[getter]
    fn post_processor(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.read().post_processor_json {
            None => Ok(py.None()),
            Some(json) => Py::new(py, PyPostProcessor { json: json.clone() }).map(|p| p.into_any()),
        }
    }

    /// Set the post-processor.
    ///
    /// Accepts anything whose ``str()`` yields a valid post-processor JSON —
    /// including our own ``PostProcessor`` objects and ``tokenizers.processors.*``
    /// objects from the HuggingFace tokenizers library.
    #[setter]
    fn set_post_processor(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if value.is_none() {
            let mut state = self.write();
            state.inner.set_post_processor(None);
            state.post_processor_json = None;
            return Ok(());
        }
        // `tokenizers.processors.*` objects expose `__getstate__` returning JSON
        // bytes — this is the reliable path across all tokenizers versions.
        // For our own `PyPostProcessor` (no `__getstate__`), fall back to
        // `__str__` which returns the JSON string directly.
        let json_str = if let Ok(state) = value.call_method0("__getstate__") {
            if let Ok(bytes) = state.extract::<Vec<u8>>() {
                String::from_utf8(bytes)
                    .map_err(|e| PyValueError::new_err(format!("non-UTF-8 processor state: {e}")))?
            } else {
                value.str()?.to_cow()?.to_string()
            }
        } else {
            value.str()?.to_cow()?.to_string()
        };
        self.write().update_post_processor_json(&json_str)
    }

    // ── Truncation ────────────────────────────────────────────────────

    #[pyo3(signature = (max_length, stride = 0, strategy = "longest_first", direction = "right"))]
    fn enable_truncation(&self, max_length: usize, stride: usize, strategy: &str, direction: &str) {
        self.write().trunc = Some(TruncationParams {
            max_length,
            stride,
            strategy: strategy.to_string(),
            direction: direction.to_string(),
        });
    }

    fn no_truncation(&self) {
        self.write().trunc = None;
    }

    #[getter]
    fn truncation(&self, py: Python<'_>) -> PyObject {
        match &self.read().trunc {
            None => py.None(),
            Some(t) => {
                let d = PyDict::new(py);
                d.set_item("max_length", t.max_length).unwrap();
                d.set_item("stride", t.stride).unwrap();
                d.set_item("strategy", &t.strategy).unwrap();
                d.set_item("direction", &t.direction).unwrap();
                d.into()
            }
        }
    }

    // ── Padding ───────────────────────────────────────────────────────

    #[pyo3(signature = (direction = "right", pad_id = 0, pad_type_id = 0, pad_token = "[PAD]", length = None, pad_to_multiple_of = None))]
    fn enable_padding(
        &self,
        direction: &str,
        pad_id: u32,
        pad_type_id: u32,
        pad_token: &str,
        length: Option<usize>,
        pad_to_multiple_of: Option<usize>,
    ) {
        self.write().pad = Some(PaddingParams {
            direction: direction.to_string(),
            pad_id,
            pad_type_id,
            pad_token: pad_token.to_string(),
            length,
            pad_to_multiple_of,
        });
    }

    fn no_padding(&self) {
        self.write().pad = None;
    }

    #[getter]
    fn padding(&self, py: Python<'_>) -> PyObject {
        match &self.read().pad {
            None => py.None(),
            Some(p) => {
                let d = PyDict::new(py);
                d.set_item("direction", &p.direction).unwrap();
                d.set_item("pad_id", p.pad_id).unwrap();
                d.set_item("pad_type_id", p.pad_type_id).unwrap();
                d.set_item("pad_token", &p.pad_token).unwrap();
                match p.length {
                    Some(l) => d.set_item("length", l).unwrap(),
                    None => d.set_item("length", py.None()).unwrap(),
                }
                match p.pad_to_multiple_of {
                    Some(m) => d.set_item("pad_to_multiple_of", m).unwrap(),
                    None => d.set_item("pad_to_multiple_of", py.None()).unwrap(),
                }
                d.into()
            }
        }
    }

    // ── Encoding ──────────────────────────────────────────────────────

    /// Run the full encoding pipeline.
    ///
    /// Truncation and padding configured via `enable_truncation` /
    /// `enable_padding` are applied before returning.
    #[pyo3(signature = (input, add_special_tokens = false, split_special_tokens = false))]
    fn encode(
        &self,
        input: &str,
        add_special_tokens: bool,
        split_special_tokens: bool,
        py: Python<'_>,
    ) -> PyResult<Py<PyEncoding>> {
        let encoding = py
            .allow_threads(|| {
                let state = self.read();
                let mut ids = state
                    .inner
                    .encode_with_options(input, add_special_tokens, split_special_tokens)
                    .map_err(|e| e.to_string())?;
                state.do_truncate(&mut ids);
                let target = state.single_pad_target(ids.len());
                Ok::<PyEncoding, String>(build_encoding(ids, state.pad.as_ref(), target))
            })
            .map_err(PyValueError::new_err)?;

        Py::new(py, encoding)
    }

    /// Encode text segments with per-segment control over added-token matching.
    ///
    /// Each segment is ``(text, allow_special)``. When ``allow_special`` is
    /// false, all added-token matching is bypassed for that segment, matching
    /// ``tiktoken.encode(..., disallowed_special=())`` for user/tool text.
    ///
    /// ``tiktoken_safe=True`` is the default and reproduces legacy tiktoken
    /// tokenizer chunking for token-ID parity. Pass ``False`` only when you
    /// explicitly want whole-segment BPE encoding.
    #[pyo3(signature = (segments, add_special_tokens = false, tiktoken_safe = true))]
    fn encode_segments(
        &self,
        segments: Vec<(String, bool)>,
        add_special_tokens: bool,
        tiktoken_safe: bool,
        py: Python<'_>,
    ) -> PyResult<Py<PyEncoding>> {
        let encoding = py
            .allow_threads(|| {
                let state = self.read();
                let segment_iter = segments.iter().map(|(text, allow)| (text, *allow));
                let mut ids = if tiktoken_safe {
                    state
                        .inner
                        .encode_segments_tiktoken_safe(segment_iter, add_special_tokens)
                } else {
                    state
                        .inner
                        .encode_segments(segment_iter, add_special_tokens)
                }
                .map_err(|e| e.to_string())?;
                state.do_truncate(&mut ids);
                let target = state.single_pad_target(ids.len());
                Ok::<PyEncoding, String>(build_encoding(ids, state.pad.as_ref(), target))
            })
            .map_err(PyValueError::new_err)?;

        Py::new(py, encoding)
    }

    /// Encode a rendered prompt containing structural token strings.
    ///
    /// Structural tokens are emitted as token IDs. Text spans are encoded with
    /// split-special-token behavior, after restoring placeholders to their
    /// original user text. Keep `add_special_tokens=false` when replacing an
    /// existing rendered chat-template encode path.
    #[pyo3(signature = (
        input,
        structural_config,
        placeholder_map = None,
        add_special_tokens = false
    ))]
    fn encode_with_structural_tokens(
        &self,
        input: &str,
        structural_config: PyRef<'_, PyStructuralTokenConfig>,
        placeholder_map: Option<HashMap<String, String>>,
        add_special_tokens: bool,
        py: Python<'_>,
    ) -> PyResult<Py<PyEncoding>> {
        let placeholder_map = placeholder_map.unwrap_or_default();
        let structural_config = Arc::clone(&structural_config.inner);
        let ids = py
            .allow_threads(|| {
                let state = self.read();
                state
                    .inner
                    .encode_with_structural_tokens(
                        input,
                        &structural_config,
                        &placeholder_map,
                        add_special_tokens,
                    )
                    .map_err(|e| e.to_string())
            })
            .map_err(PyValueError::new_err)?;
        let n = ids.len();

        Py::new(py, PyEncoding::make(ids, vec![1u32; n]))
    }

    /// Encode a batch of inputs in parallel.
    ///
    /// Truncation is applied per-sequence; padding (if enabled) pads the
    /// batch to a uniform length.
    #[pyo3(signature = (inputs, add_special_tokens = false, split_special_tokens = false))]
    fn encode_batch(
        &self,
        inputs: Vec<String>,
        add_special_tokens: bool,
        split_special_tokens: bool,
        py: Python<'_>,
    ) -> PyResult<Vec<Py<PyEncoding>>> {
        let encodings = py
            .allow_threads(|| {
                let state = self.read();
                state.encode_batch_encodings(&inputs, add_special_tokens, split_special_tokens)
            })
            .map_err(PyValueError::new_err)?;
        encodings
            .into_iter()
            .map(|encoding| Py::new(py, encoding))
            .collect()
    }

    /// Encode a batch of inputs in parallel and return a Python awaitable.
    ///
    /// Truncation is applied per-sequence; padding (if enabled) pads the
    /// batch to a uniform length.
    #[pyo3(signature = (inputs, add_special_tokens = false, split_special_tokens = false))]
    fn async_encode_batch<'py>(
        &self,
        py: Python<'py>,
        inputs: Vec<String>,
        add_special_tokens: bool,
        split_special_tokens: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let state = Arc::clone(&self.state);
        let rt = Arc::clone(&TOKIO_RUNTIME);

        future_into_py(py, async move {
            let encodings = rt
                .spawn_blocking(move || {
                    let state = state.read().expect("PyTokenizer state lock poisoned");
                    state.encode_batch_encodings(&inputs, add_special_tokens, split_special_tokens)
                })
                .await
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .map_err(PyValueError::new_err)?;

            Python::with_gil(|py| {
                let encodings: PyResult<Vec<Py<PyEncoding>>> = encodings
                    .into_iter()
                    .map(|encoding| Py::new(py, encoding))
                    .collect();
                Ok(encodings?.into_pyobject(py)?.unbind().into_any())
            })
        })
    }

    // ── Post-processing ───────────────────────────────────────────────

    /// Apply the post-processor to an existing encoding.
    ///
    /// When `add_special_tokens` is true the post-processor inserts special
    /// tokens (BOS/EOS/etc.).  Pair encodings are not supported.
    #[pyo3(signature = (encoding, pair = None, add_special_tokens = true))]
    fn post_process(
        &self,
        encoding: Py<PyEncoding>,
        pair: Option<Py<PyEncoding>>,
        add_special_tokens: bool,
        py: Python<'_>,
    ) -> PyResult<Py<PyEncoding>> {
        if pair.is_some() {
            return Err(PyNotImplementedError::new_err(
                "pair post-processing is not supported by fastokens",
            ));
        }
        if !add_special_tokens {
            return Ok(encoding);
        }
        let ids = encoding.borrow(py).ids.clone();
        let new_ids = self.read().inner.post_process(ids, true);
        let n = new_ids.len();
        Py::new(py, PyEncoding::make(new_ids, vec![1u32; n]))
    }

    /// Return the number of special tokens added for a single or pair sequence.
    fn num_special_tokens_to_add(&self, is_pair: bool) -> usize {
        if is_pair {
            return 0; // pair not supported
        }
        // Probe: encode empty IDs with and without special tokens.
        let with_special = self.read().inner.post_process(vec![], true);
        with_special.len()
    }

    // ── Decoding ──────────────────────────────────────────────────────

    /// Decode a list of token strings back into text using the decoder pipeline.
    ///
    /// This is what `convert_tokens_to_string` needs: token strings (e.g.
    /// "Ġhello") → decoded text (" hello").  The decoder (e.g. ByteLevel)
    /// is applied exactly as during normal `decode`.
    fn decode_tokens(&self, tokens: Vec<String>) -> PyResult<String> {
        self.read()
            .inner
            .decode_tokens(tokens)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Decode token IDs back into text.
    #[pyo3(signature = (ids, skip_special_tokens = false))]
    fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        self.read()
            .inner
            .decode(&ids, skip_special_tokens)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Decode a batch of token ID sequences.
    #[pyo3(signature = (sentences, skip_special_tokens = false))]
    fn decode_batch(
        &self,
        sentences: Vec<Vec<u32>>,
        skip_special_tokens: bool,
    ) -> PyResult<Vec<String>> {
        let state = self.read();
        let refs: Vec<&[u32]> = sentences.iter().map(Vec::as_slice).collect();
        state
            .inner
            .decode_batch(&refs, skip_special_tokens)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Decode a batch of token ID sequences and return a Python awaitable.
    #[pyo3(signature = (sentences, skip_special_tokens = false))]
    fn async_decode_batch<'py>(
        &self,
        py: Python<'py>,
        sentences: Vec<Vec<u32>>,
        skip_special_tokens: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let state = Arc::clone(&self.state);
        let rt = Arc::clone(&TOKIO_RUNTIME);

        future_into_py(py, async move {
            let decoded = rt
                .spawn_blocking(move || {
                    let state = state.read().expect("PyTokenizer state lock poisoned");
                    let refs: Vec<&[u32]> = sentences.iter().map(Vec::as_slice).collect();
                    state
                        .inner
                        .decode_batch(&refs, skip_special_tokens)
                        .map_err(|e| e.to_string())
                })
                .await
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .map_err(PyValueError::new_err)?;

            Python::with_gil(|py| Ok(decoded.into_pyobject(py)?.unbind().into_any()))
        })
    }

    // ── Vocabulary ────────────────────────────────────────────────────

    /// Look up the token ID for a string.
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.read().inner.token_to_id(token)
    }

    /// Look up the string for a token ID.
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.read().inner.id_to_token(id).map(String::from)
    }

    /// Return the vocabulary size.
    #[getter]
    fn vocab_size(&self) -> usize {
        self.read().inner.vocab_size()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `PyEncoding::pad` correctly fills `type_ids` with `pad_type_id` for
    /// padded positions.  This is the expected behaviour.
    #[test]
    fn encoding_pad_applies_pad_type_id() {
        let mut enc = PyEncoding::new(vec![10u32, 20, 30], None);
        // 3 real tokens → pad to length 5 with pad_type_id = 1
        enc.pad(5, "right", 0u32, 1u32, "[PAD]");

        assert_eq!(enc.ids, vec![10u32, 20, 30, 0, 0]);
        assert_eq!(enc.attention_mask, vec![1u32, 1, 1, 0, 0]);
        assert_eq!(
            enc.type_ids,
            vec![0u32, 0, 0, 1, 1],
            "padded positions should carry pad_type_id=1 in type_ids"
        );
    }

    /// The tokenizer encode paths build returned encodings through the same
    /// padding owner as `PyEncoding::pad`, preserving `pad_type_id` metadata.
    #[test]
    fn encode_batch_pad_type_id_applied_to_type_ids() {
        let pad = PaddingParams {
            direction: "right".to_string(),
            pad_id: 0,
            pad_type_id: 1,
            pad_token: "[PAD]".to_string(),
            length: None,
            pad_to_multiple_of: None,
        };
        let enc = build_encoding(vec![10u32, 20, 30], Some(&pad), 5);

        assert_eq!(enc.ids, vec![10u32, 20, 30, 0, 0]);
        assert_eq!(enc.attention_mask, vec![1u32, 1, 1, 0, 0]);
        assert_eq!(enc.type_ids, vec![0u32, 0, 0, 1, 1]);
    }

    #[test]
    fn build_encoding_left_padding_applies_pad_type_id() {
        let pad = PaddingParams {
            direction: "left".to_string(),
            pad_id: 0,
            pad_type_id: 7,
            pad_token: "[PAD]".to_string(),
            length: None,
            pad_to_multiple_of: None,
        };
        let enc = build_encoding(vec![10u32, 20, 30], Some(&pad), 5);

        assert_eq!(enc.ids, vec![0u32, 0, 10, 20, 30]);
        assert_eq!(enc.attention_mask, vec![0u32, 0, 1, 1, 1]);
        assert_eq!(enc.type_ids, vec![7u32, 7, 0, 0, 0]);
    }
}

// ---------------------------------------------------------------------------
// DecodeStream
// ---------------------------------------------------------------------------

/// Python binding for [`fastokens::DecodeStream`].
///
/// Drop-in replacement for `tokenizers.decoders.DecodeStream`. Accepts both a
/// bare `fastokens.Tokenizer` and any shim that stores one in `._fast`
/// (e.g. `_TokenizerShim`).
#[pyclass(name = "DecodeStream")]
struct PyDecodeStream {
    inner: fastokens::DecodeStream,
}

#[pymethods]
impl PyDecodeStream {
    #[new]
    #[pyo3(signature = (ids = None, skip_special_tokens = false))]
    fn new(ids: Option<Vec<u32>>, skip_special_tokens: bool) -> Self {
        Self {
            inner: fastokens::DecodeStream::new(ids.unwrap_or_default(), skip_special_tokens),
        }
    }

    #[pyo3(signature = (tokenizer, id))]
    fn step(
        &mut self,
        tokenizer: &Bound<'_, PyAny>,
        id: &Bound<'_, PyAny>,
        py: Python<'_>,
    ) -> PyResult<Option<String>> {
        let new_ids: Vec<u32> = if let Ok(single) = id.extract::<u32>() {
            vec![single]
        } else {
            id.extract::<Vec<u32>>()?
        };

        // Accept a PyTokenizer directly or any shim that stores one in ._fast.
        let py_tok: Py<PyTokenizer> = tokenizer
            .extract::<Py<PyTokenizer>>()
            .or_else(|_| tokenizer.getattr("_fast")?.extract::<Py<PyTokenizer>>())?;

        let tok = py_tok.borrow(py);
        let state = tok.read();
        self.inner
            .step(&state.inner, new_ids)
            .map_err(PyValueError::new_err)
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEncoding>()?;
    m.add_class::<PyPostProcessor>()?;
    m.add_class::<PyStructuralTokenConfig>()?;
    m.add_class::<PyTokenizer>()?;
    m.add_class::<PyDecodeStream>()?;
    Ok(())
}
