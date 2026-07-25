from typing import Any, Optional

class Encoding:
    """Rust-backed encoding returned by ``Tokenizer.encode`` / ``encode_batch``."""

    ids: list[int]
    attention_mask: list[int]
    type_ids: list[int]
    special_tokens_mask: list[int]
    n_sequences: int
    overflowing: list["Encoding"]

    def __new__(
        cls,
        ids: list[int],
        attention_mask: Optional[list[int]] = None,
    ) -> "Encoding": ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def into_numpy(
        self,
        ids: bool = True,
        attention_mask: bool = False,
        type_ids: bool = False,
        special_tokens_mask: bool = False,
    ) -> dict[str, Any]:
        """Move selected fields into numpy.uint32 arrays and drain the encoding."""
        ...

    # Properties that raise NotImplementedError
    @property
    def tokens(self) -> list[str]: ...
    @tokens.setter
    def tokens(self, value: list[str]) -> None: ...
    @property
    def offsets(self) -> list[tuple[int, int]]: ...
    @offsets.setter
    def offsets(self, value: list[tuple[int, int]]) -> None: ...
    @property
    def sequence_ids(self) -> list[Optional[int]]: ...
    @sequence_ids.setter
    def sequence_ids(self, value: list[Optional[int]]) -> None: ...
    @property
    def word_ids(self) -> list[Optional[int]]: ...
    @word_ids.setter
    def word_ids(self, value: list[Optional[int]]) -> None: ...
    @property
    def words(self) -> list[Optional[int]]: ...
    @words.setter
    def words(self, value: list[Optional[int]]) -> None: ...
    def set_sequence_id(self, sequence_id: int) -> None: ...
    def char_to_token(
        self, char_pos: int, sequence_index: int = 0
    ) -> Optional[int]: ...
    def char_to_word(self, char_pos: int, sequence_index: int = 0) -> Optional[int]: ...
    def token_to_chars(self, token_index: int) -> Optional[tuple[int, int]]: ...
    def token_to_sequence(self, token_index: int) -> Optional[int]: ...
    def token_to_word(self, token_index: int) -> Optional[int]: ...
    def word_to_chars(
        self, word_index: int, sequence_index: int = 0
    ) -> Optional[tuple[int, int]]: ...
    def word_to_tokens(
        self, word_index: int, sequence_index: int = 0
    ) -> Optional[tuple[int, int]]: ...
    def truncate(
        self, max_length: int, stride: int = 0, direction: str = "right"
    ) -> None: ...
    def pad(
        self,
        length: int,
        direction: str = "right",
        pad_id: int = 0,
        pad_type_id: int = 0,
        pad_token: str = "[PAD]",
    ) -> None: ...
    @staticmethod
    def merge(
        encodings: list["Encoding"], growing_offsets: bool = True
    ) -> "Encoding": ...

class StructuralTokenConfig:
    """Constant structural-token state for rendered-prompt encoding.

    Include every token string the rendered template may use as a structural
    boundary, including tag-like non-special added tokens.
    """

    def __new__(
        cls,
        structural_tokens: list[str] | set[str] | tuple[str, ...],
        non_special_added_tokens: Optional[
            set[str] | list[str] | tuple[str, ...]
        ] = None,
    ) -> "StructuralTokenConfig": ...

class Tokenizer:
    """An LLM tokenizer backed by ``tokenizer.json``."""

    def __new__(cls, model: str) -> "Tokenizer": ...
    @staticmethod
    def from_file(path: str) -> "Tokenizer": ...
    @staticmethod
    def from_json_str(json: str) -> "Tokenizer": ...
    @staticmethod
    def from_model(model: str) -> "Tokenizer": ...
    @property
    def vocab_size(self) -> int: ...
    @property
    def truncation(self) -> Optional[dict]: ...
    @property
    def padding(self) -> Optional[dict]: ...
    def enable_truncation(
        self,
        max_length: int,
        stride: int = 0,
        strategy: str = "longest_first",
        direction: str = "right",
    ) -> None: ...
    def no_truncation(self) -> None: ...
    def enable_padding(
        self,
        direction: str = "right",
        pad_id: int = 0,
        pad_type_id: int = 0,
        pad_token: str = "[PAD]",
        length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None: ...
    def no_padding(self) -> None: ...
    def post_process(
        self,
        encoding: Encoding,
        pair: Optional[Encoding] = None,
        add_special_tokens: bool = True,
    ) -> Encoding: ...
    def num_special_tokens_to_add(self, is_pair: bool) -> int: ...
    def encode(
        self,
        input: str,
        add_special_tokens: bool = False,
        split_special_tokens: bool = False,
    ) -> Encoding: ...
    def encode_segments(
        self,
        segments: list[tuple[str, bool]],
        add_special_tokens: bool = False,
        tiktoken_safe: bool = True,
    ) -> Encoding:
        """Encode ``(text, allow_special)`` segments in one backend call.

        ``tiktoken_safe=True`` is the default and reproduces legacy tiktoken
        tokenizer chunking for token-ID parity. Pass ``False`` only when you
        explicitly want whole-segment BPE encoding.
        """
        ...

    def encode_batch(
        self,
        inputs: list[str],
        add_special_tokens: bool = False,
        split_special_tokens: bool = False,
    ) -> list[Encoding]: ...
    def encode_with_structural_tokens(
        self,
        input: str,
        structural_config: StructuralTokenConfig,
        placeholder_map: Optional[dict[str, str]] = None,
        add_special_tokens: bool = False,
    ) -> Encoding:
        """Encode rendered template text with structural-token boundaries.

        ``placeholder_map`` is per request and maps placeholder text to the
        original user text. Keep ``add_special_tokens=False`` when replacing an
        existing rendered chat-template encode path. Backend truncation and
        padding settings are intentionally not applied.
        """
        ...

    async def async_encode_batch(
        self,
        inputs: list[str],
        add_special_tokens: bool = False,
        split_special_tokens: bool = False,
    ) -> list[Encoding]: ...
    def decode_tokens(self, tokens: list[str]) -> str: ...
    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str: ...
    def decode_batch(
        self, sentences: list[list[int]], skip_special_tokens: bool = False
    ) -> list[str]: ...
    async def async_decode_batch(
        self, sentences: list[list[int]], skip_special_tokens: bool = False
    ) -> list[str]: ...
    def token_to_id(self, token: str) -> Optional[int]: ...
    def id_to_token(self, id: int) -> Optional[str]: ...
