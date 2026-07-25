from pathlib import Path

import pytest

from fastokens._compat import _TokenizerShim
from fastokens._native import Tokenizer
from test_async_stub import TOKENIZER_JSON

REPO_ROOT = Path(__file__).resolve().parents[2]

CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{%- if message['role'] == 'user' -%}"
    "user: {{ message['content'] }}"
    "{%- elif message['role'] == 'assistant' -%}"
    "assistant: {{ message.get('content') }}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}assistant: {%- endif -%}"
)


def test_native_apply_chat_template_renders_string() -> None:
    tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        chat_template=CHAT_TEMPLATE,
        tokenize=False,
        add_generation_prompt=True,
    )

    assert rendered == "user: helloassistant:"


def test_native_apply_chat_template_rejects_tokenize_true() -> None:
    tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)

    with pytest.raises(NotImplementedError, match="tokenize=True"):
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            chat_template="{{ messages[0]['content'] }}",
            tokenize=True,
        )


def test_native_apply_chat_template_consumes_render_only_tokenizer_kwargs() -> None:
    tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)
    template = (
        "{% if return_dict is defined %}return_dict leaked{% endif %}"
        "{% if tokenizer_kwargs is defined %}tokenizer_kwargs leaked{% endif %}"
        "{% if padding is defined %}padding leaked{% endif %}"
        "{{ messages[0]['content'] }}"
    )

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        chat_template=template,
        tokenize=False,
        return_dict=False,
        tokenizer_kwargs={"add_prefix_space": True},
        padding=False,
        truncation=False,
        max_length=128,
        return_tensors=None,
    )

    assert rendered == "hello"


def test_native_apply_chat_template_omits_null_special_tokens() -> None:
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    template = (
        "{% if pad_token is defined %}defined{% else %}undefined{% endif %}"
        "{{ pad_token }}"
    )
    messages = [{"role": "user", "content": "hello"}]

    hf_tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizers.Tokenizer.from_str(TOKENIZER_JSON),
        bos_token="<s>",
        eos_token="</s>",
    )
    fast_tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)

    expected = hf_tokenizer.apply_chat_template(
        messages,
        chat_template=template,
        tokenize=False,
    )
    rendered = fast_tokenizer.apply_chat_template(
        messages,
        chat_template=template,
        tokenize=False,
    )

    assert rendered == expected == "undefined"


def test_native_apply_chat_template_uses_persistent_special_tokens() -> None:
    tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)
    tokenizer.set_special_tokens({"bos_token": "<s>", "eos_token": "</s>"})

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        chat_template="{{ bos_token }}{{ messages[0]['content'] }}{{ eos_token }}",
        tokenize=False,
    )

    assert rendered == "<s>hello</s>"


def test_native_apply_chat_template_rejects_reserved_kwargs() -> None:
    tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)

    with pytest.raises(TypeError, match="multiple values"):
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            chat_template="{{ messages }}",
            tokenize=False,
            messages="shadow",
        )


def test_native_apply_chat_template_rejects_assistant_mask_without_tokenize() -> None:
    tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)

    with pytest.raises(ValueError, match="return_assistant_tokens_mask=True"):
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            chat_template="{{ messages[0]['content'] }}",
            tokenize=False,
            return_assistant_tokens_mask=True,
        )


def test_native_apply_chat_template_rejects_batched_conversations() -> None:
    tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)

    with pytest.raises(NotImplementedError, match="batched conversations"):
        tokenizer.apply_chat_template(
            [[{"role": "user", "content": "hello"}]],
            chat_template="{{ messages[0]['content'] }}",
            tokenize=False,
        )


def test_native_apply_chat_template_supports_kwargs_and_special_tokens() -> None:
    tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)
    template = (
        "{{ bos_token }}"
        "{{ messages[0].content.upper() }}"
        "{{ payload | tojson }}"
        "{{ eos_token }}"
    )

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        chat_template=template,
        tokenize=False,
        payload={"text": "<tag>&'"},
        special_tokens={"bos_token": "<s>", "eos_token": "</s>"},
    )

    assert rendered == r"""<s>HELLO{"text": "<tag>&'"}</s>"""


def test_native_apply_chat_template_rejects_continue_and_generation_prompt() -> None:
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    template = "{{ messages[0]['content'] }}"
    messages = [{"role": "assistant", "content": "partial"}]
    hf_tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizers.Tokenizer.from_str(TOKENIZER_JSON),
    )
    fast_tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)

    with pytest.raises(ValueError, match="continue_final_message"):
        hf_tokenizer.apply_chat_template(
            messages,
            chat_template=template,
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=True,
        )
    with pytest.raises(ValueError, match="continue_final_message"):
        fast_tokenizer.apply_chat_template(
            messages,
            chat_template=template,
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=True,
        )


def test_native_apply_chat_template_strips_generation_markers() -> None:
    tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)

    rendered = tokenizer.apply_chat_template(
        [],
        chat_template="a{% generation %}b{% endgeneration %}c",
        tokenize=False,
    )

    assert rendered == "abc"


def test_native_apply_chat_template_supports_generation_whitespace_control() -> None:
    tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)

    rendered = tokenizer.apply_chat_template(
        [],
        chat_template="a\n{%- generation -%}\nb\n{%- endgeneration -%}\nc",
        tokenize=False,
    )

    assert rendered == "abc"


def test_native_apply_chat_template_rejects_invalid_strftime_without_panic() -> None:
    tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)

    with pytest.raises(ValueError, match="invalid strftime_now format"):
        tokenizer.apply_chat_template(
            [],
            chat_template="{{ strftime_now('%Q %#z') }}",
            tokenize=False,
        )


def test_continue_final_message_error_truncates_rendered_prompt() -> None:
    tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)
    message = "x" * 10_000

    with pytest.raises(ValueError) as exc_info:
        tokenizer.apply_chat_template(
            [{"role": "assistant", "content": message}],
            chat_template="{{ messages[0]['content'] | replace('x', 'y') }}",
            tokenize=False,
            continue_final_message=True,
        )

    error = str(exc_info.value)
    assert len(error) < 5_000
    assert "..." in error


@pytest.mark.parametrize(
    ("fixture", "kwargs"),
    [
        ("glm-5.2", {"enable_thinking": False}),
        ("gpt-oss-120b", {"builtin_tools": []}),
    ],
)
def test_vendored_chat_templates_match_transformers(fixture: str, kwargs: dict) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    template = (REPO_ROOT / "vendored_tokenizers" / fixture / "chat_template.jinja").read_text()
    messages = [{"role": "user", "content": "hello"}]

    hf_tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizers.Tokenizer.from_str(TOKENIZER_JSON),
        bos_token="<s>",
        eos_token="</s>",
    )
    fast_tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)

    expected = hf_tokenizer.apply_chat_template(
        messages,
        chat_template=template,
        tokenize=False,
        **kwargs,
    )
    rendered = fast_tokenizer.apply_chat_template(
        messages,
        chat_template=template,
        tokenize=False,
        special_tokens={"bos_token": "<s>", "eos_token": "</s>"},
        **kwargs,
    )

    assert rendered == expected


def test_kimi_25_template_matches_transformers_with_tools_and_thinking() -> None:
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    template = (
        REPO_ROOT / "vendored_tokenizers" / "kimi-k2.5" / "chat_template.jinja"
    ).read_text()
    messages = [
        {"role": "system", "content": "Be terse."},
        {
            "role": "user",
            "name": "alice",
            "content": [
                {"type": "text", "text": "look this up"},
                {"type": "image_url", "image_url": {"url": "https://example.test/i.png"}},
            ],
        },
        {
            "role": "assistant",
            "content": "I will check.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "search",
                        "arguments": {"query": "rust tokenizer", "top_k": 2},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result text"},
        {
            "role": "assistant",
            "content": "final answer",
            "reasoning_content": "private reasoning",
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search docs",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]

    hf_tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizers.Tokenizer.from_str(TOKENIZER_JSON),
        bos_token="<s>",
        eos_token="</s>",
    )
    fast_tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)

    expected = hf_tokenizer.apply_chat_template(
        messages,
        tools=tools,
        chat_template=template,
        tokenize=False,
        add_generation_prompt=True,
        thinking=False,
        tools_ts_str="",
    )
    rendered = fast_tokenizer.apply_chat_template(
        messages,
        tools=tools,
        chat_template=template,
        tokenize=False,
        add_generation_prompt=True,
        thinking=False,
        tools_ts_str="",
        special_tokens={"bos_token": "<s>", "eos_token": "</s>"},
    )

    assert rendered == expected


@pytest.mark.parametrize(
    ("messages", "continue_final_message"),
    [
        ([{"role": "assistant", "content": "partial answer"}], True),
        (
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.test/i.png"}},
                        {"type": "text", "text": "partial answer"},
                    ],
                }
            ],
            "content",
        ),
    ],
)
def test_continue_final_message_matches_transformers(
    messages: list[dict], continue_final_message: bool | str
) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    template = (
        "{%- for message in messages -%}"
        "{%- if message.content is string -%}"
        "{{ message.content }}{{ eos_token }}"
        "{%- else -%}"
        "{%- for content in message.content -%}"
        "{%- if content.text is defined -%}{{ content.text }}{{ eos_token }}{%- endif -%}"
        "{%- endfor -%}"
        "{%- endif -%}"
        "{%- endfor -%}"
    )

    hf_tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizers.Tokenizer.from_str(TOKENIZER_JSON),
        bos_token="<s>",
        eos_token="</s>",
    )
    fast_tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)

    expected = hf_tokenizer.apply_chat_template(
        messages,
        chat_template=template,
        tokenize=False,
        continue_final_message=continue_final_message,
    )
    rendered = fast_tokenizer.apply_chat_template(
        messages,
        chat_template=template,
        tokenize=False,
        continue_final_message=continue_final_message,
        special_tokens={"bos_token": "<s>", "eos_token": "</s>"},
    )

    assert rendered == expected == "partial answer"


def test_strict_template_rejects_unknown_kwargs_with_suggestion() -> None:
    tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)
    template = "{% if enable_thinking %}think: {% endif %}{{ messages[0]['content'] }}"

    with pytest.raises(TypeError, match=r"enable_thinkng.*Did you mean `enable_thinking`"):
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            chat_template=template,
            tokenize=False,
            enable_thinkng=True,
        )


def test_strict_template_rejects_unknown_kwargs_on_cached_renderer() -> None:
    tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)
    tokenizer.set_chat_template("{{ messages[0]['content'] }}")

    with pytest.raises(TypeError, match="unexpected keyword argument `enable_thinking`"):
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            tokenize=False,
            enable_thinking=False,
        )


def test_strict_template_opt_out_allows_unused_kwargs() -> None:
    tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        chat_template="{{ messages[0]['content'] }}",
        tokenize=False,
        enable_thinkng=True,
        fastokens_strict_template=False,
    )

    assert rendered == "hello"


def test_chat_template_variables_property() -> None:
    tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)
    assert tokenizer.chat_template_variables is None

    tokenizer.set_chat_template("{% if enable_thinking %}{{ messages }}{% endif %}")

    assert tokenizer.chat_template_variables == ["enable_thinking", "messages"]


def test_shim_apply_chat_template_forwards_to_native() -> None:
    tokenizer = _TokenizerShim.from_str(TOKENIZER_JSON)

    rendered = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": "world"}],
        chat_template=CHAT_TEMPLATE,
    )

    assert rendered == "assistant: world"


def test_async_apply_chat_template_matches_sync() -> None:
    import asyncio

    tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)
    messages = [{"role": "user", "content": "hello"}]
    sync_rendered = tokenizer.apply_chat_template(
        messages, chat_template=CHAT_TEMPLATE, tokenize=False, add_generation_prompt=True
    )

    async def render() -> str:
        return await tokenizer.async_apply_chat_template(
            messages, chat_template=CHAT_TEMPLATE, tokenize=False, add_generation_prompt=True
        )

    assert asyncio.run(render()) == sync_rendered == "user: helloassistant:"


def test_async_apply_chat_template_propagates_errors() -> None:
    import asyncio

    tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)

    async def render_tokenize_true() -> str:
        return await tokenizer.async_apply_chat_template(
            [], chat_template="x", tokenize=True
        )

    with pytest.raises(NotImplementedError, match="tokenize=True"):
        asyncio.run(render_tokenize_true())

    async def render_unknown_kwarg() -> str:
        return await tokenizer.async_apply_chat_template(
            [], chat_template="{{ messages }}", tokenize=False, bogus_kwarg=1
        )

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        asyncio.run(render_unknown_kwarg())


def test_shim_async_apply_chat_template_forwards_to_native() -> None:
    import asyncio

    tokenizer = _TokenizerShim.from_str(TOKENIZER_JSON)
    tokenizer.set_chat_template(CHAT_TEMPLATE)

    async def render() -> str:
        return await tokenizer.async_apply_chat_template(
            [{"role": "assistant", "content": "world"}]
        )

    assert asyncio.run(render()) == "assistant: world"


def test_shim_preserves_special_tokens_across_copy_and_pickle() -> None:
    import copy
    import pickle

    tokenizer = _TokenizerShim.from_str(TOKENIZER_JSON)
    tokenizer.set_chat_template("{{ bos_token }}x")
    tokenizer.set_special_tokens({"bos_token": "<s>"})
    assert tokenizer.apply_chat_template([], tokenize=False) == "<s>x"

    clones = {
        "deepcopy": copy.deepcopy(tokenizer),
        "pickle": pickle.loads(pickle.dumps(tokenizer)),
        "from_shim": _TokenizerShim(tokenizer),
    }
    for name, clone in clones.items():
        assert clone.apply_chat_template([], tokenize=False) == "<s>x", name
