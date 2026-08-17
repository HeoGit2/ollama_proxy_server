import json

from app.core.vllm_translator import (
    CHAIN_OF_THOUGHT_PROMPT,
    translate_ollama_to_vllm_chat,
    translate_ollama_to_vllm_embeddings,
    translate_vllm_to_ollama_embeddings,
    vllm_stream_to_ollama_stream,
)


async def _agen(chunks):
    for chunk in chunks:
        yield chunk


async def _collect(chunks, model_name="llama3"):
    return [
        json.loads(raw.decode("utf-8"))
        async for raw in vllm_stream_to_ollama_stream(_agen(chunks), model_name)
    ]


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n"


def _delta(content=None, tool_arguments=None, finish_reason=None, created=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if tool_arguments is not None:
        delta["tool_calls"] = [{"function": {"arguments": tool_arguments}}]
    payload = {"choices": [{"delta": delta, "finish_reason": finish_reason}]}
    if created is not None:
        payload["created"] = created
    return _sse(payload)


class TestChatRequestTranslation:
    def test_maps_model_and_messages(self):
        payload = translate_ollama_to_vllm_chat(
            {"model": "llama3", "messages": [{"role": "user", "content": "hi"}]}
        )

        assert payload == {
            "model": "llama3",
            "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
        }

    def test_preserves_stream_flag(self):
        assert translate_ollama_to_vllm_chat({"stream": True})["stream"] is True

    def test_handles_missing_fields(self):
        assert translate_ollama_to_vllm_chat({}) == {
            "model": None,
            "stream": False,
            "messages": [],
        }

    def test_think_inserts_system_prompt_when_absent(self):
        payload = translate_ollama_to_vllm_chat(
            {"messages": [{"role": "user", "content": "hi"}], "think": True}
        )

        assert payload["messages"][0] == {
            "role": "system",
            "content": CHAIN_OF_THOUGHT_PROMPT,
        }
        assert payload["messages"][1]["role"] == "user"

    def test_think_prepends_to_existing_system_prompt(self):
        payload = translate_ollama_to_vllm_chat(
            {
                "messages": [
                    {"role": "system", "content": "Be terse."},
                    {"role": "user", "content": "hi"},
                ],
                "think": True,
            }
        )

        assert len(payload["messages"]) == 2
        assert (
            payload["messages"][0]["content"]
            == f"{CHAIN_OF_THOUGHT_PROMPT}\n\nBe terse."
        )

    def test_think_only_applies_when_exactly_true(self):
        payload = translate_ollama_to_vllm_chat(
            {"messages": [{"role": "user", "content": "hi"}], "think": "yes"}
        )
        assert payload["messages"] == [{"role": "user", "content": "hi"}]

    def test_images_become_openai_content_parts(self):
        payload = translate_ollama_to_vllm_chat(
            {
                "messages": [
                    {"role": "user", "content": "what is this?", "images": ["QUJD"]}
                ]
            }
        )

        message = payload["messages"][0]
        assert "images" not in message
        assert message["content"] == [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
        ]

    def test_images_without_text_content_are_dropped(self):
        payload = translate_ollama_to_vllm_chat(
            {"messages": [{"role": "user", "content": "", "images": ["QUJD"]}]}
        )

        assert payload["messages"][0] == {"role": "user", "content": ""}


class TestEmbeddingTranslation:
    def test_request_translation(self):
        assert translate_ollama_to_vllm_embeddings(
            {"model": "nomic-embed-text", "prompt": "hello"}
        ) == {"model": "nomic-embed-text", "input": "hello"}

    def test_request_translation_with_missing_fields(self):
        assert translate_ollama_to_vllm_embeddings({}) == {"model": None, "input": None}

    def test_response_translation_takes_first_embedding(self):
        vllm_payload = {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [9.9]}]}
        assert translate_vllm_to_ollama_embeddings(vllm_payload) == {
            "embedding": [0.1, 0.2]
        }

    def test_response_translation_with_empty_data(self):
        assert translate_vllm_to_ollama_embeddings({}) == {"embedding": []}


class TestStreamTranslation:
    async def test_content_chunks_are_translated(self):
        chunks = await _collect([_delta("Hel"), _delta("lo"), "data: [DONE]\n"])

        assert [c["message"]["content"] for c in chunks] == ["Hel", "lo", ""]
        assert [c["done"] for c in chunks] == [False, False, True]
        assert all(c["model"] == "llama3" for c in chunks)

    async def test_done_chunk_reports_eval_metrics(self):
        chunks = await _collect([_delta("12345678"), "data: [DONE]\n"])

        done = chunks[-1]
        assert done["done"] is True
        assert done["eval_count"] == 2
        assert done["eval_duration"] >= 0
        assert done["created_at"].endswith("Z")

    async def test_created_timestamp_is_converted_to_iso_utc(self):
        chunks = await _collect([_delta("hi", created=0)])
        assert chunks[0]["created_at"] == "1970-01-01T00:00:00Z"

    async def test_stops_after_done_marker(self):
        chunks = await _collect(["data: [DONE]\n", _delta("ignored")])
        assert len(chunks) == 1
        assert chunks[0]["done"] is True

    async def test_partial_lines_are_buffered_across_chunks(self):
        raw = _delta("split")
        chunks = await _collect([raw[:12], raw[12:], "data: [DONE]\n"])
        assert chunks[0]["message"]["content"] == "split"

    async def test_trailing_done_without_newline_is_processed(self):
        chunks = await _collect([_delta("hi"), "data: [DONE]"])
        assert chunks[-1]["done"] is True

    async def test_non_data_and_blank_lines_are_ignored(self):
        chunks = await _collect(["\n", ": ping\n", _delta("hi"), "data: [DONE]\n"])
        assert len(chunks) == 2

    async def test_malformed_json_is_skipped(self):
        chunks = await _collect(["data: {not json}\n", _delta("hi"), "data: [DONE]\n"])
        assert chunks[0]["message"]["content"] == "hi"

    async def test_tool_call_is_wrapped_in_think_tags(self):
        arguments = json.dumps({"steps": ["step one", "step two"]})
        chunks = await _collect(
            [
                _delta(tool_arguments=arguments[:10]),
                _delta(tool_arguments=arguments[10:]),
                _delta(finish_reason="tool_calls"),
                _delta("final answer"),
                "data: [DONE]\n",
            ]
        )

        contents = [c["message"]["content"] for c in chunks]
        assert contents == [
            "<think>",
            "step one\nstep two",
            "</think>",
            "final answer",
            "",
        ]

    async def test_tool_call_and_content_in_same_chunk_are_both_emitted(self):
        payload = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"function": {"arguments": json.dumps({"steps": ["s"]})}}
                        ],
                        "content": "answer",
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        chunks = await _collect([_sse(payload), "data: [DONE]\n"])

        assert [c["message"]["content"] for c in chunks] == [
            "<think>",
            "s",
            "</think>",
            "answer",
            "",
        ]

    async def test_non_list_steps_are_stringified(self):
        arguments = json.dumps({"steps": "just one thought"})
        chunks = await _collect(
            [
                _delta(tool_arguments=arguments),
                _delta(finish_reason="tool_calls"),
                "data: [DONE]\n",
            ]
        )

        assert [c["message"]["content"] for c in chunks] == [
            "<think>",
            "just one thought",
            "</think>",
            "",
        ]

    async def test_unparsable_tool_arguments_emit_no_thought(self):
        chunks = await _collect(
            [
                _delta(tool_arguments="{broken"),
                _delta(finish_reason="tool_calls"),
                _delta("answer"),
                "data: [DONE]\n",
            ]
        )

        assert [c["message"]["content"] for c in chunks] == ["<think>", "answer", ""]
