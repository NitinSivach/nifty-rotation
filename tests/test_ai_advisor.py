import pytest

from ai_advisor import call_advisor, resolve_api_key, stream_chat


class FakeResponse:
    def __init__(self, lines, status_code=200, text=""):
        self.status_code = status_code
        self._lines = list(lines)
        self.text = text
        self.closed = False

    def iter_lines(self, decode_unicode=False):
        for line in self._lines:
            if isinstance(line, str):
                yield line.encode("utf-8") if not decode_unicode else line
            else:
                yield line.decode("utf-8") if decode_unicode else line

    def close(self):
        self.closed = True


def _fake_post(response):
    return lambda *args, **kwargs: response


def test_stream_chat_parses_sse_events(monkeypatch):
    events = [
        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        "data: [DONE]",
    ]
    monkeypatch.setattr("ai_advisor.requests.post", _fake_post(FakeResponse(events)))
    out = "".join(stream_chat("https://example.com/v1", "key", "model", [{"role": "user", "content": "hi"}]))
    assert out == "Hello"


def test_stream_chat_skips_malformed_lines(monkeypatch):
    events = [
        "data: not-json",
        'data: {"choices":[{"delta":{"content":"ok"}}]}',
        "data: [DONE]",
    ]
    monkeypatch.setattr("ai_advisor.requests.post", _fake_post(FakeResponse(events)))
    out = "".join(stream_chat("https://example.com/v1", "", "model", []))
    assert out == "ok"


def test_stream_chat_preserves_utf8_unicode(monkeypatch):
    content = "Hit rate (58.8%) \u2013 In 10 of 17 historical 6\u2011month windows"
    events = [
        f'data: {{"choices":[{{"delta":{{"content":"{content}"}}}}]}}'.encode("utf-8"),
        b"data: [DONE]",
    ]
    monkeypatch.setattr("ai_advisor.requests.post", _fake_post(FakeResponse(events)))
    out = "".join(stream_chat("https://example.com/v1", "key", "model", []))
    assert out == content
    assert "â" not in out
    assert "€" not in out


def test_stream_chat_raises_on_http_error(monkeypatch):
    resp = FakeResponse([], status_code=429, text="rate limited")
    monkeypatch.setattr("ai_advisor.requests.post", _fake_post(resp))
    with pytest.raises(RuntimeError, match="429"):
        list(stream_chat("https://example.com/v1", "key", "model", []))
    assert resp.closed


def test_call_advisor_falls_back_to_next_provider(monkeypatch):
    def fake_stream(base_url, api_key, model, messages):
        if "openrouter" in base_url:
            raise RuntimeError("boom")
        yield "OK"

    monkeypatch.setattr("ai_advisor.stream_chat", fake_stream)
    out = "".join(
        call_advisor(
            [{"role": "user", "content": "hi"}],
            provider="Auto",
            extra_keys={"OpenRouter": "key"},
        )
    )
    assert out == "OK"


def test_call_advisor_raises_when_all_providers_fail(monkeypatch):
    monkeypatch.setattr("ai_advisor.stream_chat", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="All AI providers failed"):
        list(call_advisor([{"role": "user", "content": "hi"}], provider="Auto"))


def test_call_advisor_unknown_provider_raises():
    with pytest.raises(RuntimeError, match="unknown provider"):
        list(call_advisor([], provider="NotAProvider", extra_keys={}))


def test_resolve_api_key_prefers_extra_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "envkey")
    assert resolve_api_key("Groq") == "envkey"
    assert resolve_api_key("Groq", {"Groq": "explicit"}) == "explicit"


def test_resolve_api_key_empty_when_not_configured(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert resolve_api_key("Groq", {}) == ""
