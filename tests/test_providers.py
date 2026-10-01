import json

import pytest

from chgk_team.agents import PLAYERS
from chgk_team.config import Settings
from chgk_team.domain import AgentReply, AnswerDraft
from chgk_team.providers import ContextLimitError, DemoProvider, GigaChatProvider, ModelRequest


def request(**overrides):
    context = {"question": "Опыт?", "utterances": [], "hypotheses": [], "sources": []}
    return ModelRequest(PLAYERS["captain"], {**context, **overrides})


@pytest.mark.parametrize("schema", [AgentReply, AnswerDraft])
def test_schemas_bind_with_installed_gigachat_wrapper(schema):
    from langchain_gigachat import GigaChat

    # Exercise the real schema converter offline: GigaChat requires a description.
    client = GigaChat(model="GigaChat-2-Max")
    assert client.with_structured_output(schema, method="function_calling") is not None


async def test_async_gigachat_configuration_and_prompt(monkeypatch):
    seen = {}

    class Client:
        def __init__(self, **kwargs):
            seen["config"] = kwargs

        def with_structured_output(self, schema, *, method):
            seen["schema"] = schema
            seen["method"] = method
            self.schema = schema
            return self

        async def ainvoke(self, messages, config):
            seen["messages"] = messages
            if self.schema is AnswerDraft:
                return AnswerDraft(answer="Ответ", explanation="Обоснование")
            return AgentReply(utterance="Тест")

    monkeypatch.setattr("langchain_gigachat.GigaChat", Client)
    provider = GigaChatProvider(
        Settings(
            _env_file=None,
            gigachat_credentials="server-secret",
            model="ChosenModel",
        )
    )
    assert json.loads(await provider.generate(request()))["utterance"] == "Тест"
    assert seen["config"]["model"] == "ChosenModel"
    assert seen["config"]["credentials"] == "server-secret"
    assert seen["config"]["verify_ssl_certs"] is True
    assert seen["config"]["max_retries"] == 0
    assert seen["schema"] is AgentReply
    assert seen["method"] == "function_calling"
    assert json.loads(seen["messages"][1].content)["question"] == "Опыт?"
    final = await provider.generate(ModelRequest(PLAYERS["science"], request().context, final=True))
    assert json.loads(final)["answer"] == "Ответ"
    assert seen["schema"] is AnswerDraft
    await provider.close()


async def test_timeout_includes_semaphore_wait():
    class NeverCalled:
        async def ainvoke(self, *args, **kwargs):
            pytest.fail("Client must not be called while semaphore is locked")

    provider = GigaChatProvider(Settings(_env_file=None, turn_timeout=0.01), client=NeverCalled())
    for _ in range(provider.settings.model_concurrency):
        await provider.semaphore.acquire()
    with pytest.raises(TimeoutError):
        await provider.generate(request())


def test_disabled_ssl_does_not_pass_ca_bundle(monkeypatch):
    seen = {}

    class Client:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr("langchain_gigachat.GigaChat", Client)
    GigaChatProvider(
        Settings(
            _env_file=None,
            gigachat_credentials="server-secret",
            verify_ssl=False,
            ca_bundle="/unused/ca.pem",
        )
    )
    assert seen["verify_ssl_certs"] is False
    assert seen["ca_bundle_file"] is None


async def test_demo_never_constructs_real_client(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Demo attempted to create GigaChat")

    monkeypatch.setattr("langchain_gigachat.GigaChat", forbidden)
    provider = DemoProvider(delay=0)
    result = json.loads(await provider.generate(request()))
    assert result["next_speakers"] == ["ideas", "humanities"]
    final = await provider.generate(ModelRequest(PLAYERS["science"], request().context, final=True))
    assert json.loads(final)["answer"] == "Маятник Фуко"
    await provider.close()


def test_context_never_truncates_accepted_utterances():
    provider = GigaChatProvider(Settings(_env_file=None, context_tokens=8000), client=object())
    with pytest.raises(ContextLimitError):
        provider.prepare(request(utterances=[{"text": "История" * 5000}]))
