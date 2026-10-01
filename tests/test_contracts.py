import json

import pytest
from pydantic import ValidationError

from chgk_team.agents import PLAYERS, TEAM, messages_for, validate_reply
from chgk_team.config import Settings
from chgk_team.domain import AgentReply, AnswerDraft, HypothesisChange, Role, SessionInput, Source


def context():
    return {
        "question": "Какой опыт?",
        "utterances": [
            {"speaker_id": "ideas", "text": "Попробуем часы"},
            {"speaker_id": "logic", "text": "Часы не подходят"},
        ],
        "hypotheses": [{"id": "h1", "text": "Часы", "status": "rejected"}],
        "sources": [],
        "direction": "Проверь доказательство",
        "answerer": "science",
    }


def test_credentials_do_not_serialize_and_errors_hide_inputs():
    settings = Settings(_env_file=None, gigachat_credentials="secret-value")
    assert settings.real_ready
    assert "secret-value" not in str(settings.public())
    assert "secret-value" not in repr(settings)
    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None, mode="secret-value")
    assert "secret-value" not in str(error.value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_active_sessions=4, max_saved_sessions=2)


@pytest.mark.parametrize("name", ["CHGK_VERIFY_SSL", "GIGACHAT__VERIFY_SSL"])
def test_ssl_setting_from_environment_and_dotenv(name, monkeypatch, tmp_path):
    for alias in ("CHGK_VERIFY_SSL", "GIGACHAT__VERIFY_SSL"):
        monkeypatch.delenv(alias, raising=False)
    assert Settings(_env_file=None).verify_ssl is True
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"{name}=false\n")
    assert Settings(_env_file=dotenv).verify_ssl is False
    monkeypatch.setenv(name, "false")
    assert Settings(_env_file=None).verify_ssl is False
    monkeypatch.setenv(name, "true")
    assert Settings(_env_file=dotenv).verify_ssl is True


def test_ssl_alias_precedence(monkeypatch):
    monkeypatch.setenv("CHGK_VERIFY_SSL", "true")
    monkeypatch.setenv("GIGACHAT__VERIFY_SSL", "false")
    assert Settings(_env_file=None).verify_ssl is True
    assert Settings(_env_file=None, verify_ssl=False).verify_ssl is False


def test_team_profiles_and_context():
    assert len(TEAM) == len(PLAYERS) == 6
    assert sum(p.role == Role.CAPTAIN for p in TEAM) == 1
    scholars = [p for p in TEAM if p.role == Role.SCHOLAR]
    assert len(scholars) == 2
    assert scholars[0].strengths != scholars[1].strengths
    prompts = []
    for player in TEAM:
        messages = messages_for(player, context())
        prompts.append(messages[0].content)
        assert json.loads(messages[1].content) == context()
        assert player.name in messages[0].content
        assert all(topic in messages[0].content for topic in player.weaknesses)
    assert len(set(prompts)) == 6
    assert "окончательный" in messages_for(TEAM[0], context(), final=True)[0].content


@pytest.mark.parametrize("question", ["", "   ", "x" * 10001])
def test_invalid_question(question):
    with pytest.raises(ValidationError):
        SessionInput(question=question)


def test_trim_question_and_invalid_reply():
    assert SessionInput(question="  Текст?  ").question == "Текст?"
    with pytest.raises(ValidationError):
        AgentReply(utterance="x" * 351)
    with pytest.raises(ValidationError):
        HypothesisChange(action="new", reason="Не хватает текста")
    with pytest.raises(ValidationError):
        Source(id="s", title="X", url="http://localhost", snippet="x")


@pytest.mark.parametrize(
    "text",
    [
        "Поддерживаю версию веса. Следующие: ideas, humanities.",
        "Назначаю отвечающего: captain. Готовность: true.",
        'вес", "proposed_by": "humanities"',
        'вес", "explanation',
        'Версия подходит. next_speakers=["science"].',
    ],
)
def test_control_fragments_are_rejected_in_public_fields(text):
    for constructor, fields in (
        (AgentReply, {"utterance": text}),
        (HypothesisChange, {"action": "new", "text": text, "reason": "Довод"}),
        (HypothesisChange, {"action": "argument", "hypothesis_id": "h1", "reason": text}),
        (AnswerDraft, {"answer": text, "explanation": "Обоснование"}),
        (AnswerDraft, {"answer": "Ответ", "explanation": text}),
    ):
        with pytest.raises(ValidationError):
            constructor(**fields)


@pytest.mark.parametrize(
    "text",
    [
        "Ника, Михаил, проверьте эту версию. Софья, отвечай.",
        "Я отвечу. Останавливаемся на этой версии.",
        "Следующие подсказки связаны с историей и литературой.",
        "Слово «ready» означает «готовый» — возможно, это игра слов.",
    ],
)
def test_natural_speech_and_captain_controls_remain_valid(text):
    reply = AgentReply(
        utterance=text, next_speakers=["ideas", "humanities"], answerer="science", ready=True
    )
    validate_reply(PLAYERS["captain"], reply, context())
    assert reply.utterance == text
    assert reply.next_speakers == ["ideas", "humanities"]


@pytest.mark.parametrize(
    "data",
    [
        {"answerer": "science"},
        {"next_speakers": ["captain"]},
        {"changes": [{"action": "reject", "hypothesis_id": "h1", "reason": "Нет"}]},
        {"search_query": "Фуко"},
        {"source_ids": ["invented"]},
        {"changes": [{"action": "argument", "hypothesis_id": "missing", "reason": "Нет"}]},
    ],
)
def test_reject_unauthorized_fields_and_unknown_references(data):
    reply = AgentReply(utterance="Версия", **data)
    with pytest.raises(ValueError):
        validate_reply(PLAYERS["ideas"], reply, context())


def test_captain_and_scholar_permissions():
    validate_reply(
        PLAYERS["captain"],
        AgentReply(
            utterance="Отсекаем",
            changes=[HypothesisChange(action="reject", hypothesis_id="h1", reason="Противоречие")],
            answerer="science",
            next_speakers=["unknown"],
            leading_id="h1",
        ),
        context(),
    )
    validate_reply(
        PLAYERS["science"], AgentReply(utterance="Проверю", search_query="Фуко"), context()
    )
