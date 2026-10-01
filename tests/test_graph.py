import asyncio
import json

import pytest
from conftest import ManualClock, ScaledClock, settle
from pydantic import ValidationError

from chgk_team.agents import PLAYERS
from chgk_team.config import Settings
from chgk_team.domain import AgentReply, Hypothesis
from chgk_team.graph import DiscussionEngine
from chgk_team.providers import DemoProvider
from chgk_team.search import DemoSearch
from chgk_team.sessions import Session


def make_engine(model=None, **settings):
    config = Settings(_env_file=None, demo_delay=0, **settings)
    return DiscussionEngine(config, model or DemoProvider(0), DemoSearch())


async def test_real_graph_demo_all_players_memory_and_delegation():
    class RecordingDemo(DemoProvider):
        def __init__(self):
            super().__init__(0)
            self.requests = []

        async def generate(self, request):
            self.requests.append(request)
            return await super().generate(request)

    model = RecordingDemo()
    engine = make_engine(model)
    session = Session("Маятник?", engine.settings, ScaledClock(10))
    await asyncio.wait_for(engine.run(session), 2)
    assert session.phase == "finished"
    assert session.final_answer.speaker_id == "science"
    assert {u.speaker_id for u in session.utterances} == set(PLAYERS)
    assert session.hypotheses[0].status == "rejected"
    assert session.hypotheses[0].reasons[-1]["text"]
    assert session.hypotheses[1].status == "supported"
    assert session.search_count == 1
    assert session.sources[0].id == "demo-foucault"
    assert model.requests[-1].final
    previous_count = 0
    for request in model.requests:
        assert request.context["question"] == "Маятник?"
        assert len(request.context["utterances"]) >= previous_count
        previous_count = len(request.context["utterances"])
    assert set(engine.graph.get_graph().nodes) >= set(PLAYERS)
    assert sum(e.type == "session_finished" for e in session.events) == 1
    await engine.close()


def test_routing_fallback_fairness_and_repeats():
    engine = make_engine()
    session = Session("?", engine.settings, ManualClock())
    assert engine.choose_speaker(session) == "captain"
    session.attempts = ["captain", "ideas", "ideas"]
    session.queue = ["missing", "ideas"]
    assert engine.choose_speaker(session) == "humanities"
    session.since_captain = 2
    assert engine.choose_speaker(session) == "captain"


async def test_invalid_response_does_not_modify_versions_or_other_sessions():
    class InvalidModel(DemoProvider):
        async def generate(self, request):
            if request.final:
                return '{"answer":"X","explanation":"Y"}'
            return json.dumps(
                {
                    "utterance": "Отсекаем",
                    "changes": [{"action": "reject", "hypothesis_id": "missing", "reason": "Нет"}],
                }
            )

    engine = make_engine(InvalidModel(0), max_turns=1)
    clock = ManualClock()
    session = Session("Вопрос 1", engine.settings, clock)
    other = Session("Вопрос 2", engine.settings, clock)
    task = asyncio.create_task(engine.run(session))
    await settle()
    assert session.errors[0]["code"] == "invalid_reply"
    assert not session.hypotheses and not session.utterances
    assert not other.utterances and not other.errors
    await clock.advance(50)
    await task
    assert session.final_answer.speaker_id == "captain"
    await engine.close()


async def test_contaminated_speech_does_not_reach_history_or_export():
    class CaptainModel(DemoProvider):
        text = "Проверим версию. Следующие: ideas, humanities."

        async def generate(self, request):
            return json.dumps(
                {
                    "utterance": self.text,
                    "next_speakers": ["ideas", "humanities"],
                    "answerer": "science",
                    "ready": True,
                },
                ensure_ascii=False,
            )

    model = CaptainModel(0)
    engine = make_engine(model)
    session = Session("?", engine.settings, ManualClock())
    state = {"session": session, "epoch": 0, "speaker": "captain", "turn_deadline": 6}
    with pytest.raises(ValidationError):
        await engine.call(state)
    assert not session.utterances and not session.queue and not session.ready
    assert "Следующие:" not in json.dumps(session.export(), ensure_ascii=False)
    model.text = "Ника, Михаил, проверьте версию. Софья, отвечай."
    reply = await engine.call(state)
    await engine.commit({**state, "reply": reply})
    assert session.queue == ["ideas", "humanities"]
    assert session.answerer == "science" and session.ready
    assert session.export()["utterances"][0]["text"] == model.text
    assert session.context()["team"][0]["name"] == "Александр"
    await engine.close()


async def test_watchdog_discards_uncancellable_late_reply():
    release = asyncio.Event()
    entered = asyncio.Event()

    class SlowModel(DemoProvider):
        async def generate(self, request):
            if request.final:
                return '{"answer":"Финал","explanation":"Принятый контекст"}'
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()  # Simulate a remote result surviving local cancellation.
            return '{"utterance":"ПОЗДНЯЯ РЕПЛИКА"}'

    engine = make_engine(SlowModel(0))
    clock = ManualClock()
    session = Session("?", engine.settings, clock)
    task = asyncio.create_task(engine.run(session))
    await entered.wait()
    await clock.advance(50)
    await task
    assert session.phase == "finished"
    assert session.final_answer.answer == "Финал"
    release.set()
    await settle()
    assert not session.utterances
    assert sum(e.type == "session_finished" for e in session.events) == 1
    await engine.close()


async def test_final_timeout_preserves_only_preliminary_version():
    entered = asyncio.Event()

    class HangingFinal(DemoProvider):
        async def generate(self, request):
            if request.final:
                entered.set()
                await asyncio.Event().wait()
            return '{"utterance":"Не успеваю"}'

    engine = make_engine(HangingFinal(0), max_turns=1)
    clock = ManualClock()
    session = Session("?", engine.settings, clock)
    session.hypotheses = [Hypothesis(id="h1", text="Версия", proposed_by="ideas")]
    session.leading_id = "h1"
    task = asyncio.create_task(engine.run(session))
    await settle()
    await clock.advance(50)
    await entered.wait()
    await clock.advance(10)
    await task
    assert session.phase == "finished" and session.final_answer is None
    assert session.leading_id == "h1"
    assert session.finish_reason == "deadline"
    assert session.elapsed() == 60
    await engine.close()


async def test_bounds_hypotheses_and_captain_reconsideration():
    engine = make_engine()
    session = Session("?", engine.settings, ManualClock())
    session.hypotheses = [
        Hypothesis(id=f"h{i}", text="Версия", proposed_by="ideas") for i in range(1, 13)
    ]
    session.hypotheses[0].status = "rejected"
    await engine.commit(
        {
            "session": session,
            "epoch": 0,
            "speaker": "captain",
            "reply": AgentReply(
                utterance="Новый аргумент возвращает версию",
                leading_id="h1",
                changes=[
                    {"action": "new", "text": "Лишняя", "reason": "Ассоциация"},
                    {"action": "support", "hypothesis_id": "h1", "reason": "Новое подтверждение"},
                ],
            ),
        }
    )
    assert len(session.hypotheses) == 12
    assert session.hypotheses[0].status == "supported"
    assert session.leading_id == "h1"
    assert session.context()["utterances"][0]["text"] == "Новый аргумент возвращает версию"
    await engine.close()


async def test_final_error_is_not_a_fabricated_answer():
    class BrokenFinal(DemoProvider):
        async def generate(self, request):
            if request.final:
                raise RuntimeError("credential-value: provider failed")
            return '{"utterance":"Готовы","ready":true}'

    engine = make_engine(BrokenFinal(0))
    session = Session("?", engine.settings, ManualClock())
    await engine.run(session)
    assert session.final_answer is None
    assert session.finish_reason == "final_failed"
    assert "credential-value" not in json.dumps(session.export())
    await engine.close()
