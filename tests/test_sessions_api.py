import asyncio
import json

import pytest
from conftest import ManualClock, ScaledClock, settle
from fastapi.testclient import TestClient

from chgk_team.api import create_api
from chgk_team.config import Settings
from chgk_team.domain import Event, FinalAnswer
from chgk_team.graph import DiscussionEngine
from chgk_team.providers import DemoProvider
from chgk_team.search import DemoSearch
from chgk_team.sessions import CapacityError, Session, SessionRegistry


class IdleEngine:
    def __init__(self):
        self.calls = 0
        self.closed = False

    async def run(self, session):
        self.calls += 1
        await asyncio.Event().wait()

    async def close(self):
        self.closed = True


async def test_capacity_ttl_and_evict_completed_only():
    settings = Settings(_env_file=None, max_active_sessions=1, max_saved_sessions=2, session_ttl=60)
    clock, engine = ManualClock(), IdleEngine()
    registry = SessionRegistry(settings, engine, clock=clock)
    first = await registry.create("Первый")
    with pytest.raises(CapacityError):
        await registry.create("Слишком много")
    await first.finish(None, "test", first.epoch)
    second = await registry.create("Второй")
    await second.finish(None, "test", second.epoch)
    third = await registry.create("Третий")
    assert first.id not in registry.sessions
    assert second.id in registry.sessions and third.id in registry.sessions
    await clock.advance(60)
    assert registry.get(second.id) is None
    assert registry.get(third.id) is third  # Active games survive pruning.
    await registry.close()
    assert engine.closed and third.task.done()


async def test_snapshot_subscription_replay_and_slow_subscriber():
    session = Session("?", Settings(_env_file=None), ManualClock())
    session.emit("session_started", session.snapshot())
    replay, queue = await session.subscribe(None)
    assert replay[0].type == "snapshot"
    assert replay[0].data["last_event_id"] == 1
    session.emit("speaker_started", {"speaker_id": "captain"})
    assert (await queue.get()).id == 2
    replay2, _ = await session.subscribe(1)
    assert [event.id for event in replay2] == [2]
    for i in range(65):
        session.emit("test", {"n": i})
    assert queue not in session.subscribers
    assert await queue.get() is None
    # Force event-retention rollover and restore from a snapshot.
    for i in range(520):
        session.emit("test", {"n": i})
    replay3, _ = await session.subscribe(1)
    assert replay3[0].type == "snapshot"
    assert replay3[0].id == session.event_id


async def test_disconnect_does_not_restart_server_game():
    settings = Settings(_env_file=None)
    engine = IdleEngine()
    registry = SessionRegistry(settings, engine, clock=ManualClock())
    session = await registry.create("?")
    await settle()
    _, first = await session.subscribe(None)
    session.subscribers.discard(first)  # Browser disconnects.
    _, second = await session.subscribe(session.event_id)
    assert engine.calls == 1
    assert not session.task.done()
    assert second in session.subscribers
    await registry.close()


async def test_export_explicit_demo_mode_and_no_secrets():
    settings = Settings(_env_file=None, gigachat_credentials="secret-value")
    engine = DiscussionEngine(settings, DemoProvider(0), DemoSearch())
    session = Session("?", settings, ScaledClock(10))
    await engine.run(session)
    exported = session.export()
    assert exported["mode"] == "demo"
    assert exported["final_answer"]["speaker_id"] == "science"
    assert exported["metrics"]["model_calls"] > len(exported["utterances"])
    assert exported["prompt_version"]
    assert exported["sources"]
    assert "secret-value" not in json.dumps(exported)
    await engine.close()


def test_api_input_statuses_and_team():
    settings = Settings(_env_file=None, max_active_sessions=1)
    with TestClient(create_api(settings, model=DemoProvider(0))) as client:
        team = client.get("/api/team").json()
        assert len(team["players"]) == 6 and team["mode"] == "demo"
        for question in ["", " ", "x" * 10001]:
            assert client.post("/api/sessions", json={"question": question}).status_code == 422
        assert not client.app.state.registry.sessions
        created = client.post("/api/sessions", json={"question": "  Опыт?  "})
        assert created.status_code == 201
        session_id = created.json()["id"]
        assert created.json()["question"] == "Опыт?"
        assert client.post("/api/sessions", json={"question": "Ещё?"}).status_code == 429
        assert client.get(f"/api/sessions/{session_id}").status_code == 200
        assert client.get("/api/sessions/missing").status_code == 404
        exported = client.get(f"/api/sessions/{session_id}/export")
        assert (
            exported.status_code == 200 and "attachment" in exported.headers["content-disposition"]
        )
        assert exported.json()["mode"] == "demo"


def test_real_missing_credentials_does_not_start_graph():
    settings = Settings(_env_file=None, mode="real", gigachat_credentials=None)
    with TestClient(create_api(settings)) as client:
        assert client.get("/api/team").json()["ready"] is False
        assert client.post("/api/sessions", json={"question": "?"}).status_code == 503
        assert not client.app.state.registry.sessions


def test_finished_sse_replays_header_cursor_and_snapshot():
    settings = Settings(_env_file=None)
    with TestClient(create_api(settings)) as client:
        registry = client.app.state.registry
        session = Session("Сохранённый вопрос", settings, ManualClock())
        registry.sessions[session.id] = session
        session.emit("session_started", session.snapshot())
        session.phase = "finished"
        session.emit("session_finished", session.snapshot())
        path = f"/api/sessions/{session.id}/events"
        response = client.get(path, headers={"Last-Event-ID": "1"})
        assert response.status_code == 200
        assert "event: session_finished" in response.text
        assert "event: session_started" not in response.text
        snapshot = client.get(path, headers={"Last-Event-ID": "invalid"})
        assert "event: snapshot" in snapshot.text


async def test_only_one_final_result_is_accepted():
    session = Session("?", Settings(_env_file=None), ManualClock())
    answer = FinalAnswer(answer="X", explanation="Y", speaker_id="captain")
    assert await session.finish(answer, "answered", 0)
    assert not await session.finish(answer, "answered", 0)
    assert sum(e.type == "session_finished" for e in session.events) == 1
    assert isinstance(session.events[-1], Event)
