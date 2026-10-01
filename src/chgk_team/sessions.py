import asyncio
from collections import deque
from datetime import timedelta
from uuid import uuid4

from chgk_team.agents import PROMPT_VERSION, TEAM
from chgk_team.clock import Clock, RealClock
from chgk_team.config import Settings
from chgk_team.domain import Event, FinalAnswer, Hypothesis, Source, Utterance


class CapacityError(Exception):
    pass


class Session:
    def __init__(self, question: str, settings: Settings, clock: Clock):
        self.id = str(uuid4())
        self.question = question
        self.settings = settings
        self.clock = clock
        self.started = clock.now()
        self.started_at = clock.utc()
        self.discussion_until = self.started + 50
        self.deadline = self.started + 60
        self.phase = "discussion"
        self.epoch = 0
        self.lock = asyncio.Lock()
        self.utterances: list[Utterance] = []
        self.hypotheses: list[Hypothesis] = []
        self.sources: list[Source] = []
        self.queue: list[str] = []
        self.answerer = "captain"
        self.leading_id: str | None = None
        self.direction = ""
        self.ready = False
        self.turns = 0
        self.since_captain = 0
        self.attempts: list[str] = []
        self.search_count = 0
        self.model_calls = 0
        self.call_metrics: list[dict] = []
        self.errors: list[dict] = []
        self.active_speaker: str | None = None
        self.search_status: str | None = None
        self.final_answer: FinalAnswer | None = None
        self.finish_reason: str | None = None
        self.finished_at: float | None = None
        self.events: deque[Event] = deque(maxlen=512)
        self.event_id = 0
        self.subscribers: set[asyncio.Queue] = set()
        self.task: asyncio.Task | None = None

    def elapsed(self) -> float:
        end = self.finished_at if self.finished_at is not None else self.clock.now()
        return round(max(0, end - self.started), 3)

    def emit(self, kind: str, data: dict) -> Event:
        """Called synchronously on the event loop; mutation and fan-out cannot interleave."""
        self.event_id += 1
        event = Event(
            id=self.event_id,
            session_id=self.id,
            type=kind,
            server_time=self.clock.utc().isoformat(),
            data=data,
        )
        self.events.append(event)
        for queue in tuple(self.subscribers):
            if queue.full():
                self.subscribers.discard(queue)
                # Replace pending data with a sentinel; reconnect replays from a snapshot.
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(None)
            else:
                queue.put_nowait(event)
        return event

    def context(self, *, search_notice: str | None = None) -> dict:
        result = {
            "question": self.question,
            "utterances": [u.model_dump() for u in self.utterances],
            "hypotheses": [h.model_dump() for h in self.hypotheses],
            "sources": [s.model_dump() for s in self.sources],
            "direction": self.direction,
            "answerer": self.answerer,
            "leading_id": self.leading_id,
            "remaining_seconds": max(0, 60 - self.elapsed()),
            "team": [
                {"id": p.id, "name": p.name, "role": p.label, "strengths": p.strengths}
                for p in TEAM
            ],
            "search_remaining": max(0, 2 - self.search_count),
        }
        if search_notice:
            result["search_notice"] = search_notice
        return result

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "mode": self.settings.mode,
            "phase": self.phase,
            "started_at": self.started_at.isoformat(),
            "deadline_at": (self.started_at + timedelta(seconds=60)).isoformat(),
            "server_time": self.clock.utc().isoformat(),
            "elapsed": self.elapsed(),
            "last_event_id": self.event_id,
            "active_speaker": self.active_speaker,
            "search_status": self.search_status,
            "utterances": [u.model_dump() for u in self.utterances],
            "hypotheses": [h.model_dump() for h in self.hypotheses],
            "sources": [s.model_dump() for s in self.sources],
            "leading_id": self.leading_id,
            "answerer": self.answerer,
            "final_answer": self.final_answer.model_dump() if self.final_answer else None,
            "finish_reason": self.finish_reason,
            "errors": list(self.errors),
        }

    def export(self) -> dict:
        return {
            **self.snapshot(),
            "format_version": 1,
            "team": [p.model_dump(mode="json", exclude={"instruction"}) for p in TEAM],
            "configuration": self.settings.public(),
            "prompt_version": PROMPT_VERSION,
            "metrics": {
                "turns": self.turns,
                "model_calls": self.model_calls,
                "searches": self.search_count,
                "calls": list(self.call_metrics),
            },
        }

    async def finish(self, answer: FinalAnswer | None, reason: str, epoch: int) -> bool:
        async with self.lock:
            if self.phase == "finished" or epoch != self.epoch:
                return False
            # An answer which arrives at or after the deadline is never accepted.
            if self.clock.now() >= self.deadline:
                answer, reason = None, "deadline"
            self.phase = "finished"
            self.epoch += 1
            self.final_answer = answer
            self.finish_reason = reason
            self.finished_at = min(self.clock.now(), self.deadline)
            self.active_speaker = None
            self.search_status = None
            self.emit("session_finished", self.snapshot())
            return True

    async def subscribe(self, cursor: int | None) -> tuple[list[Event], asyncio.Queue]:
        async with self.lock:
            queue = asyncio.Queue(maxsize=64)
            if (
                cursor is None
                or cursor > self.event_id
                or (self.events and cursor < self.events[0].id - 1)
            ):
                replay = [
                    Event(
                        id=self.event_id,
                        session_id=self.id,
                        type="snapshot",
                        server_time=self.clock.utc().isoformat(),
                        data=self.snapshot(),
                    )
                ]
            else:
                replay = [event for event in self.events if event.id > cursor]
            if self.phase != "finished":
                self.subscribers.add(queue)
            return replay, queue


class SessionRegistry:
    def __init__(self, settings: Settings, engine, *, clock: Clock | None = None):
        self.settings = settings
        self.engine = engine
        self.clock = clock or RealClock()
        self.sessions: dict[str, Session] = {}
        self.lock = asyncio.Lock()
        self.cleanup_task: asyncio.Task | None = None

    def prune(self) -> None:
        now = self.clock.now()
        for session_id, session in tuple(self.sessions.items()):
            if (
                session.finished_at is not None
                and now - session.finished_at >= self.settings.session_ttl
            ):
                del self.sessions[session_id]

    async def create(self, question: str) -> Session:
        async with self.lock:
            self.prune()
            active = sum(s.phase != "finished" for s in self.sessions.values())
            if active >= self.settings.max_active_sessions:
                raise CapacityError
            if len(self.sessions) >= self.settings.max_saved_sessions:
                completed = [s for s in self.sessions.values() if s.phase == "finished"]
                if not completed:
                    raise CapacityError
                oldest = min(completed, key=lambda s: s.finished_at)
                del self.sessions[oldest.id]
            session = Session(question, self.settings, self.clock)
            self.sessions[session.id] = session
            session.emit("session_started", session.snapshot())
            session.task = asyncio.create_task(self.engine.run(session))
            return session

    def get(self, session_id: str) -> Session | None:
        self.prune()
        return self.sessions.get(session_id)

    async def cleanup(self) -> None:
        while True:
            await self.clock.sleep(30)
            self.prune()

    async def close(self) -> None:
        if self.cleanup_task:
            self.cleanup_task.cancel()
        for session in self.sessions.values():
            if session.task and not session.task.done():
                session.task.cancel()
        tasks = [s.task for s in self.sessions.values() if s.task]
        if self.cleanup_task:
            tasks.append(self.cleanup_task)
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.engine.close()
