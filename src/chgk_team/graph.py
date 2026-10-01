import asyncio
from collections.abc import Awaitable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from chgk_team.agents import PLAYERS, validate_reply
from chgk_team.config import Settings
from chgk_team.domain import AgentReply, AnswerDraft, FinalAnswer, Hypothesis, Role, Utterance
from chgk_team.providers import ContextLimitError, ModelProvider, ModelRequest
from chgk_team.search import SearchProvider
from chgk_team.sessions import Session


class TurnState(TypedDict, total=False):
    session: Session
    epoch: int
    speaker: str | None
    reply: AgentReply | AnswerDraft | None
    error: str | None
    turn_deadline: float


class DiscussionEngine:
    def __init__(self, settings: Settings, model: ModelProvider, search: SearchProvider):
        self.settings = settings
        self.model = model
        self.search_provider = search
        self.pending: set[asyncio.Task] = set()
        builder = StateGraph(TurnState)
        builder.add_node("control", self.control)
        for player_id in PLAYERS:
            builder.add_node(player_id, self.player_node(player_id))
            builder.add_edge(
                player_id, "search" if PLAYERS[player_id].role == Role.SCHOLAR else "commit"
            )
        builder.add_node("search", self.search)
        builder.add_node("commit", self.commit)
        builder.add_edge(START, "control")
        builder.add_conditional_edges(
            "control", lambda state: state["speaker"] or END, {**{p: p for p in PLAYERS}, END: END}
        )
        builder.add_edge("search", "commit")
        builder.add_edge("commit", "control")
        self.graph = builder.compile()

    def track(self, operation: Awaitable) -> asyncio.Task:
        task = asyncio.ensure_future(operation)
        self.pending.add(task)

        def completed(done):
            self.pending.discard(done)
            if not done.cancelled():
                done.exception()  # Consume exceptions even when a timed-out caller detached.

        task.add_done_callback(completed)
        return task

    async def bounded(self, operation: Awaitable, session: Session, deadline: float):
        """Stop waiting at the server deadline, even if an adapter ignores cancellation."""
        task = self.track(operation)
        timer = self.track(session.clock.sleep(max(0, deadline - session.clock.now())))
        try:
            done, _ = await asyncio.wait({task, timer}, return_when=asyncio.FIRST_COMPLETED)
            if task in done and session.clock.now() < deadline:
                return task.result()
            raise TimeoutError
        finally:
            if not task.done():
                task.cancel()
            if not timer.done():
                timer.cancel()

    async def run(self, session: Session) -> None:
        try:
            epoch = session.epoch
            try:
                await self.bounded(
                    self.graph.ainvoke(
                        {"session": session, "epoch": epoch}, {"recursion_limit": 150}
                    ),
                    session,
                    session.discussion_until,
                )
            except TimeoutError:
                pass
            if session.phase == "finished":
                return
            if not session.ready:
                await session.clock.sleep(max(0, session.discussion_until - session.clock.now()))
            async with session.lock:
                if session.phase == "finished":
                    return
                session.epoch += 1
                epoch = session.epoch
                session.phase = "finalizing"
                session.active_speaker = None
                session.search_status = None
                session.emit("phase_changed", {"phase": "finalizing", "answerer": session.answerer})
            try:
                await self.bounded(
                    self.graph.ainvoke(
                        {"session": session, "epoch": epoch}, {"recursion_limit": 20}
                    ),
                    session,
                    session.deadline,
                )
            except TimeoutError:
                await session.finish(None, "deadline", epoch)
            if session.phase != "finished":
                await session.finish(None, "final_failed", epoch)
        except asyncio.CancelledError:
            await session.finish(None, "shutdown", session.epoch)
            raise
        except Exception:
            # External exception strings can contain credentials or provider payloads.
            await session.finish(None, "engine_error", session.epoch)

    def valid_epoch(self, session: Session, epoch: int) -> bool:
        return session.epoch == epoch and session.phase != "finished"

    async def control(self, state: TurnState) -> dict:
        session, epoch = state["session"], state["epoch"]
        async with session.lock:
            if not self.valid_epoch(session, epoch):
                return {"speaker": None}
            final = session.phase == "finalizing"
            boundary = session.deadline if final else session.discussion_until
            if session.clock.now() >= boundary:
                return {"speaker": None}
            if not final and (session.ready or session.turns >= self.settings.max_turns):
                return {"speaker": None}
            if final:
                speaker = session.answerer
            else:
                speaker = self.choose_speaker(session)
                session.turns += 1
                session.attempts.append(speaker)
                session.since_captain = 0 if speaker == "captain" else session.since_captain + 1
            session.active_speaker = speaker
            session.emit("speaker_started", {"speaker_id": speaker, "final": final})
            return {
                "speaker": speaker,
                "reply": None,
                "error": None,
                "turn_deadline": boundary
                if final
                else min(boundary, session.clock.now() + self.settings.turn_timeout),
            }

    @staticmethod
    def choose_speaker(session: Session) -> str:
        if not session.attempts:
            return "captain"
        blocked = (
            session.attempts[-1]
            if (len(session.attempts) >= 2 and session.attempts[-1] == session.attempts[-2])
            else None
        )
        if session.since_captain >= 2 and blocked != "captain":
            return "captain"
        while session.queue:
            candidate = session.queue.pop(0)
            if candidate in PLAYERS and candidate != blocked:
                return candidate
        eligible = [p for p in PLAYERS if p != "captain" and p != blocked]
        for candidate in eligible:
            if candidate not in session.attempts:
                return candidate
        return min(
            eligible,
            key=lambda p: max(
                (i for i, previous in enumerate(session.attempts) if p == previous), default=-1
            ),
        )

    async def call(self, state: TurnState, *, notice: str | None = None):
        session, epoch = state["session"], state["epoch"]
        if not self.valid_epoch(session, epoch):
            raise TimeoutError
        final = session.phase == "finalizing"
        context = session.context(search_notice=notice)
        request = ModelRequest(PLAYERS[state["speaker"]], context, final=final)
        session.model_calls += 1
        metric = {
            "speaker_id": state["speaker"],
            "final": final,
            "started": session.elapsed(),
            "latency": None,
            "status": "pending",
        }
        session.call_metrics.append(metric)
        began = session.clock.now()
        try:
            raw = await self.bounded(self.model.generate(request), session, state["turn_deadline"])
            if len(raw) > 30000:
                raise ValueError("Model output is too large")
            if final:
                reply = AnswerDraft.model_validate_json(raw)
                if set(reply.source_ids) - {s["id"] for s in context["sources"]}:
                    raise ValueError("Unknown source")
            else:
                reply = AgentReply.model_validate_json(raw)
                validate_reply(request.player, reply, context)
            metric["status"] = "ok"
            return reply
        except asyncio.CancelledError:
            metric["status"] = "cancelled"
            raise
        except Exception as error:
            metric["status"] = "failed"
            metric["error_type"] = type(error).__name__
            raise
        finally:
            metric["latency"] = round(max(0, session.clock.now() - began), 3)

    @staticmethod
    def error_code(error: Exception) -> str:
        if isinstance(error, TimeoutError):
            return "model_timeout"
        if isinstance(error, ContextLimitError):
            return "context_limit"
        if isinstance(error, (ValueError, ValidationError)):
            return "invalid_reply"
        return "model_unavailable"

    def player_node(self, player_id: str):
        async def node(state: TurnState) -> dict:
            try:
                return {"reply": await self.call(state), "error": None}
            except Exception as error:
                return {"reply": None, "error": self.error_code(error)}

        return node

    async def search(self, state: TurnState) -> dict:
        session, epoch, reply = state["session"], state["epoch"], state.get("reply")
        if not isinstance(reply, AgentReply) or not reply.search_query:
            return {}
        if not self.valid_epoch(session, epoch):
            return {}
        async with session.lock:
            if not self.valid_epoch(session, epoch):
                return {}
            allowed = (
                self.settings.search_enabled
                and session.phase == "discussion"
                and session.search_count < 2
                and session.elapsed() < 45
                and session.clock.now() < state["turn_deadline"]
            )
            if allowed:
                session.search_count += 1
                session.search_status = "searching"
                session.emit(
                    "search_started", {"speaker_id": state["speaker"], "query": reply.search_query}
                )
        notice = "Поиск недоступен: отключён либо исчерпан лимит. Продолжи без подтверждения."
        if allowed:
            try:
                sources = await self.bounded(
                    self.search_provider.search(reply.search_query),
                    session,
                    min(state["turn_deadline"], session.clock.now() + 3),
                )
                notice = "Получены справочные данные. Используй только выданные source_ids."
                async with session.lock:
                    if (
                        not self.valid_epoch(session, epoch)
                        or session.clock.now() >= session.discussion_until
                    ):
                        return {}
                    existing = {s.id for s in session.sources}
                    session.sources.extend(s for s in sources[:3] if s.id not in existing)
                    session.search_status = None
                    session.emit(
                        "search_finished",
                        {
                            "speaker_id": state["speaker"],
                            "status": "found" if sources else "empty",
                            "sources": [s.model_dump() for s in sources[:3]],
                        },
                    )
            except Exception:
                notice = "Поиск не дал подтверждения: источник недоступен. Продолжи по памяти."
                async with session.lock:
                    if self.valid_epoch(session, epoch):
                        session.search_status = None
                        session.emit(
                            "search_finished",
                            {
                                "speaker_id": state["speaker"],
                                "status": "unavailable",
                                "sources": [],
                            },
                        )
        try:
            updated = await self.call(state, notice=notice)
            if updated.search_query:
                raise ValueError("A second search in the same turn is forbidden")
            return {"reply": updated, "error": None}
        except Exception as error:
            return {"reply": None, "error": self.error_code(error)}

    async def commit(self, state: TurnState) -> dict:
        session, epoch = state["session"], state["epoch"]
        final = session.phase == "finalizing"
        reply, error = state.get("reply"), state.get("error")
        if final:
            answer = (
                FinalAnswer(**reply.model_dump(), speaker_id=state["speaker"])
                if (isinstance(reply, AnswerDraft) and not error)
                else None
            )
            if error and self.valid_epoch(session, epoch):
                session.errors.append(
                    {"speaker_id": state["speaker"], "code": error, "final": True}
                )
            await session.finish(answer, "answered" if answer else "final_failed", epoch)
            return {}
        async with session.lock:
            if (
                not self.valid_epoch(session, epoch)
                or session.clock.now() >= session.discussion_until
            ):
                return {}
            session.active_speaker = None
            if error or not isinstance(reply, AgentReply):
                entry = {
                    "speaker_id": state["speaker"],
                    "code": error or "invalid_reply",
                    "final": False,
                }
                session.errors.append(entry)
                session.emit("turn_failed", entry)
                return {}
            speaker = state["speaker"]
            for change in reply.changes:
                if change.action == "new":
                    if len(session.hypotheses) >= 12:
                        continue
                    hypothesis = Hypothesis(
                        id=f"h{len(session.hypotheses) + 1}",
                        text=change.text,
                        proposed_by=speaker,
                    )
                    session.hypotheses.append(hypothesis)
                else:
                    hypothesis = next(h for h in session.hypotheses if h.id == change.hypothesis_id)
                hypothesis.reasons.append(
                    {"speaker_id": speaker, "action": change.action, "text": change.reason}
                )
                hypothesis.source_ids = list(
                    dict.fromkeys(hypothesis.source_ids + change.source_ids)
                )
                if change.action in {"support", "reject"}:
                    hypothesis.status = "supported" if change.action == "support" else "rejected"
                    if change.action == "reject" and session.leading_id == hypothesis.id:
                        session.leading_id = None
            if PLAYERS[speaker].role == Role.CAPTAIN:
                session.direction = reply.utterance
                session.queue = [p for p in reply.next_speakers if p in PLAYERS]
                session.answerer = reply.answerer or session.answerer
                session.leading_id = reply.leading_id or session.leading_id
                if session.leading_id and any(
                    h.id == session.leading_id and h.status == "rejected"
                    for h in session.hypotheses
                ):
                    session.leading_id = None
                session.ready = reply.ready
            utterance = Utterance(
                index=len(session.utterances) + 1,
                speaker_id=speaker,
                text=reply.utterance,
                elapsed=session.elapsed(),
                source_ids=reply.source_ids,
            )
            session.utterances.append(utterance)
            session.emit("utterance", utterance.model_dump())
            session.emit(
                "hypotheses_updated",
                {
                    "hypotheses": [h.model_dump() for h in session.hypotheses],
                    "leading_id": session.leading_id,
                    "answerer": session.answerer,
                },
            )
        return {}

    async def close(self) -> None:
        for task in tuple(self.pending):
            task.cancel()
        if self.pending:
            await asyncio.wait(self.pending, timeout=1)
        await self.model.close()
        await self.search_provider.close()
