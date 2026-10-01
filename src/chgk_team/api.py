import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from chgk_team.agents import TEAM
from chgk_team.clock import Clock
from chgk_team.config import Settings
from chgk_team.domain import SessionInput
from chgk_team.graph import DiscussionEngine
from chgk_team.providers import DemoProvider, GigaChatProvider
from chgk_team.search import DemoSearch, WikipediaSearch
from chgk_team.sessions import CapacityError, SessionRegistry

ASSETS = Path(__file__).parent


class UnavailableModel:
    async def generate(self, request):
        raise RuntimeError("Real provider is not configured")

    async def close(self):
        pass


def create_api(
    settings: Settings | None = None, *, model=None, search=None, clock: Clock | None = None
):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        provider = model
        if provider is None:
            provider = (
                DemoProvider(settings.demo_delay)
                if settings.mode == "demo"
                else GigaChatProvider(settings)
                if settings.real_ready
                else UnavailableModel()
            )
        source = search or (DemoSearch() if settings.mode == "demo" else WikipediaSearch())
        engine = DiscussionEngine(settings, provider, source)
        registry = SessionRegistry(settings, engine, clock=clock)
        app.state.registry = registry
        registry.cleanup_task = asyncio.create_task(registry.cleanup())
        try:
            yield
        finally:
            await registry.close()

    app = FastAPI(title="Минута · команда ЧГК", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=ASSETS / "static"), name="static")

    @app.get("/api/team")
    async def team():
        return {
            "players": [p.model_dump(mode="json", exclude={"instruction"}) for p in TEAM],
            "mode": settings.mode,
            "search_enabled": settings.search_enabled,
            "ready": settings.mode == "demo" or settings.real_ready,
        }

    @app.post("/api/sessions", status_code=201)
    async def start(body: SessionInput):
        if settings.mode == "real" and not settings.real_ready:
            raise HTTPException(503, "GigaChat не настроен. Добавьте серверные credentials в .env.")
        try:
            session = await app.state.registry.create(body.question)
        except CapacityError:
            raise HTTPException(
                429, "Все места заняты. Дождитесь окончания одной из игр."
            ) from None
        return session.snapshot()

    def lookup(session_id):
        session = app.state.registry.get(session_id)
        if session is None:
            raise HTTPException(404, "Сессия недоступна: удалена или сервер был перезапущен.")
        return session

    @app.get("/api/sessions/{session_id}")
    async def snapshot(session_id: str):
        return lookup(session_id).snapshot()

    @app.get("/api/sessions/{session_id}/export")
    async def export(session_id: str):
        session = lookup(session_id)
        return JSONResponse(
            session.export(),
            headers={
                "Content-Disposition": f'attachment; filename="chgk-{session.id}.json"',
            },
        )

    @app.get("/api/sessions/{session_id}/events")
    async def events(request: Request, session_id: str, cursor: int | None = Query(None, ge=0)):
        session = lookup(session_id)
        last_id = request.headers.get("last-event-id")
        if last_id is not None:
            try:
                cursor = max(0, int(last_id))
            except ValueError:
                cursor = None
        replay, queue = await session.subscribe(cursor)

        def encode(event):
            return (
                f"id: {event.id}\nevent: {event.type}\ndata: "
                + json.dumps(event.model_dump(), ensure_ascii=False)
                + "\n\n"
            )

        async def stream():
            try:
                for event in replay:
                    yield encode(event)
                while True:
                    if session.phase == "finished" and queue.empty():
                        return
                    if await request.is_disconnected():
                        return
                    try:
                        event = await asyncio.wait_for(queue.get(), 15)
                    except TimeoutError:
                        yield ": heartbeat\n\n"
                        continue
                    if event is None:
                        return
                    yield encode(event)
                    if event.type == "session_finished":
                        return
            finally:
                session.subscribers.discard(queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def create_app(
    settings: Settings | None = None, *, model=None, search=None, clock: Clock | None = None
):
    from chgk_team.web import mount_ui

    settings = settings or Settings()
    app = create_api(settings, model=model, search=search, clock=clock)
    mount_ui(app, settings)
    return app


def main():
    parser = argparse.ArgumentParser(description="Запуск команды ЧГК")
    parser.add_argument("--mode", choices=["real", "demo"])
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    settings = Settings(**({"mode": args.mode} if args.mode else {}))
    uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port, workers=1)
