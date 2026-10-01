import asyncio
import json
from contextlib import asynccontextmanager

from conftest import ManualClock, settle
from nicegui import ui
from nicegui.testing.user_simulation import user_simulation

from chgk_team.config import Settings
from chgk_team.graph import DiscussionEngine
from chgk_team.providers import DemoProvider
from chgk_team.search import DemoSearch
from chgk_team.sessions import SessionRegistry
from chgk_team.web import GamePage


@asynccontextmanager
async def simulation(*, settings=None, model=None):
    settings = settings or Settings(_env_file=None)
    engine = DiscussionEngine(settings, model or DemoProvider(0), DemoSearch())
    registry = SessionRegistry(settings, engine, clock=ManualClock())
    pages = []

    async def root():
        page = GamePage(registry, settings)
        pages.append(page)
        await ui.context.client.connected()
        page.restore()

    try:
        async with user_simulation(root) as user:
            await user.open("/")
            await settle()
            yield user, pages, registry
    finally:
        await registry.close()


async def test_nicegui_demo_interaction_export_and_new_game():
    async with simulation() as (user, pages, registry):
        await user.should_see("Команда за столом")
        user.find("Попробовать пример ↗").click()
        user.find(marker="start").click()
        await user.should_see(marker="final-result", retries=20)
        await user.should_see("Отвечает Софья")
        await user.should_see("Поддержана капитаном")
        session = pages[0].session
        assert len(session.utterances) == 10
        assert pages[0].rendered_utterances == 10
        user.find(marker="export").click()
        download = await user.download.next()
        exported = json.loads(download.content)
        assert exported["mode"] == "demo"
        assert exported["metrics"]["searches"] == 1
        user.find(marker="new-game").click()
        await user.should_see(marker="question-form")
        assert pages[0].session is None
        assert "session_id" not in pages[0].tab_storage


async def test_nicegui_reload_restores_same_session_without_duplicates():
    async with simulation() as (user, pages, registry):
        user.find(marker="question").type("Вопрос?")
        user.find(marker="start").click()
        await user.should_see(marker="final-result", retries=20)
        session = pages[0].session
        calls = session.model_calls
        await user.open("/")
        await user.should_see("Отвечает Софья", retries=20)
        assert pages[-1].session is session
        assert pages[-1].rendered_utterances == len(session.utterances)
        assert session.model_calls == calls
        assert len(registry.sessions) == 1


async def test_nicegui_invalid_input_and_missing_credentials():
    settings = Settings(_env_file=None, mode="real", gigachat_credentials=None)
    async with simulation(settings=settings) as (user, pages, registry):
        user.find(marker="start").click()
        await user.should_see("Введите вопрос длиной от 1 до 10 000 символов.")
        assert not registry.sessions
        user.find(marker="question").type("Реальный вопрос?")
        user.find(marker="start").click()
        await user.should_see("GigaChat не настроен. Добавьте серверные credentials в .env.")
        assert not registry.sessions


async def test_nicegui_failed_final_and_deleted_session():
    class Failing(DemoProvider):
        async def generate(self, request):
            if request.final:
                raise RuntimeError("Provider failed with private data")
            return '{"utterance":"Готовы","ready":true}'

    async with simulation(model=Failing(0)) as (user, pages, registry):
        user.find(marker="question").type("?")
        user.find(marker="start").click()
        await user.should_see(marker="final-result", retries=20)
        await user.should_see("Окончательный ответ не получен")
        assert pages[0].session.final_answer is None
        assert "private data" not in json.dumps(pages[0].session.export())
        registry.sessions.clear()
        await user.should_see("Сессия недоступна: удалена или сервер был перезапущен.", retries=10)


async def test_nicegui_clock_does_not_depend_on_browser_time():
    class Waiting(DemoProvider):
        async def generate(self, request):
            if request.final:
                return await super().generate(request)
            await asyncio.Event().wait()

    async with simulation(model=Waiting(0)) as (user, pages, registry):
        user.find(marker="question").type("?")
        user.find(marker="start").click()
        await settle()
        await registry.clock.advance(20)
        await user.should_see("00:40", retries=10)
        assert pages[0].session.elapsed() == 20
