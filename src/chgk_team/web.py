"""NiceGUI components; all game state and clocks stay on the server."""

import json
from pathlib import Path
from urllib.parse import urlsplit

from nicegui import app as nicegui_app
from nicegui import ui
from pydantic import ValidationError

from chgk_team.agents import PLAYERS, TEAM
from chgk_team.config import Settings
from chgk_team.domain import SessionInput
from chgk_team.sessions import CapacityError, Session, SessionRegistry

EXAMPLE = (
    "В парижском здании подвесили тяжёлый шар на длинном тросе. Во время опыта направление "
    "его колебаний относительно пола постепенно менялось, хотя шар никто не поворачивал. "
    "Как называется этот прибор, наглядно показывающий вращение Земли?"
)
STATUS_LABELS = {
    "proposed": "Предварительная",
    "supported": "Поддержана капитаном",
    "rejected": "Отклонена",
}
PHASE_LABELS = {
    "discussion": "обсуждение",
    "finalizing": "выбираем ответ",
    "finished": "игра завершена",
}


def label(text: str, classes: str = "", *, mark: str | None = None):
    component = ui.label(text).classes(classes)
    if mark:
        component.mark(mark)
    return component


def heading(text: str, level: int = 2):
    return label(text, f"heading h{level}").props(f"role=heading aria-level={level}")


def source_link(source: dict):
    parsed = urlsplit(source["url"])
    if parsed.scheme == "https" and parsed.netloc == "ru.wikipedia.org":
        return ui.link(source["title"], source["url"], new_tab=True).classes("source-link")
    return label(source["title"], "source-link")


class GamePage:
    def __init__(self, registry: SessionRegistry, settings: Settings):
        self.registry, self.settings = registry, settings
        self.session: Session | None = None
        self.tab_storage: dict | None = None
        self.last_event = -1
        self.rendered_utterances = 0
        self.follow_latest = True
        self.player_cards = {}
        self.player_status = {}
        self.client = ui.context.client
        self.build()
        self.timer = ui.timer(0.2, self.refresh)
        self.client.on_delete(self.timer.cancel)

    def build(self):
        ui.colors(primary="#243a35", secondary="#8d744d", accent="#c5965d")
        with ui.element("header").classes("masthead"):
            with ui.row().classes("brand items-center"):
                label("М", "brand-mark")
                label("минута.", "brand-name")
            with ui.row().classes("masthead-right items-center"):
                label("ШЕСТЬ ИГРОКОВ · ОДИН ВОПРОС", "eyebrow masthead-tag")
                label(
                    "Демо · без GigaChat"
                    if self.settings.mode == "demo"
                    else "GigaChat · живая игра",
                    "mode-badge",
                    mark="mode-badge",
                )
        with ui.element("main").classes("game-main"):
            with ui.element("section").classes("intro"):
                with ui.column().classes("intro-title"):
                    label("ЗА СТОЛОМ ЕСТЬ МЕСТО ДЛЯ ЛЮБОЙ ВЕРСИИ", "eyebrow")
                    label("Внимание,", "hero-title").props("role=heading aria-level=1")
                    label("вопрос.", "hero-title hero-accent")
                label(
                    "Шесть характеров. Минута на обсуждение.\n"
                    "Наблюдайте, как ассоциации становятся версиями,\n"
                    "а версии — ответом команды.",
                    "intro-copy",
                )
            with ui.card().classes("question-panel"):
                with ui.column().classes("question-form").mark("question-form") as self.form:
                    with ui.row().classes("section-top w-full"):
                        label("ВАШ ВОПРОС", "eyebrow")
                        ui.button("Попробовать пример ↗", on_click=self.example).props(
                            "flat no-caps"
                        ).classes("text-button")
                    self.question = (
                        ui.textarea(placeholder="Введите вопрос ЧГК. Команда уже за столом…")
                        .props('outlined autogrow rows=3 maxlength=10000 aria-label="Ваш вопрос"')
                        .classes("question-input w-full")
                        .mark("question")
                    )
                    with ui.row().classes("question-bottom w-full"):
                        hint = (
                            "Демо без GigaChat: заданный сценарий про маятник. "
                            "Нажмите «Попробовать пример»."
                            if self.settings.mode == "demo"
                            else "Текстовый вопрос · до 10 000 символов"
                        )
                        label(hint, "form-hint")
                        self.start_button = (
                            ui.button("Начать обсуждение →", on_click=self.start)
                            .props("no-caps")
                            .classes("primary-button")
                            .mark("start")
                        )
                    self.form_error = label("", "error", mark="form-error").props("role=alert")
                    self.form_error.set_visibility(False)
                with ui.column().classes("w-full").mark("played-question") as self.played_question:
                    label("ВОПРОС НА СТОЛЕ", "eyebrow")
                    self.question_text = label("", "question-text", mark="question-text")
                self.played_question.set_visibility(False)
            with ui.element("section").classes("team-section"):
                with ui.row().classes("section-top"):
                    heading("Команда за столом")
                    label("У каждого — свой взгляд", "subtle")
                with ui.element("div").classes("team-grid"):
                    for index, player in enumerate(TEAM, 1):
                        with ui.card().classes("player-card").mark(f"player-{player.id}") as card:
                            self.player_cards[player.id] = card
                            with ui.row().classes("player-top"):
                                label(player.name[0], "avatar").style(
                                    f"--player-color:{player.color}"
                                )
                                label(f"{index:02d}", "player-number")
                            label(player.name, "player-name")
                            label(player.label, "player-role")
                            label(" · ".join(player.strengths), "player-topics")
                            with ui.expansion("Профиль игрока").classes("player-detail"):
                                label("Сильные темы: " + ", ".join(player.strengths))
                                label("Слабые темы: " + ", ".join(player.weaknesses))
                            self.player_status[player.id] = label("За столом", "player-state")
            with ui.element("section").classes("game-section"):
                with ui.card().classes("discussion-panel"):
                    with ui.row().classes("section-top discussion-heading"):
                        with ui.column().classes("heading-group"):
                            label("РАЗГОВОР ЗА СТОЛОМ", "eyebrow")
                            heading("Ход обсуждения")
                        self.live_label = label(
                            "Ожидаем вопрос", "live-indicator", mark="live-label"
                        )
                    with ui.scroll_area(on_scroll=self.track_scroll).classes(
                        "transcript-scroll"
                    ) as scroll:
                        self.transcript_scroll = scroll
                        self.transcript = (
                            ui.column()
                            .classes("transcript")
                            .props('role=log aria-live=polite aria-label="Реплики команды"')
                            .mark("transcript")
                        )
                    with self.transcript:
                        with ui.column().classes("empty-state") as self.empty:
                            label("✳", "table-symbol")
                            label("Пока здесь тихо", "empty-heading")
                            label(
                                "Задайте вопрос — капитан начнёт обсуждение,\n"
                                "а каждый игрок добавит свою зацепку."
                            )
                    self.speaker_status = label("", "speaker-status", mark="speaker-status").props(
                        "role=status"
                    )
                    self.speaker_status.set_visibility(False)
                    self.final = ui.column().classes("final-result").mark("final-result")
                    self.final.set_visibility(False)
                with ui.element("aside").classes("game-aside"):
                    with ui.card().classes("timer-panel").mark("timer-panel"):
                        label("ВРЕМЯ КОМАНДЫ", "eyebrow")
                        with ui.element("div").classes("timer-wrap"):
                            self.progress = (
                                ui.circular_progress(
                                    value=1,
                                    size="145px",
                                    color="#8d9c77",
                                    show_value=False,
                                )
                                .props('track-color="grey-4" thickness=0.025')
                                .classes("timer-circle")
                            )
                            with ui.column().classes("timer-digits"):
                                self.time_label = label("01:00", "timer-value", mark="timer")
                                self.phase_label = label(
                                    "до начала", "phase-label", mark="phase-label"
                                )
                        self.time_note = label(
                            "50 секунд на версии.\n10 — на окончательный ответ.", "timer-note"
                        )
                    with ui.card().classes("versions-panel"):
                        with ui.row().classes("section-top"):
                            heading("Версии")
                            self.version_count = label("0", "count")
                        self.versions = ui.column().classes("w-full").mark("versions")
                        with self.versions:
                            label("Первые версии появятся\nво время обсуждения.", "versions-empty")
                    self.sources_panel = ui.card().classes("sources-panel").mark("sources-panel")
                    with self.sources_panel:
                        heading("Проверка фактов")
                        self.sources_box = ui.column().classes("w-full").mark("sources")
                    self.sources_panel.set_visibility(False)
                    self.export_button = (
                        ui.button("Сохранить обсуждение ↓", on_click=self.download)
                        .props("outline no-caps")
                        .classes("secondary-button")
                        .mark("export")
                    )
                    self.new_button = (
                        ui.button("Новый вопрос ↗", on_click=self.new_game)
                        .props("outline no-caps")
                        .classes("secondary-button")
                        .mark("new-game")
                    )
                    self.export_button.set_visibility(False)
                    self.new_button.set_visibility(False)
            self.connection_status = label("", "connection-status", mark="connection-status").props(
                "role=status"
            )
        with ui.element("footer").classes("footer"):
            label("минута. Искусство думать вместе.")
            label("Версия — ещё не правильный ответ.")

    def restore(self):
        self.tab_storage = nicegui_app.storage.tab
        session_id = self.tab_storage.get("session_id")
        if not session_id:
            return
        self.session = self.registry.get(session_id)
        if self.session is None:
            self.connection_status.set_text(
                "Сессия недоступна: удалена или сервер был перезапущен."
            )
            self.tab_storage.pop("session_id", None)
            return
        self.form.set_visibility(False)
        self.played_question.set_visibility(True)
        self.question_text.set_text(self.session.question)
        self.refresh()

    def example(self):
        self.question.set_value(EXAMPLE)

    async def start(self):
        if self.session and self.session.phase != "finished":
            return
        self.start_button.disable()
        self.form_error.set_visibility(False)
        try:
            question = SessionInput(question=self.question.value or "").question
            if self.settings.mode == "real" and not self.settings.real_ready:
                self.show_error("GigaChat не настроен. Добавьте серверные credentials в .env.")
                return
            self.session = await self.registry.create(question)
            self.tab_storage["session_id"] = self.session.id
            self.last_event = -1
            self.form.set_visibility(False)
            self.played_question.set_visibility(True)
            self.question_text.set_text(question)
            self.refresh()
        except ValidationError:
            self.show_error("Введите вопрос длиной от 1 до 10 000 символов.")
        except CapacityError:
            self.show_error("Все места заняты. Дождитесь окончания одной из игр.")
        finally:
            self.start_button.enable()

    def show_error(self, text):
        self.form_error.set_text(text)
        self.form_error.set_visibility(True)

    def refresh(self):
        if not self.client.has_socket_connection or not self.session:
            return
        session = self.registry.get(self.session.id)
        if session is None:
            self.connection_status.set_text(
                "Сессия недоступна: удалена или сервер был перезапущен."
            )
            self.new_button.set_visibility(True)
            self.export_button.set_visibility(False)
            return
        remaining = max(0, 60 - session.elapsed())
        seconds = max(0, int(remaining + 0.999))
        self.time_label.set_text(f"{seconds // 60:02d}:{seconds % 60:02d}")
        self.progress.set_value(remaining / 60)
        self.phase_label.set_text(PHASE_LABELS[session.phase])
        if session.phase == "finished":
            self.time_note.set_text(f"Команда обсуждала {round(session.elapsed())} с")
        if session.event_id == self.last_event:
            return
        self.last_event = session.event_id
        snapshot = session.snapshot()
        self.live_label.set_text(
            "Игра завершена" if session.phase == "finished" else "Идёт обсуждение"
        )
        active = session.active_speaker
        for player_id, card in self.player_cards.items():
            card.classes(
                add="active" if player_id == active else "",
                remove="active" if player_id != active else "",
            )
            self.player_status[player_id].set_text(
                "Сейчас говорит" if player_id == active else "За столом"
            )
        self.speaker_status.set_visibility(bool(active) and session.phase != "finished")
        if active:
            operation = "проверяет факт…" if session.search_status else "готовит реплику…"
            self.speaker_status.set_text(f"{PLAYERS[active].name} {operation}")
        self.draw_utterances(snapshot)
        self.draw_versions(snapshot)
        self.draw_sources(snapshot)
        finished = session.phase == "finished"
        self.export_button.set_visibility(finished)
        self.new_button.set_visibility(finished)
        self.final.set_visibility(finished)
        if finished:
            self.draw_final(snapshot)

    def draw_utterances(self, snapshot):
        previous_count = self.rendered_utterances
        if snapshot["utterances"] and not self.rendered_utterances:
            self.transcript.clear()
        sources = {s["id"]: s for s in snapshot["sources"]}
        with self.transcript:
            for utterance in snapshot["utterances"][self.rendered_utterances :]:
                player = PLAYERS[utterance["speaker_id"]]
                with ui.row().classes("utterance").mark(f"utterance-{utterance['index']}"):
                    label(player.name[0], "avatar small").style(f"--player-color:{player.color}")
                    with ui.column().classes("utterance-body"):
                        with ui.row().classes("utterance-meta"):
                            label(player.name, "utterance-name")
                            label(player.label, "utterance-role")
                            label(f"{int(utterance['elapsed'])} с", "utterance-time")
                        label(utterance["text"], "utterance-text")
                        for source_id in utterance["source_ids"]:
                            if source_id in sources:
                                source_link(sources[source_id])
        self.rendered_utterances = len(snapshot["utterances"])
        if self.follow_latest and self.rendered_utterances > previous_count:
            self.transcript_scroll.scroll_to(percent=1)

    def track_scroll(self, event):
        remaining = event.vertical_size - event.vertical_container_size - event.vertical_position
        self.follow_latest = remaining < 100

    def draw_versions(self, snapshot):
        self.version_count.set_text(str(len(snapshot["hypotheses"])))
        self.versions.clear()
        with self.versions:
            if not snapshot["hypotheses"]:
                label("Первые версии появятся во время обсуждения.", "versions-empty")
            for hypothesis in snapshot["hypotheses"]:
                with ui.column().classes(f"version {hypothesis['status']}"):
                    prefix = "Ведущая · " if hypothesis["id"] == snapshot["leading_id"] else ""
                    label(prefix + STATUS_LABELS[hypothesis["status"]], "version-status")
                    label(hypothesis["text"], "version-title")
                    if hypothesis["reasons"]:
                        label(hypothesis["reasons"][-1]["text"], "version-reason")

    def draw_sources(self, snapshot):
        self.sources_panel.set_visibility(bool(snapshot["sources"]))
        self.sources_box.clear()
        with self.sources_box:
            for source in snapshot["sources"]:
                with ui.column().classes("source-entry"):
                    source_link(source)
                    label(source["snippet"], "source-snippet")

    def draw_final(self, snapshot):
        self.final.clear()
        with self.final:
            answer = snapshot["final_answer"]
            if answer:
                label("ОТВЕТ КОМАНДЫ", "eyebrow")
                label(answer["answer"], "answer-title")
                label(answer["explanation"], "answer-explanation")
                label("Отвечает " + PLAYERS[answer["speaker_id"]].name, "answerer")
            else:
                label("ОКОНЧАТЕЛЬНЫЙ ОТВЕТ НЕ ПОЛУЧЕН", "eyebrow")
                label("Окончательный ответ не получен", "answer-title")
                leading = next(
                    (h for h in snapshot["hypotheses"] if h["id"] == snapshot["leading_id"]), None
                )
                if leading:
                    label("Предварительная версия: " + leading["text"], "answer-explanation")
                label("Модель не успела или не смогла завершить ответ.", "answer-explanation")
            if snapshot["mode"] == "demo":
                label(
                    "Учебный сценарий: реплики и ответ заданы заранее. "
                    "Это не решение введённого вопроса моделью.",
                    "demo-result-note",
                )

    def download(self):
        if not self.session:
            return
        ui.download.content(
            json.dumps(self.session.export(), ensure_ascii=False, indent=2).encode("utf-8"),
            filename=f"chgk-{self.session.id}.json",
            media_type="application/json",
        )

    def new_game(self):
        self.session = None
        self.tab_storage.pop("session_id", None)
        self.last_event = -1
        self.rendered_utterances = 0
        self.follow_latest = True
        self.form.set_visibility(True)
        self.played_question.set_visibility(False)
        self.question.set_value("")
        self.form_error.set_visibility(False)
        self.final.set_visibility(False)
        self.export_button.set_visibility(False)
        self.new_button.set_visibility(False)
        self.sources_panel.set_visibility(False)
        self.speaker_status.set_visibility(False)
        self.connection_status.set_text("")
        self.time_label.set_text("01:00")
        self.phase_label.set_text("до начала")
        self.progress.set_value(1)
        self.time_note.set_text("50 секунд на версии.\n10 — на окончательный ответ.")
        self.live_label.set_text("Ожидаем вопрос")
        self.version_count.set_text("0")
        self.versions.clear()
        self.transcript.clear()
        with self.transcript:
            label("Задайте следующий вопрос — команда готова.", "empty-state")
        for player_id, card in self.player_cards.items():
            card.classes(remove="active")
            self.player_status[player_id].set_text("За столом")


def mount_ui(api, settings: Settings):
    @ui.page("/", title="Минута · команда ЧГК", language="ru")
    async def index():
        ui.add_css((Path(__file__).parent / "static/style.css").read_text())
        page = GamePage(api.state.registry, settings)
        await ui.context.client.connected()
        page.restore()

    ui.run_with(
        api,
        title="Минута · команда ЧГК",
        language="ru",
        favicon="✳",
        reconnect_timeout=10,
        show_welcome_message=False,
    )
