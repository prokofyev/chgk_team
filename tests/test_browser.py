"""Opt-in browser integration: CHGK_BROWSER_TESTS=1 pytest tests/test_browser.py."""

import json
import os
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import expect, sync_playwright

pytestmark = pytest.mark.skipif(
    os.environ.get("CHGK_BROWSER_TESTS") != "1", reason="Opt-in local Chromium integration"
)


@pytest.fixture(scope="module")
def server():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    environment = {**os.environ, "CHGK_MODE": "demo", "CHGK_DEMO_DELAY": "0.3"}
    process = subprocess.Popen(
        [
            str(Path(".venv/bin/python").absolute()),
            "-m",
            "uvicorn",
            "chgk_team.api:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            if process.poll() is not None:
                pytest.fail("Local server failed to start")
            try:
                if httpx.get(url + "/api/team").status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.05)
        else:
            pytest.fail("Local server startup timed out")
        yield url
    finally:
        process.terminate()
        process.wait(timeout=10)


@pytest.fixture
def page(server):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(base_url=server, viewport={"width": 1440, "height": 1100})
        context.route("https://**", lambda route: route.abort())  # Fully offline browser rehearsal.
        page = context.new_page()
        page.goto(server)
        expect(page.locator(".player-card")).to_have_count(6)
        yield page
        context.close()
        browser.close()


def start_example(page):
    page.get_by_role("button", name="Попробовать пример").click()
    question = page.locator(".question-input textarea")
    expect(question).not_to_have_value("")
    question.focus()
    page.keyboard.press("Tab")
    expect(page.get_by_role("button", name="Начать обсуждение")).to_be_focused()
    page.keyboard.press("Enter")
    expect(page.locator(".question-text")).to_be_visible()


def download_export(page):
    with page.expect_download() as download:
        page.get_by_role("button", name="Сохранить обсуждение").click()
    item = download.value
    assert item.suggested_filename.startswith("chgk-")
    return json.loads(Path(item.path()).read_text(encoding="utf-8"))


def test_full_nicegui_game_export_and_mobile_layout(page, server):
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    Path("artifacts").mkdir(exist_ok=True)
    page.screenshot(path="artifacts/nicegui-desktop-idle.png", full_page=True)
    start_example(page)
    expect(page.locator(".final-result")).to_be_visible(timeout=20000)
    expect(page.locator(".final-result")).to_contain_text("Маятник Фуко")
    expect(page.locator(".final-result")).to_contain_text("Отвечает Софья")
    expect(page.locator(".sources-panel")).to_be_visible()
    expect(page.locator(".version.rejected")).to_have_count(1)
    expect(page.locator(".version.supported")).to_have_count(1)
    expect(page.locator(".utterance")).to_have_count(10)
    exported = download_export(page)
    assert exported["mode"] == "demo"
    assert exported["final_answer"]["speaker_id"] == "science"
    assert len(exported["team"]) == 6
    assert exported["metrics"]["searches"] == 1
    assert "credentials" not in json.dumps(exported)
    assert page.request.get(f"/api/sessions/{exported['id']}/export").json()["id"] == exported["id"]
    page.screenshot(path="artifacts/nicegui-desktop-finished.png", full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    page.screenshot(path="artifacts/nicegui-mobile-finished.png", full_page=True)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    page.get_by_role("button", name="Новый вопрос").click()
    expect(page.locator(".question-form")).to_be_visible()
    expect(page.locator(".question-input textarea")).to_have_value("")
    assert not errors


def test_nicegui_reload_and_reconnect_preserve_session(page, server):
    start_example(page)
    expect(page.locator(".utterance").first).to_be_visible()
    timer_before = page.locator(".timer-value").inner_text()
    page.reload()
    expect(page.locator(".question-text")).to_be_visible()
    assert page.locator(".timer-value").inner_text() <= timer_before
    page.context.set_offline(True)
    page.wait_for_timeout(250)
    page.context.set_offline(False)
    expect(page.locator(".final-result")).to_be_visible(timeout=20000)
    expect(page.locator(".utterance")).to_have_count(10)
    exported = download_export(page)
    assert exported["metrics"]["model_calls"] == 12
    page.reload()
    expect(page.locator(".final-result")).to_be_visible()
    assert download_export(page)["id"] == exported["id"]


def test_nicegui_question_renders_as_text(page):
    payload = (
        '<img src=x onerror="window.compromised=true"><script>window.compromised=true</script>'
    )
    page.locator(".question-input textarea").fill(payload)
    page.get_by_role("button", name="Начать обсуждение").click()
    expect(page.locator(".question-text")).to_have_text(payload)
    expect(page.locator(".final-result")).to_be_visible(timeout=20000)
    assert page.evaluate("window.compromised === undefined")
    assert page.locator(".question-text img, .transcript img, .sources-panel img").count() == 0
