import json

import pytest
from conftest import ManualClock

from chgk_team.config import Settings
from chgk_team.evaluation import EvalQuestion, evaluate, load_questions, normalize_answer
from chgk_team.graph import DiscussionEngine
from chgk_team.providers import DemoProvider
from chgk_team.search import DemoSearch


@pytest.mark.parametrize("actual,expected", [(" Ё  ", "е"), ("Маятник  ФУКО", "маятник фуко")])
def test_answer_normalization(actual, expected):
    assert normalize_answer(actual) == expected


async def test_expected_answers_never_reach_model():
    captured = []

    class AnsweringModel(DemoProvider):
        async def generate(self, request):
            captured.append(json.dumps(request.context, ensure_ascii=False))
            if request.final:
                return '{"answer":"  Ё  ","explanation":"Две точки"}'
            return '{"utterance":"Есть версия","ready":true}'

    question = EvalQuestion(
        id="x",
        question="Две точки?",
        answer="PRIVATE EXPECTED",
        accepted_answers=["е"],
        source="Private annotation",
    )
    engine = DiscussionEngine(Settings(_env_file=None), AnsweringModel(0), DemoSearch())
    report, transcripts = await evaluate([question], engine, clock=ManualClock())
    assert all("PRIVATE EXPECTED" not in content for content in captured)
    assert all("Private annotation" not in content for content in captured)
    assert report["summary"]["exact_matches"] == 1
    assert report["results"][0]["expected"] == "PRIVATE EXPECTED"
    assert report["mode"] == "demo" and report["demo_notice"]
    assert report["results"][0]["model_calls"] == 2
    assert transcripts[0]["prompt_version"]
    await engine.close()


async def test_evaluation_no_answer_and_errors():
    class FailingModel(DemoProvider):
        async def generate(self, request):
            if request.final:
                raise ValueError("provider failure")
            return '{"utterance":"Готовы","ready":true}'

    engine = DiscussionEngine(Settings(_env_file=None), FailingModel(0), DemoSearch())
    report, _ = await evaluate(
        [EvalQuestion(id="x", question="?", answer="X")], engine, clock=ManualClock()
    )
    assert report["summary"]["no_answers"] == 1
    assert report["summary"]["exact_match_rate"] == 0
    assert report["results"][0]["errors"]
    await engine.close()


def test_load_questions_and_duplicate_ids(tmp_path):
    path = tmp_path / "questions.jsonl"
    record = {"id": "x", "question": "?", "answer": "X"}
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert load_questions(path)[0].id == "x"
    path.write_text((json.dumps(record) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="Повторный ID"):
        load_questions(path)
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="строке 1"):
        load_questions(path)


def test_sample_dataset_is_valid():
    from pathlib import Path

    items = load_questions(Path("eval/sample.jsonl"))
    assert len(items) == 2
    assert all(item.source and item.usage_rights for item in items)
