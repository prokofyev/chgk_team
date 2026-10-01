import argparse
import asyncio
import json
from pathlib import Path

from pydantic import Field

from chgk_team.agents import PROMPT_VERSION
from chgk_team.clock import Clock, RealClock
from chgk_team.config import Settings
from chgk_team.domain import SessionInput, ShortText
from chgk_team.graph import DiscussionEngine
from chgk_team.providers import DemoProvider, GigaChatProvider
from chgk_team.search import DemoSearch, WikipediaSearch
from chgk_team.sessions import Session


class EvalQuestion(SessionInput):
    id: ShortText
    answer: ShortText
    accepted_answers: list[ShortText] = Field(default_factory=list, max_length=20)
    source: str | None = None
    usage_rights: str | None = None


def load_questions(path: Path) -> list[EvalQuestion]:
    questions = []
    seen = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                item = EvalQuestion.model_validate_json(line)
            except ValueError:
                raise ValueError(f"Некорректная запись JSONL в строке {line_number}") from None
            if item.id in seen:
                raise ValueError(f"Повторный ID в строке {line_number}")
            seen.add(item.id)
            questions.append(item)
    if not questions:
        raise ValueError("Набор вопросов пуст")
    return questions


def normalize_answer(answer: str) -> str:
    return " ".join(answer.casefold().replace("ё", "е").split())


async def evaluate(
    questions: list[EvalQuestion], engine: DiscussionEngine, *, clock: Clock | None = None
) -> tuple[dict, list[dict]]:
    results, transcripts = [], []
    for item in questions:
        # Expected answers never cross the boundary into Session, graph, or model prompts.
        session = Session(item.question, engine.settings, clock or RealClock())
        await engine.run(session)
        export = session.export()
        transcripts.append(export)
        answer = session.final_answer.answer if session.final_answer else None
        accepted = {normalize_answer(a) for a in [item.answer, *item.accepted_answers]}
        match = bool(answer and normalize_answer(answer) in accepted)
        results.append(
            {
                "id": item.id,
                "question": item.question,
                "expected": item.answer,
                "accepted_answers": item.accepted_answers,
                "answer": answer,
                "exact_match": match,
                "no_answer": answer is None,
                "finish_reason": session.finish_reason,
                "duration_seconds": session.elapsed(),
                "utterance_count": len(session.utterances),
                "model_calls": session.model_calls,
                "searches": session.search_count,
                "call_latencies": session.call_metrics,
                "errors": session.errors,
                "source": item.source,
                "usage_rights": item.usage_rights,
                "session_id": session.id,
                "mode": engine.settings.mode,
            }
        )
    report = {
        "format_version": 1,
        "mode": engine.settings.mode,
        "model": engine.settings.model if engine.settings.mode == "real" else "scripted-demo",
        "prompt_version": PROMPT_VERSION,
        "configuration": engine.settings.public(),
        "metric": "Normalized exact match against accepted answers; manual review recommended.",
        "demo_notice": "Scripted rehearsal, not model accuracy."
        if engine.settings.mode == "demo"
        else None,
        "summary": {
            "total": len(results),
            "exact_matches": sum(r["exact_match"] for r in results),
            "no_answers": sum(r["no_answer"] for r in results),
            "exact_match_rate": sum(r["exact_match"] for r in results) / len(results),
        },
        "results": results,
    }
    return report, transcripts


async def run_cli(args) -> dict:
    questions = load_questions(args.questions)[: args.limit]
    settings = Settings(mode=args.mode)
    if settings.mode == "real" and not settings.real_ready:
        raise ValueError("Для реального прогона настройте CHGK_GIGACHAT_CREDENTIALS на сервере")
    model = (
        DemoProvider(settings.demo_delay) if settings.mode == "demo" else GigaChatProvider(settings)
    )
    search = DemoSearch() if settings.mode == "demo" else WikipediaSearch()
    engine = DiscussionEngine(settings, model, search)
    try:
        report, transcripts = await evaluate(questions, engine)
    finally:
        await engine.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    directory = args.output.parent / (args.output.stem + "-transcripts")
    directory.mkdir(exist_ok=True)
    for index, transcript in enumerate(transcripts, 1):
        name = f"{index:03d}-{transcript['id']}.json"
        path = directory / name
        path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
        report["results"][index - 1]["transcript"] = str(path)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Оценка ответов команды на локальном JSONL")
    parser.add_argument("--questions", type=Path, default=Path("eval/sample.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    parser.add_argument("--mode", choices=["demo", "real"], default="demo")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit должен быть положительным")
    try:
        report = asyncio.run(run_cli(args))
    except (ValueError, OSError) as error:
        parser.exit(2, f"Ошибка: {error}\n")
    print(json.dumps({"mode": report["mode"], **report["summary"]}, ensure_ascii=False))
    print(f"Отчёт: {args.output}")
