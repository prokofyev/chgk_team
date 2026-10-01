import asyncio
import json
from dataclasses import dataclass
from typing import Protocol

from chgk_team.agents import messages_for
from chgk_team.config import Settings
from chgk_team.domain import AgentReply, AnswerDraft, Player


class ContextLimitError(ValueError):
    pass


@dataclass(frozen=True)
class ModelRequest:
    player: Player
    context: dict
    final: bool = False


class ModelProvider(Protocol):
    async def generate(self, request: ModelRequest) -> str: ...
    async def close(self) -> None: ...


class GigaChatProvider:
    def __init__(self, settings: Settings, *, client=None):
        self.settings = settings
        self.semaphore = asyncio.Semaphore(settings.model_concurrency)
        if client is None:
            from langchain_gigachat import GigaChat

            client = GigaChat(
                credentials=settings.gigachat_credentials.get_secret_value(),
                scope=settings.gigachat_scope,
                model=settings.model,
                verify_ssl_certs=settings.verify_ssl,
                ca_bundle_file=settings.ca_bundle if settings.verify_ssl else None,
                timeout=10,
                max_retries=0,
                streaming=False,
                max_tokens=settings.output_tokens,
            )
        self.client = client
        self.structured_clients = {}

    def prepare(self, request: ModelRequest) -> list:
        messages = messages_for(request.player, request.context, final=request.final)
        # UTF-8 byte count is a deliberately conservative offline token upper bound.
        # No remote token-count request consumes the minute.
        budget = self.settings.context_tokens - self.settings.output_tokens
        size = sum(len(m.content.encode("utf-8")) for m in messages)
        if size > budget:
            context = {
                **request.context,
                "sources": [{**source, "snippet": ""} for source in request.context["sources"]],
            }
            messages = messages_for(request.player, context, final=request.final)
            size = sum(len(m.content.encode("utf-8")) for m in messages)
        if size > budget:
            raise ContextLimitError("Context exceeds the configured model window")
        return messages

    async def generate(self, request: ModelRequest) -> str:
        messages = self.prepare(request)
        timeout = 10 if request.final else self.settings.turn_timeout
        async with asyncio.timeout(timeout), self.semaphore:
            schema = AnswerDraft if request.final else AgentReply
            if request.final not in self.structured_clients:
                self.structured_clients[request.final] = self.client.with_structured_output(
                    schema, method="function_calling"
                )
            response = await self.structured_clients[request.final].ainvoke(
                messages, config={"callbacks": []}
            )
        if not isinstance(response, schema):
            raise ValueError("A structured player response is required")
        return response.model_dump_json()

    async def close(self) -> None:
        # The pinned wrapper exposes the async SDK client through this property.
        if "_client" in self.client.__dict__:
            await self.client._client.aclose()


class DemoProvider:
    """Scripted staging rehearsal, explicitly not a general question solver."""

    def __init__(self, delay: float = 1.6):
        self.delay = delay

    async def generate(self, request: ModelRequest) -> str:
        await asyncio.sleep(self.delay)
        context = request.context
        player = request.player.id
        hypotheses = context["hypotheses"]
        count = len(context["utterances"])
        sources = context["sources"]
        if request.final:
            result = {
                "answer": "Маятник Фуко",
                "explanation": "В учебном сценарии меняется направление колебаний "
                "относительно пола: это демонстрирует вращение Земли.",
            }
        elif player == "captain":
            if count == 0:
                result = {
                    "utterance": "Ищем не прибор для измерения времени, а наглядное "
                    "доказательство движения. Ника, дай ассоциации; Михаил, вспомни историю.",
                    "next_speakers": ["ideas", "humanities"],
                }
            else:
                second = hypotheses[1]["id"] if len(hypotheses) > 1 else None
                result = {
                    "utterance": "Версия с часами не объясняет опыт. Сохраняем маятник. "
                    "Софья, проверь физику; Даниил, сверь все условия.",
                    "answerer": "science",
                    "next_speakers": ["science", "logic"],
                }
                if second:
                    result["leading_id"] = second
                    result["changes"] = [
                        {
                            "action": "reject",
                            "hypothesis_id": hypotheses[0]["id"],
                            "reason": "Часы не доказывают вращение Земли.",
                        },
                        {
                            "action": "support",
                            "hypothesis_id": second,
                            "reason": "Версия соответствует условию о наглядном опыте.",
                        },
                    ]
                if count >= 6:
                    result["next_speakers"] = ["intuition", "humanities"]
                    result["utterance"] = (
                        "Ася, есть ли другой образ? Михаил, проверь связь "
                        "с названием. Отвечать поручаю Софье."
                    )
                if count >= 9:
                    result["ready"] = True
                    result["utterance"] = (
                        "Останавливаемся на маятнике Фуко. Версия "
                        "объясняет опыт и движение Земли. Софья, отвечай от команды."
                    )
        elif player == "ideas":
            result = {
                "utterance": "Если что-то качается в здании, это могут быть часы. "
                "Но слово «доказательство» заставляет искать физический опыт.",
                "changes": [
                    {
                        "action": "new",
                        "text": "Большие часы",
                        "reason": "Ассоциация с качающимся механизмом.",
                    }
                ],
            }
        elif player == "humanities":
            if len(hypotheses) < 2:
                result = {
                    "utterance": "Фуко показывал опыт с маятником в Париже. "
                    "Это как раз наглядное доказательство вращения Земли.",
                    "changes": [
                        {
                            "action": "new",
                            "text": "Маятник Фуко",
                            "reason": "Исторический опыт с вращением плоскости колебаний.",
                        }
                    ],
                }
            else:
                result = {
                    "utterance": "Название связано с физиком Леоном Фуко. "
                    "Историческая и физическая зацепки сходятся."
                }
        elif player == "science":
            result = {
                "utterance": "Подвес почти сохраняет плоскость колебаний, "
                "а Земля под ним поворачивается. Проверю название опыта."
            }
            if not context.get("search_notice") and not sources:
                result["search_query"] = "Маятник Фуко"
            else:
                result["utterance"] = (
                    "По физике маятник подходит: наблюдатель видит "
                    "поворот плоскости колебаний относительно Земли."
                )
                result["source_ids"] = [s["id"] for s in sources[:1]]
        elif player == "logic":
            result = {
                "utterance": "Часы измеряют время, но не доказывают вращение. "
                "Маятник объясняет именно изменение направления относительно пола."
            }
            if len(hypotheses) > 1:
                result["changes"] = [
                    {
                        "action": "argument",
                        "hypothesis_id": hypotheses[1]["id"],
                        "reason": "Все условия опыта согласуются с вращением плоскости колебаний.",
                    }
                ]
        else:
            result = {
                "utterance": "Образ такой: маятник рисует линию, а весь зал "
                "медленно поворачивается под ней. Мне тоже кажется, что это Фуко."
            }
        return json.dumps(result, ensure_ascii=False)

    async def close(self) -> None:
        pass
