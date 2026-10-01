import re
from enum import StrEnum
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=350)]
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=80)]

_FIELD_NAMES = (
    "utterance|changes|source_ids|search_query|next_speakers|answerer|leading_id|ready|"
    "hypothesis_id|proposed_by|speaker_id|explanation|answer"
)
_PLAYER_IDS = "captain|ideas|humanities|science|logic|intuition"
_CONTROL_TEXT = re.compile(
    rf"\b(?:{_FIELD_NAMES})\b[\"']?\s*[:=]"
    rf"|[\"']\s*,\s*[\"'](?:{_FIELD_NAMES})\b"
    rf"|\b(?:следующие|следующий|отвечающий|отвечает|назначаю отвечающего)"
    rf"\s*:\s*(?:{_PLAYER_IDS})\b"
    r"|\bготовность\s*:\s*(?:true|false)\b",
    re.IGNORECASE,
)


def validate_public_text(value: str) -> str:
    # Reject clear serialization/control fragments; preserve natural captain directions.
    # Do not strip text and accidentally turn a malformed reply into a different answer.
    if _CONTROL_TEXT.search(value):
        raise ValueError("Public text contains structured control data")
    return value


PublicText = Annotated[ShortText, AfterValidator(validate_public_text)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class Role(StrEnum):
    CAPTAIN = "captain"
    IDEAS = "ideas"
    SCHOLAR = "scholar"
    LOGIC = "logic"
    INTUITION = "intuition"


class Player(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: Identifier
    name: str
    role: Role
    label: str
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    instruction: str
    color: str


class Source(StrictModel):
    id: Identifier
    title: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    url: str
    snippet: Annotated[str, StringConstraints(max_length=600)]

    @model_validator(mode="after")
    def wikipedia_url(self):
        parsed = urlsplit(self.url)
        if parsed.scheme != "https" or parsed.netloc != "ru.wikipedia.org":
            raise ValueError("Only retrieved Wikipedia links are supported")
        return self


class HypothesisChange(StrictModel):
    action: Literal["new", "argument", "support", "reject"]
    hypothesis_id: Identifier | None = None
    text: PublicText | None = Field(
        default=None, description="Короткая формулировка новой версии без JSON и служебных полей."
    )
    reason: PublicText = Field(description="Понятный зрителю аргумент или причина решения.")
    source_ids: list[Identifier] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def action_fields(self):
        if self.action == "new" and (not self.text or self.hypothesis_id is not None):
            raise ValueError("New hypotheses need text and no identifier")
        if self.action != "new" and not self.hypothesis_id:
            raise ValueError("An existing hypothesis identifier is required")
        return self


class AgentReply(StrictModel):
    """Публичная реплика игрока, изменения версий и разрешённые решения по его роли."""

    utterance: PublicText = Field(
        description="Живая реплика игрока для зрителя. Обращения к коллегам по именам. "
        "Без идентификаторов, имён полей и служебных сводок; управление — в отдельных полях."
    )
    changes: list[HypothesisChange] = Field(default_factory=list, max_length=4)
    source_ids: list[Identifier] = Field(default_factory=list, max_length=6)
    search_query: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
        | None
    ) = None
    next_speakers: list[Identifier] = Field(
        default_factory=list,
        max_length=2,
        description="Служебная очередь: до двух ID игроков из состава. Только капитан.",
    )
    answerer: Identifier | None = Field(
        default=None, description="Служебное назначение: ID отвечающего игрока. Только капитан."
    )
    leading_id: Identifier | None = Field(
        default=None, description="ID существующей ведущей версии. Только капитан."
    )
    ready: bool = Field(
        default=False, description="Служебный сигнал готовности дать ответ. Только капитан."
    )


class AnswerDraft(StrictModel):
    """Окончательный ответ команды с кратким обоснованием по условиям вопроса."""

    answer: PublicText = Field(description="Только ответ на вопрос, без JSON и служебных полей.")
    explanation: PublicText = Field(description="Краткое обоснование ответа для зрителя.")
    source_ids: list[Identifier] = Field(default_factory=list, max_length=6)


class FinalAnswer(AnswerDraft):
    speaker_id: Identifier


class Hypothesis(StrictModel):
    id: Identifier
    text: PublicText
    proposed_by: Identifier
    status: Literal["proposed", "supported", "rejected"] = "proposed"
    reasons: list[dict] = Field(default_factory=list)
    source_ids: list[Identifier] = Field(default_factory=list)


class Utterance(StrictModel):
    index: int
    speaker_id: Identifier
    text: PublicText
    elapsed: float
    source_ids: list[Identifier] = Field(default_factory=list)


class SessionInput(StrictModel):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=10000)
    ]


class Event(StrictModel):
    id: int
    session_id: str
    type: str
    server_time: str
    data: dict
