import asyncio
import time
from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...
    def utc(self) -> datetime: ...
    async def sleep(self, seconds: float) -> None: ...


class RealClock:
    def now(self) -> float:
        return time.monotonic()

    def utc(self) -> datetime:
        return datetime.now(UTC)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0, seconds))
