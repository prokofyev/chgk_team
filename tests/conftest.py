import asyncio
from datetime import UTC, datetime, timedelta


class ManualClock:
    def __init__(self):
        self.value = 0.0
        self.sleepers = []

    def now(self):
        return self.value

    def utc(self):
        return datetime(2026, 10, 1, tzinfo=UTC) + timedelta(seconds=self.value)

    async def sleep(self, seconds):
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        future = asyncio.get_running_loop().create_future()
        self.sleepers.append((self.value + seconds, future))
        await future

    async def advance(self, seconds):
        self.value += seconds
        for deadline, future in tuple(self.sleepers):
            if deadline <= self.value and not future.done():
                future.set_result(None)
        await settle()


async def settle():
    for _ in range(50):
        await asyncio.sleep(0)


class ScaledClock:
    """Speed up integration tests without exposing a shorter game setting to users."""

    def __init__(self, factor=1000):
        self.factor = factor
        self.origin = asyncio.get_running_loop().time()

    def now(self):
        return (asyncio.get_running_loop().time() - self.origin) * self.factor

    def utc(self):
        return datetime(2026, 10, 1, tzinfo=UTC) + timedelta(seconds=self.now())

    async def sleep(self, seconds):
        await asyncio.sleep(max(0, seconds) / self.factor)
