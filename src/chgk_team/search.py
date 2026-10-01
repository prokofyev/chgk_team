import asyncio
import html
import re
from typing import Protocol
from urllib.parse import quote

import httpx

from chgk_team.domain import Source


class SearchProvider(Protocol):
    async def search(self, query: str) -> list[Source]: ...
    async def close(self) -> None: ...


class WikipediaSearch:
    ENDPOINT = "https://ru.wikipedia.org/w/api.php"

    def __init__(self, *, transport=None):
        self.client = httpx.AsyncClient(
            timeout=3,
            follow_redirects=False,
            transport=transport,
            headers={"User-Agent": "CHGKTeam/0.1 (local educational demo; MediaWiki API)"},
        )

    async def search(self, query: str) -> list[Source]:
        async with asyncio.timeout(3):
            response = await self.client.get(
                self.ENDPOINT,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query[:200],
                    "srlimit": 3,
                    "format": "json",
                    "utf8": 1,
                },
            )
            response.raise_for_status()
            data = response.json()
        if "error" in data:
            raise ValueError("Source API error")
        sources = []
        for item in data.get("query", {}).get("search", [])[:3]:
            title = str(item["title"])[:200]
            snippet = html.unescape(re.sub(r"<[^>]*>", "", str(item.get("snippet", ""))))
            sources.append(
                Source(
                    id=f"wiki-{int(item['pageid'])}",
                    title=title,
                    url="https://ru.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe=""),
                    snippet=snippet[:600],
                )
            )
        return sources

    async def close(self) -> None:
        await self.client.aclose()


class DemoSearch:
    async def search(self, query: str) -> list[Source]:
        return [
            Source(
                id="demo-foucault",
                title="Маятник Фуко · учебный источник",
                url="https://ru.wikipedia.org/wiki/Маятник_Фуко",
                snippet="Учебный фрагмент, не результат сетевого запроса: маятник "
                "иллюстрирует вращение Земли. В demo интернет не используется.",
            )
        ]

    async def close(self) -> None:
        pass
