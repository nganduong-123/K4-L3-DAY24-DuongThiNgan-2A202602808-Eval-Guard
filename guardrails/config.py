"""Windows-friendly internal search provider for the NeMo dialogue index.

The lab uses input/output self-check rails, not semantic intent routing.  A tiny
lexical index is sufficient for resolving the configured refusal message and
avoids Annoy's native compiler requirement on Windows.
"""

from __future__ import annotations

import re

from nemoguardrails.embeddings.index import EmbeddingsIndex, IndexItem


class LexicalSearchIndex(EmbeddingsIndex):
    def __init__(self, **_kwargs):
        self._items: list[IndexItem] = []

    @property
    def embedding_size(self):
        return 0

    @property
    def cache_config(self):
        return None

    async def add_item(self, item: IndexItem):
        self._items.append(item)

    async def add_items(self, items: list[IndexItem]):
        self._items.extend(items)

    async def build(self):
        return None

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))

    async def search(self, text: str, max_results: int = 20, threshold=None):
        query = self._tokens(text)
        ranked = []
        for position, item in enumerate(self._items):
            candidate = self._tokens(item.text)
            union = query | candidate
            similarity = len(query & candidate) / len(union) if union else 1.0
            ranked.append((similarity, -position, item))
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        minimum = float(threshold) if threshold not in (None, float("inf")) else 0.0
        return [item for score, _, item in ranked if score >= minimum][:max_results]


def init(app):
    app.register_embedding_search_provider("local_lexical", LexicalSearchIndex)
