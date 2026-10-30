"""A small, inspectable TF-IDF retriever for the curated knowledge base."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class RetrievedDocument:
    title: str
    source_url: str
    content: str
    relevance_score: float


class LocalRetriever:
    """Retrieve curated Markdown documents without sending them externally."""

    def __init__(self, documents_dir: str | Path) -> None:
        documents_dir = Path(documents_dir)
        paths = sorted(documents_dir.glob("*.md"))
        if not paths:
            raise ValueError(f"No knowledge-base documents found in {documents_dir}")

        self.documents = [_read_document(path) for path in paths]
        self.vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1, 2))
        self.document_vectors = self.vectorizer.fit_transform(
            [document.content for document in self.documents]
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedDocument]:
        """Return relevant sources, or none when the local corpus has no match."""
        if not query.strip():
            return []
        query_vector = self.vectorizer.transform([query])
        scores = (self.document_vectors @ query_vector.T).toarray().ravel()
        ranked_indices = np.argsort(scores)[::-1]
        results: list[RetrievedDocument] = []
        for index in ranked_indices[:top_k]:
            if scores[index] <= 0:
                continue
            document = self.documents[index]
            results.append(
                RetrievedDocument(
                    title=document.title,
                    source_url=document.source_url,
                    content=document.content,
                    relevance_score=float(scores[index]),
                )
            )
        return results


@dataclass(frozen=True)
class _KnowledgeDocument:
    title: str
    source_url: str
    content: str


def _read_document(path: Path) -> _KnowledgeDocument:
    lines = path.read_text(encoding="utf-8").splitlines()
    title = next((line.removeprefix("# ") for line in lines if line.startswith("# ")), path.stem)
    source_line = next((line for line in lines if line.startswith("Source: ")), "Source: local")
    return _KnowledgeDocument(
        title=title,
        source_url=source_line.removeprefix("Source: "),
        content="\n".join(line for line in lines if not line.startswith("Source: ")),
    )
