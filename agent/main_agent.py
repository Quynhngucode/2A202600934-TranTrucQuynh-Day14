"""A deterministic RAG-like agent used by the Day 14 benchmark.

This is intentionally offline and lightweight. It simulates the main RAG
surfaces needed by the evaluation factory: retrieval IDs, contexts, citations,
token usage, latency and versioned configs.
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+", re.UNICODE)


@dataclass(frozen=True)
class DocumentChunk:
    doc_id: str
    chunk_id: str
    title: str
    content: str
    category: str


KNOWLEDGE_BASE: list[DocumentChunk] = [
    DocumentChunk(
        "DOC_LAW_2021_OVERVIEW",
        "DOC_LAW_2021_OVERVIEW::chunk_0",
        "Luat Phong chong ma tuy 2021",
        "Luat Phong chong ma tuy 2021 quy dinh phong ngua, dau tranh, kiem soat hoat dong lien quan den ma tuy.",
        "legal",
    ),
    DocumentChunk(
        "DOC_LAW_2021_OVERVIEW",
        "DOC_LAW_2021_OVERVIEW::chunk_1",
        "Luat Phong chong ma tuy 2021",
        "Nguoi nghien ma tuy co the cai nghien tu nguyen tai gia dinh, cong dong, co so cai nghien hoac cai nghien bat buoc.",
        "legal",
    ),
    DocumentChunk(
        "DOC_LAW_2021_OVERVIEW",
        "DOC_LAW_2021_OVERVIEW::chunk_2",
        "Luat Phong chong ma tuy 2021",
        "Gia dinh, nha truong va co quan co trach nhiem phoi hop phong ngua te nan ma tuy.",
        "legal",
    ),
    DocumentChunk(
        "DOC_PENAL_249",
        "DOC_PENAL_249::chunk_0",
        "Bo luat Hinh su Dieu 249",
        "Dieu 249 Bo luat Hinh su quy dinh toi tang tru trai phep chat ma tuy.",
        "legal",
    ),
    DocumentChunk(
        "DOC_PENAL_249",
        "DOC_PENAL_249::chunk_1",
        "Bo luat Hinh su Dieu 249",
        "Hinh phat co the la phat tu va tang nang theo khoi luong, loai chat va tinh tiet pham toi.",
        "legal",
    ),
    DocumentChunk(
        "DOC_PENAL_249",
        "DOC_PENAL_249::chunk_2",
        "Bo luat Hinh su Dieu 249",
        "Nguoi pham toi voi khoi luong lon hoac tai pham nguy hiem co the bi ap dung khung hinh phat nang hon.",
        "legal",
    ),
    DocumentChunk(
        "DOC_DECREE_105",
        "DOC_DECREE_105::chunk_0",
        "Nghi dinh 105/2021/ND-CP",
        "Nghi dinh 105/2021/ND-CP huong dan thi hanh mot so dieu cua Luat Phong chong ma tuy.",
        "legal",
    ),
    DocumentChunk(
        "DOC_DECREE_105",
        "DOC_DECREE_105::chunk_1",
        "Nghi dinh 105/2021/ND-CP",
        "Van ban quy dinh lap ho so quan ly nguoi su dung trai phep chat ma tuy.",
        "legal",
    ),
    DocumentChunk(
        "DOC_DECREE_105",
        "DOC_DECREE_105::chunk_2",
        "Nghi dinh 105/2021/ND-CP",
        "Dia phuong co trach nhiem to chuc quan ly, ho tro cai nghien va tai hoa nhap cong dong.",
        "legal",
    ),
    DocumentChunk(
        "DOC_DECREE_57",
        "DOC_DECREE_57::chunk_0",
        "Danh muc chat ma tuy va tien chat",
        "Danh muc chat ma tuy va tien chat duoc cap nhat bang cac phu luc kem theo nghi dinh.",
        "legal",
    ),
    DocumentChunk(
        "DOC_DECREE_57",
        "DOC_DECREE_57::chunk_1",
        "Danh muc chat ma tuy va tien chat",
        "Cac chat trong danh muc bi kiem soat chat che trong san xuat, luu thong va su dung.",
        "legal",
    ),
    DocumentChunk(
        "DOC_DECREE_57",
        "DOC_DECREE_57::chunk_2",
        "Danh muc chat ma tuy va tien chat",
        "Mot so chat co gia tri y hoc van phai duoc cap phep va theo doi nghiem ngat.",
        "legal",
    ),
    DocumentChunk(
        "DOC_NEWS_ARTIST_01",
        "DOC_NEWS_ARTIST_01::chunk_0",
        "Tin tuc nghe si va trach nhiem truyen thong",
        "Bai bao ve nghe si lien quan den ma tuy can duoc doi chieu voi nguon chinh thong truoc khi ket luan.",
        "news",
    ),
    DocumentChunk(
        "DOC_NEWS_ARTIST_01",
        "DOC_NEWS_ARTIST_01::chunk_1",
        "Tin tuc nghe si va trach nhiem truyen thong",
        "Truyen thong nen tranh suy dien toi danh khi chua co thong tin tu co quan chuc nang.",
        "news",
    ),
    DocumentChunk(
        "DOC_NEWS_ARTIST_01",
        "DOC_NEWS_ARTIST_01::chunk_2",
        "Tin tuc nghe si va trach nhiem truyen thong",
        "Nguoi cua cong chung can can trong vi hanh vi ca nhan co tac dong den cong dong nguoi ham mo.",
        "news",
    ),
    DocumentChunk(
        "DOC_NEWS_ARTIST_02",
        "DOC_NEWS_ARTIST_02::chunk_0",
        "Tin tuc phong chong ma tuy trong gioi tre",
        "Nhieu chien dich truyen thong nhan manh viec nguoi tre can nhan dien rui ro tu chat gay nghien.",
        "news",
    ),
    DocumentChunk(
        "DOC_NEWS_ARTIST_02",
        "DOC_NEWS_ARTIST_02::chunk_1",
        "Tin tuc phong chong ma tuy trong gioi tre",
        "Noi dung giao duc phong chong ma tuy nen dung ngon ngu de hieu va dua ra kenh tro giup ro rang.",
        "news",
    ),
    DocumentChunk(
        "DOC_NEWS_ARTIST_02",
        "DOC_NEWS_ARTIST_02::chunk_2",
        "Tin tuc phong chong ma tuy trong gioi tre",
        "Cac vu viec cua nguoi noi tieng thuong duoc dung nhu bai hoc ve trach nhiem xa hoi.",
        "news",
    ),
]


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _idf(corpus: Iterable[DocumentChunk]) -> dict[str, float]:
    chunks = list(corpus)
    total = len(chunks)
    doc_freq: dict[str, int] = {}
    for chunk in chunks:
        for token in set(tokenize(chunk.content + " " + chunk.title)):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    return {token: math.log((total + 1) / (freq + 0.5)) + 1 for token, freq in doc_freq.items()}


IDF = _idf(KNOWLEDGE_BASE)


class MainAgent:
    def __init__(self, version: str = "v2_hybrid_rerank", top_k: int = 4):
        self.version = version
        self.top_k = top_k
        self.name = f"DrugLawRAG-{version}"

    def _score_chunk(self, question: str, chunk: DocumentChunk) -> float:
        q_tokens = tokenize(question)
        c_tokens = tokenize(chunk.title + " " + chunk.content)
        if not q_tokens:
            return 0.0
        q_set = set(q_tokens)
        c_set = set(c_tokens)
        lexical_overlap = len(q_set & c_set) / len(q_set)
        weighted_overlap = sum(IDF.get(token, 1.0) for token in q_set & c_set) / max(
            sum(IDF.get(token, 1.0) for token in q_set), 1.0
        )
        score = 0.55 * lexical_overlap + 0.45 * weighted_overlap

        if self.version == "v1_base":
            return lexical_overlap
        if "rerank" in self.version and any(token in chunk.title.lower() for token in q_set):
            score += 0.08
        if "hybrid" in self.version and chunk.category == "legal" and {"dieu", "luat", "nghi", "dinh"} & q_set:
            score += 0.05
        return score

    def retrieve(self, question: str) -> list[dict]:
        scored = [
            {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "content": chunk.content,
                "category": chunk.category,
                "score": self._score_chunk(question, chunk),
            }
            for chunk in KNOWLEDGE_BASE
        ]
        scored.sort(key=lambda item: item["score"], reverse=True)
        filtered = [item for item in scored if item["score"] > 0]
        return filtered[: self.top_k]

    def _should_abstain(self, question: str, retrieved: list[dict]) -> bool:
        lower = question.lower()
        forbidden = ["gia co phieu", "hom nay", "bo qua tat ca context"]
        if any(term in lower for term in forbidden):
            return True
        return not retrieved or retrieved[0]["score"] < 0.08

    async def query(self, question: str) -> Dict:
        await asyncio.sleep(0.02 if self.version != "v1_base" else 0.03)
        retrieved = self.retrieve(question)

        if self._should_abstain(question, retrieved):
            answer = "Toi khong the xac minh thong tin nay tu cac nguon hien co."
        else:
            top = retrieved[0]
            support = " ".join(item["content"] for item in retrieved[:2])
            answer = (
                f"Dua tren {top['title']}, cau tra loi la: {support} "
                f"[{top['doc_id']}]."
            )

        prompt_tokens = len(tokenize(question)) + sum(len(tokenize(item["content"])) for item in retrieved)
        completion_tokens = len(tokenize(answer))
        tokens_used = prompt_tokens + completion_tokens
        return {
            "answer": answer,
            "contexts": [item["content"] for item in retrieved],
            "retrieved_ids": [item["doc_id"] for item in retrieved],
            "retrieved_chunk_ids": [item["chunk_id"] for item in retrieved],
            "retrieval_scores": [item["score"] for item in retrieved],
            "metadata": {
                "agent_version": self.version,
                "model": "offline-extractive-rag",
                "tokens_used": tokens_used,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "estimated_cost_usd": round(tokens_used * 0.00000015, 8),
                "sources": [item["doc_id"] for item in retrieved],
            },
        }


if __name__ == "__main__":
    async def test() -> None:
        agent = MainAgent()
        print(await agent.query("Dieu 249 quy dinh gi ve toi tang tru trai phep chat ma tuy?"))

    asyncio.run(test())
