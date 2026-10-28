"""Searching a task's files with LlamaIndex: by keyword, by meaning, or both."""

import shutil
import uuid
from functools import lru_cache
from pathlib import Path

from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.llms import MockLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

from autonoms.models import IndexInfo, SearchHit, SearchResult
from autonoms.workspace import Workspaces

MODES = ("keyword", "semantic", "hybrid")
SNIPPET_CHARS = 400


class NoIndex(Exception):
    pass


class MatchesOnly(BaseRetriever):
    """Wraps a retriever and drops results with a score of 0.

    BM25 returns *every* document when nothing matches, all with score 0 and in
    arbitrary order. Left in, those fake matches add noise to the rank fusion
    (a bug the hybrid test caught).
    """

    def __init__(self, inner: BaseRetriever):
        self._inner = inner
        super().__init__()

    def _retrieve(self, query_bundle):
        return [r for r in self._inner.retrieve(query_bundle) if (r.score or 0) > 0]


@lru_cache
def load_embedder(model_name: str) -> HuggingFaceEmbedding:
    """Load the embedding model once per process; loading takes seconds."""
    return HuggingFaceEmbedding(model_name=model_name, device="cpu")


class SearchIndexes:
    def __init__(self, root: Path, workspaces: Workspaces, embedding_model: str):
        self.root = Path(root).resolve()
        self._workspaces = workspaces
        self._model_name = embedding_model
        self._loaded: dict[str, VectorStoreIndex] = {}

    def _dir(self, task_id: str) -> Path:
        return self.root / str(uuid.UUID(task_id))  # raises on non-UUIDs

    def build(self, task_id: str) -> IndexInfo:
        files = self._workspaces.text_files(task_id)  # raises NoWorkspace
        documents = [Document(text=text, metadata={"path": path}) for path, text in files]
        # Chunks of about 256 tokens: big enough to hold an idea, small enough
        # that a hit points at the relevant part of a long file.
        nodes = SentenceSplitter(chunk_size=256, chunk_overlap=32).get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes, embed_model=load_embedder(self._model_name))
        target = self._dir(task_id)
        shutil.rmtree(target, ignore_errors=True)  # replace any older index
        index.storage_context.persist(persist_dir=str(target))
        self._loaded[task_id] = index
        return IndexInfo(files=len(files), chunks=len(nodes), embedding_model=self._model_name)

    def remove(self, task_id: str) -> bool:
        self._loaded.pop(task_id, None)
        target = self._dir(task_id)
        if not target.is_dir() or target.parent != self.root:
            return False
        shutil.rmtree(target)
        return True

    def _index(self, task_id: str) -> VectorStoreIndex:
        if task_id not in self._loaded:
            target = self._dir(task_id)
            if not target.is_dir():
                raise NoIndex(task_id)
            storage = StorageContext.from_defaults(persist_dir=str(target))
            self._loaded[task_id] = load_index_from_storage(
                storage, embed_model=load_embedder(self._model_name)
            )
        return self._loaded[task_id]

    def search(self, task_id: str, query: str, mode: str = "hybrid", top_k: int = 5) -> SearchResult:
        if mode not in MODES:
            raise ValueError(mode)
        index = self._index(task_id)
        nodes = list(index.docstore.docs.values())
        top_k = max(1, min(top_k, len(nodes)))  # BM25 fails if asked for more than exist
        semantic = index.as_retriever(similarity_top_k=top_k)
        keyword = MatchesOnly(BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k))
        if mode == "semantic":
            retriever = semantic
        elif mode == "keyword":
            retriever = keyword
        else:
            # Reciprocal rank fusion: combine the two rankings by position, not by
            # raw score, because cosine similarity and BM25 scores are not comparable.
            # num_queries=1 means "do not ask an LLM to rewrite the query".
            retriever = QueryFusionRetriever(
                [semantic, keyword],
                similarity_top_k=top_k,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False,
                llm=MockLLM(),
            )
        hits = []
        for result in retriever.retrieve(query):
            hits.append(
                SearchHit(
                    path=result.node.metadata["path"],
                    score=round(float(result.score or 0), 4),
                    text=result.node.get_content()[:SNIPPET_CHARS],
                )
            )
        return SearchResult(query=query, mode=mode, hits=hits)

