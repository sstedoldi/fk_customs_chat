# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/retrievers/llama-index-retrievers-bm25/llama_index/retrievers/bm25/base.py
# adapted to simulate an update of the retriever
# https://chatgpt.com/share/67f6cd5f-5f78-8004-8e48-ad7bba1267c9

import json
import logging
import os

from typing import Any, Callable, Dict, List, Optional, cast

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import (
    BaseNode,
    IndexNode,
    NodeWithScore,
    QueryBundle,
    MetadataMode,
)
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.vector_stores.utils import (
    node_to_metadata_dict,
    metadata_dict_to_node,
)

import bm25s
import Stemmer

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_ARGS = {"similarity_top_k": "similarity_top_k", "_verbose": "verbose"}

DEFAULT_PERSIST_FILENAME = "retriever.json"

class BM25Retriever(BaseRetriever):
    r"""A BM25 retriever that uses the BM25 algorithm to retrieve nodes.

    Args:
        nodes (List[BaseNode], optional):
            The nodes to index. If not provided, an existing BM25 object must be passed.
        stemmer (Stemmer.Stemmer, optional):
            The stemmer to use. Defaults to an english stemmer.
        language (str, optional):
            The language to use for stopword removal. Defaults to "en".
        existing_bm25 (bm25s.BM25, optional):
            An existing BM25 object to use. If not provided, nodes must be passed.
        similarity_top_k (int, optional):
            The number of results to return. Defaults to DEFAULT_SIMILARITY_TOP_K.
        callback_manager (CallbackManager, optional):
            The callback manager to use. Defaults to None.
        objects (List[IndexNode], optional):
            The objects to retrieve. Defaults to None.
        object_map (dict, optional):
            A map of object IDs to nodes. Defaults to None.
        token_pattern (str, optional):
            The token pattern to use. Defaults to (?u)\\b\\w\\w+\\b.
        skip_stemming (bool, optional):
            Whether to skip stemming. Defaults to False.
        verbose (bool, optional):
            Whether to show progress. Defaults to False.
    """

    def __init__(
        self,
        nodes: Optional[List[BaseNode]] = None,
        stemmer: Optional[Stemmer.Stemmer] = None,
        language: str = "en",
        existing_bm25: Optional[bm25s.BM25] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,
        objects: Optional[List[IndexNode]] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
        skip_stemming: bool = False,
        token_pattern: str = r"(?u)\b\w\w+\b",
    ) -> None:
        self.stemmer = stemmer or Stemmer.Stemmer("english")
        self.similarity_top_k = similarity_top_k
        self.token_pattern = token_pattern
        self.skip_stemming = skip_stemming

        if existing_bm25 is not None:
            self.bm25 = existing_bm25
            self.corpus = existing_bm25.corpus
        else:
            if nodes is None:
                raise ValueError("Please pass nodes or an existing BM25 object.")

            self.corpus = [node_to_metadata_dict(node) for node in nodes]

            corpus_tokens = bm25s.tokenize(
                [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes],
                stopwords=language,
                stemmer=self.stemmer if not skip_stemming else None,
                token_pattern=self.token_pattern,
                show_progress=verbose,
            )
            self.bm25 = bm25s.BM25()
            self.bm25.index(corpus_tokens, show_progress=verbose)
        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )

    @classmethod
    def from_defaults(
        cls,
        index: Optional[VectorStoreIndex] = None,
        nodes: Optional[List[BaseNode]] = None,
        docstore: Optional[BaseDocumentStore] = None,
        stemmer: Optional[Stemmer.Stemmer] = None,
        language: str = "en",
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        verbose: bool = False,
        skip_stemming: bool = False,
        token_pattern: str = r"(?u)\b\w\w+\b",
        # deprecated
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> "BM25Retriever":
        if tokenizer is not None:
            logger.warning(
                "The tokenizer parameter is deprecated and will be removed in a future release. "
                "Use a stemmer from PyStemmer instead."
            )

        # ensure only one of index, nodes, or docstore is passed
        if sum(bool(val) for val in [index, nodes, docstore]) != 1:
            raise ValueError("Please pass exactly one of index, nodes, or docstore.")

        if index is not None:
            docstore = index.docstore

        if docstore is not None:
            nodes = cast(List[BaseNode], list(docstore.docs.values()))

        assert (
            nodes is not None
        ), "Please pass exactly one of index, nodes, or docstore."

        return cls(
            nodes=nodes,
            stemmer=stemmer,
            language=language,
            similarity_top_k=similarity_top_k,
            verbose=verbose,
            skip_stemming=skip_stemming,
            token_pattern=token_pattern,
        )

    def get_persist_args(self) -> Dict[str, Any]:
        """Get Persist Args Dict to Save."""
        return {
            DEFAULT_PERSIST_ARGS[key]: getattr(self, key)
            for key in DEFAULT_PERSIST_ARGS
            if hasattr(self, key)
        }

    def persist(self, path: str, encoding: str = "utf-8", **kwargs: Any) -> None:
        """Persist the retriever to a directory."""
        self.bm25.save(path, corpus=self.corpus, **kwargs)
        with open(
            os.path.join(path, DEFAULT_PERSIST_FILENAME), "w", encoding=encoding
        ) as f:
            json.dump(self.get_persist_args(), f, indent=2)

    @classmethod
    def from_persist_dir(
        cls, path: str, encoding: str = "utf-8", **kwargs: Any
    ) -> "BM25Retriever":
        """Load the retriever from a directory."""
        bm25 = bm25s.BM25.load(path, load_corpus=True, **kwargs)
        with open(os.path.join(path, DEFAULT_PERSIST_FILENAME), encoding=encoding) as f:
            retriever_data = json.load(f)
        return cls(existing_bm25=bm25, **retriever_data)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str
        tokenized_query = bm25s.tokenize(
            query,
            stemmer=self.stemmer if not self.skip_stemming else None,
            token_pattern=self.token_pattern,
            show_progress=self._verbose,
        )
        indexes, scores = self.bm25.retrieve(
            tokenized_query, k=self.similarity_top_k, show_progress=self._verbose
        )

        # batched, but only one query
        indexes = indexes[0]
        scores = scores[0]

        nodes: List[NodeWithScore] = []
        for idx, score in zip(indexes, scores):
            # idx can be an int or a dict of the node
            if isinstance(idx, dict):
                node = metadata_dict_to_node(idx)
            else:
                node_dict = self.corpus[int(idx)]
                node = metadata_dict_to_node(node_dict)
            nodes.append(NodeWithScore(node=node, score=float(score)))

        return nodes
    
    def update_index(self, new_nodes: List[BaseNode], language: str = "en", verbose: bool = False) -> None:
        """
        Simulate an incremental update of the BM25 index by:
        1. Appending new nodes to the existing corpus.
        2. Re-tokenizing the entire corpus (both existing and new entries).
        3. Re-building the BM25 index using the complete updated token set.
        
        Parameters:
            new_nodes (List[BaseNode]): List of new node objects to add to the index.
            language (str): The language parameter used for stopword removal. Defaults to "en".
            verbose (bool): Whether to show progress during tokenization and indexing. Defaults to False.
        """
        # Convert each new node to its metadata representation and extend the corpus.
        new_corpus_entries = [node_to_metadata_dict(node) for node in new_nodes]
        self.corpus.extend(new_corpus_entries)
        
        # Reconstruct full list of nodes from the updated corpus.
        # Here we use the helper to convert metadata dictionaries back into nodes.
        all_nodes = [metadata_dict_to_node(metadata) for metadata in self.corpus]
        
        # Retrieve the content from each node for tokenization.
        corpus_texts = [node.get_content(metadata_mode=MetadataMode.EMBED) for node in all_nodes]
        
        # Tokenize the full updated corpus.
        tokenized_corpus = bm25s.tokenize(
            corpus_texts,
            stopwords=language,
            stemmer=self.stemmer if not self.skip_stemming else None,
            token_pattern=self.token_pattern,
            show_progress=verbose,
        )
        
        # Re-build the BM25 index with the updated token list.
        # This fully re-indexes the corpus, simulating an incremental update.
        self.bm25.index(tokenized_corpus, show_progress=verbose)
