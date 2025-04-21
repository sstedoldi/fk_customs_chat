'''
# VECTOR DC RETRIEVER
Implements a retriever for a PostgreSQL vector store. It uses the 
VectorDBRetriever class, which can retrieve nodes from a vector 
database based on embeddings. The retriever uses a connection pool 
for efficient database access and logs queries to a dedicated table. 
It is designed for integration with custom embedding models and can 
be used in applications by importing and instantiating the class. 
The module also supports logging configuration to provide detailed 
error and usage tracking.
'''

from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.schema import NodeWithScore
from typing import Any, List, Optional
import logging
from psycopg2 import pool as pg_pool
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class HybridRetriever(BaseRetriever):
    """Hybrid Retriever combining dense vector store retrieval and BM25 retrieval."""

    def __init__(
        self,
        vector_store,
        embed_model,
        bm25_retriever=None,
        query_mode: str = 'default',
        similarity_top_k: int = 5,
        dense_top_k: int = 10,
        bm25_top_k: int = 10,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        db_connection_params: Optional[dict] = None,
    ) -> None:
        """
        :param vector_store: Dense vector store instance.
        :param embed_model: Embedding model used to produce query embeddings.
        :param bm25_retriever: Optional BM25 retrieval model instance. Assumed to provide a .retrieve(query, top_k) method.
        :param query_mode: Query mode for vector retrieval.
        :param similarity_top_k: Number of dense results to return.
        :param bm25_top_k: Number of BM25 results to return.
        :param dense_weight: Weight for the dense retrieval component.
        :param bm25_weight: Weight for the BM25 retrieval component.
        :param db_connection_params: Optional dictionary of DB connection params used for logging.
        """
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._bm25_retriever = bm25_retriever
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        self._dense_top_k = dense_top_k
        self._bm25_top_k = bm25_top_k
        self._dense_weight = dense_weight
        self._bm25_weight = bm25_weight
        self._db_connection_params = db_connection_params

        if self._db_connection_params:
            self._connection_pool = pg_pool.SimpleConnectionPool(1, 10, **self._db_connection_params)
            self._setup_log_table()
        else:
            self._connection_pool = None

        super().__init__()

    @staticmethod
    def _setup_logging(level=logging.ERROR) -> None:
        """Set up logging configuration."""
        logging.basicConfig(level=level)

    def _setup_log_table(self) -> None:
        """Set up the logging table if it doesn’t already exist."""
        connection = None
        try:
            connection = self._connection_pool.getconn()
            cursor = connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retriever_logs (
                    id SERIAL PRIMARY KEY,
                    query TEXT,
                    result_count INT,
                    timestamp TIMESTAMP,
                    results TEXT
                )
            """)
            connection.commit()
            cursor.close()
        except Exception as e:
            logger.error(f"Error setting up log table: {e}")
        finally:
            if connection:
                self._connection_pool.putconn(connection)

    def _log_query(self, query: str, result_count: int, results: str) -> None:
        """Log the query, number of hits, and full JSON results to the database."""
        connection = None
        try:
            connection = self._connection_pool.getconn()
            cursor = connection.cursor()
            cursor.execute("""
                INSERT INTO retriever_logs (query, result_count, timestamp, results)
                VALUES (%s, %s, %s, %s)
            """, (
                query,
                result_count,
                datetime.now(),
                results
            ))
            connection.commit()
            cursor.close()
        except Exception as e:
            logger.error(f"Error logging query: {e}")
        finally:
            if connection:
                self._connection_pool.putconn(connection)

    @staticmethod
    def _normalize_scores(nodes_with_scores: List[NodeWithScore]) -> List[float]:
        """
        Normalize scores from a list of NodeWithScore to the [0, 1] range.
        If scores are not available, returns a list of zeros.
        """
        scores = [nws.score for nws in nodes_with_scores if nws.score is not None]
        if not scores:
            return [0 for _ in nodes_with_scores]
        min_score = min(scores)
        max_score = max(scores)
        normalized = []
        for nws in nodes_with_scores:
            if nws.score is None:
                normalized.append(0)
            elif max_score > min_score:
                norm_score = (nws.score - min_score) / (max_score - min_score)
                normalized.append(norm_score)
            else:
                normalized.append(1.0)
        return normalized

    def _get_node_key(self, node: Any) -> str:
        """
        Generate a unique key for a node. This function attempts to use node-specific
        attributes (such as 'chunk_id' or 'document_id') to deduplicate results.
        """
        if hasattr(node, 'chunk_id'):
            return getattr(node, 'chunk_id')
        elif hasattr(node, 'document_id'):
            return getattr(node, 'document_id')
        else:
            return str(id(node))

    def _serialize_node(self, node_with_score: NodeWithScore) -> dict:
        """
        Convert a NodeWithScore object to a dictionary representation.
        Modify this function if you want to log additional/specific attributes.
        """
        node = node_with_score.node
        return {
            "document_id": getattr(node, "document_id", None),
            "chunk_id": getattr(node, "chunk_id", None),
            "score": node_with_score.score,
            "text": getattr(node, "text", None),
        }

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve results by performing dense retrieval and BM25 search.
        Combines and re-ranks the results based on a weighted hybrid score.
        """
        try:
            # Dense retrieval
            query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
            vector_store_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=self._dense_top_k,
                mode=self._query_mode,
            )
            dense_query_result = self._vector_store.query(vector_store_query)
            dense_nodes = []
            for index, node in enumerate(dense_query_result.nodes):
                score = None
                if dense_query_result.similarities is not None:
                    score = dense_query_result.similarities[index]
                dense_nodes.append(NodeWithScore(node=node, score=score))
            # print(f"dense nodes: {dense_nodes}")
            
            # BM25 retrieval
            bm25_nodes = []
            if self._bm25_retriever:
                # Assuming the BM25 retriever implements a method 'retrieve'
                bm25_nodes = self._bm25_retriever.retrieve(query_bundle.query_str)
            # print(f"bm25 nodes: {bm25_nodes}")

            # Normalize scores from each retrieval method separately.
            dense_norm_scores = self._normalize_scores(dense_nodes) if dense_nodes else []
            bm25_norm_scores = self._normalize_scores(bm25_nodes) if bm25_nodes else []
            # print(f"dense nodes norm: {dense_norm_scores}")
            # print(f"bm25 nodes norm: {bm25_norm_scores}")

            # Calcula baseline values
            dense_baseline = min(dense_norm_scores) if dense_norm_scores else 0.0
            bm25_baseline = min(bm25_norm_scores) if bm25_norm_scores else 0.0

            # Merge results and deduplicate nodes based on a unique key
            combined_results = {}
            # Process dense results
            for i, nws in enumerate(dense_nodes):
                key = self._get_node_key(nws.node)
                combined_results[key] = {
                    'node': nws.node,
                    'dense_score': dense_norm_scores[i] if i < len(dense_norm_scores) else dense_baseline,
                    'bm25_score': bm25_baseline
                }
            # Process BM25 results
            for i, nws in enumerate(bm25_nodes):
                key = self._get_node_key(nws.node)
                if key in combined_results:
                    combined_results[key]['bm25_score'] = bm25_norm_scores[i] if i < len(bm25_norm_scores) else 0
                else:
                    combined_results[key] = {
                        'node': nws.node,
                        'dense_score': dense_baseline,
                        'bm25_score': bm25_norm_scores[i] if i < len(bm25_norm_scores) else bm25_baseline
                    }
            # print(f"combined nodes: {combined_results}")

            # Compute a weighted hybrid score for each node
            hybrid_results = []
            for entry in combined_results.values():
                hybrid_score = (self._dense_weight * entry['dense_score'] +
                                self._bm25_weight * entry['bm25_score'])
                hybrid_results.append(NodeWithScore(node=entry['node'], score=hybrid_score))
            # print(f"hydrid nodes: {hybrid_results}")

            # Sort the combined results in descending order by hybrid score
            hybrid_results.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)

            # Getting top hydrid result
            hybrid_results = hybrid_results[:self._similarity_top_k]

            # Log the query and results
            results_str = json.dumps([self._serialize_node(node) for node in hybrid_results])
            self._log_query(query_bundle.query_str, len(hybrid_results), results_str)

            return hybrid_results

        except Exception as e:
            logger.error(f"Error during hybrid retrieval: {e}")
            return []