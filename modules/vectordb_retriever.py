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

logger = logging.getLogger(__name__)

class VectorDBRetriever(BaseRetriever):
    """Retriever over a postgres vector store."""

    def __init__(self, **kwargs) -> None:
        """Init params."""
        self._vector_store = kwargs.get('vector_store')
        self._embed_model = kwargs.get('embed_model')
        self._query_mode = kwargs.get('query_mode', 'default')
        self._similarity_top_k = kwargs.get('similarity_top_k', 5)
        self._db_connection_params = kwargs.get('db_connection_params')
        self._connection_pool = pg_pool.SimpleConnectionPool(1, 10, **self._db_connection_params)
        self._setup_log_table()
        super().__init__()
    # logging setup done in the app.py
    @staticmethod
    def setup_logging(level=logging.ERROR) -> None:
        """Set up logging configuration."""
        logging.basicConfig(level=level)
    # log table iniciation/ connection
    def _setup_log_table(self) -> None:
        """Set up the log table if it doesn't exist."""
        connection = None
        try:
            connection = self._connection_pool.getconn()
            cursor = connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retriever_logs (
                    id SERIAL PRIMARY KEY,
                    query TEXT,
                    result_count INT,
                    timestamp TIMESTAMP
                )
            """)
            connection.commit()
            cursor.close()
        except Exception as e:
            logger.error(f"Error setting up log table: {e}")
        finally:
            if connection:
                self._connection_pool.putconn(connection)

    def _log_query(self, query: str, result_count: int) -> None:
        """Log the query and result count to the database."""
        connection = None
        try:
            connection = self._connection_pool.getconn()
            cursor = connection.cursor()
            cursor.execute("""
                INSERT INTO retriever_logs (query, result_count, timestamp)
                VALUES (%s, %s, %s)
            """, (query, result_count, datetime.now()))
            connection.commit()
            cursor.close()
        except Exception as e:
            logger.error(f"Error logging query: {e}")
        finally:
            if connection:
                self._connection_pool.putconn(connection)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        try:
            # Get the query embedding
            query_embedding = self._embed_model.get_query_embedding(
                query_bundle.query_str
            )
            # Create a vector store query
            vector_store_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=self._similarity_top_k,
                mode=self._query_mode,
            )
            # Query the vector store
            query_result = self._vector_store.query(vector_store_query)

            nodes_with_scores = []
            for index, node in enumerate(query_result.nodes):
                score: Optional[float] = None
                if query_result.similarities is not None:
                    score = query_result.similarities[index]
                nodes_with_scores.append(NodeWithScore(node=node, score=score))

            # Log the query and result count
            self._log_query(query_bundle.query_str, len(nodes_with_scores))

            return nodes_with_scores

        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
        
import numpy as np
import json
from indexing_pipeline import robust_tokenizer

def serialize_node(node_with_score):
    """
    Convert a NodeWithScore object to a dictionary representation.
    Modify this function if you want to log additional/specific attributes.
    """
    node = node_with_score.node
    return {
        "document_id": getattr(node, "document_id", None),
        "chunk_id": getattr(node, "chunk_id", None),
        "score": node_with_score.score
    }
class VectorDBHybridRetriever(BaseRetriever):
    def __init__(self, vector_store, bm25_model, bm25_tokenized_docs, embed_model, 
                 db_connection_params,
                 bm25_weight=0.5, vector_weight=0.5, similarity_top_k=5):
        """
        :param bm25_model: BM25 model built from document tokens.
        :param bm25_tokenized_docs: Original tokenized documents list.
        """
        self._vector_store = vector_store
        self._bm25_model = bm25_model
        self._bm25_tokenized_docs = bm25_tokenized_docs
        self._embed_model = embed_model
        self._bm25_weight = bm25_weight
        self._vector_weight = vector_weight
        self._similarity_top_k = similarity_top_k
        self._db_connection_params = db_connection_params
        
        # Set up a connection pool for logging.
        try:
            self._connection_pool = pg_pool.SimpleConnectionPool(1, 10, **self._db_connection_params)
        except Exception as e:
            logger.error(f"Error setting up DB connection pool: {e}")
            self._connection_pool = None

        self._setup_log_table()
        super().__init__()

    @staticmethod
    def setup_logging(level=logging.ERROR) -> None:
        """Set up logging configuration."""
        logging.basicConfig(level=level)

    def _setup_log_table(self) -> None:
        """
        Set up the log table if it doesn't exist.
        The table now includes a result_nodes column to store the serialized retrieved nodes.
        """
        if not self._connection_pool:
            return

        connection = None
        try:
            connection = self._connection_pool.getconn()
            cursor = connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retriever_logs (
                    id SERIAL PRIMARY KEY,
                    query TEXT,
                    result_nodes JSONB,
                    timestamp TIMESTAMP
                )
            """)
            connection.commit()
            cursor.close()
        except Exception as e:
            logger.error(f"Error setting up log table: {e}")
        finally:
            if connection:
                self._connection_pool.putconn(connection)

    def _log_query(self, query: str, retrieved_nodes: list) -> None:
        """
        Log the query and the serialized list of retrieved nodes to the database.
        
        :param query: The original query string.
        :param retrieved_nodes: List of NodeWithScore objects.
        """
        if not self._connection_pool:
            return

        # Serialize the nodes to a JSON string.
        serialized_nodes = json.dumps([serialize_node(node_with_score) for node_with_score in retrieved_nodes])
        connection = None
        try:
            connection = self._connection_pool.getconn()
            cursor = connection.cursor()
            cursor.execute("""
                INSERT INTO retriever_logs (query, result_nodes, timestamp)
                VALUES (%s, %s, %s)
            """, (query, serialized_nodes, datetime.now()))
            connection.commit()
            cursor.close()
        except Exception as e:
            logger.error(f"Error logging query: {e}")
        finally:
            if connection:
                self._connection_pool.putconn(connection)

    def _retrieve(self, query_bundle: QueryBundle) -> list:
        """
        Retrieves relevant nodes for the query by combining BM25 and vector search.
        Results are aggregated by document using the document_id attribute.
        """
        try:
            query_str = query_bundle.query_str

            # --- Vector Retrieval ---
            # Obtain the query embedding and query the vector store.
            query_embedding = self._embed_model.get_query_embedding(query_str)
            vector_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=self._similarity_top_k*2, # double the embedding retrive
                mode="default",
            )
            vector_result = self._vector_store.query(vector_query)
            vector_nodes = vector_result.nodes
            vector_scores = np.array(vector_result.similarities) if vector_result.similarities else np.zeros(len(vector_nodes))

            # --- BM25 Retrieval ---
            tokenized_query = robust_tokenizer(query_str)
            # Get BM25 scores for all indexed chunks.
            bm25_scores = (np.array(self._bm25_model.get_scores(tokenized_query)) 
                           if self._bm25_model is not None 
                           else np.zeros(len(self._bm25_tokenized_docs)))

            # --- Normalize Scores ---
            if np.max(vector_scores) > 0:
                vector_scores = vector_scores / np.max(vector_scores)
            if np.max(bm25_scores) > 0:
                bm25_scores = bm25_scores / np.max(bm25_scores)

            # --- Combine Scores ---
            # Here we assume that the order of BM25 scores corresponds to the order of nodes in the vector store.
            combined_scores = self._vector_weight * vector_scores + self._bm25_weight * bm25_scores

            # --- Grouping by Document ---
            # Use each node's document_id (set during indexing) to group scores.
            doc_score_map = {}
            doc_node_map = {}
            for i, node in enumerate(vector_nodes):
                doc_id = getattr(node, "document_id", None)
                if doc_id is None:
                    continue  # Skip nodes with missing document_id.
                score = combined_scores[i]
                if doc_id not in doc_score_map or score > doc_score_map[doc_id]:
                    doc_score_map[doc_id] = score
                    doc_node_map[doc_id] = node

            # --- Sorting and Selecting Top K Documents ---
            aggregated_results = [
                (doc_id, doc_node_map[doc_id], doc_score_map[doc_id])
                for doc_id in doc_score_map
            ]
            aggregated_results.sort(key=lambda x: x[2], reverse=True)
            top_results = aggregated_results[:self._similarity_top_k]

            # Create NodeWithScore objects for the results.
            nodes_with_scores = [
                NodeWithScore(node=node, score=score)
                for _, node, score in top_results
            ]

            # Log the query along with the serialized list of node details.
            self._log_query(query_str, nodes_with_scores)
            return nodes_with_scores

        except Exception as e:
            logger.error(f"Error during hybrid retrieval: {e}")
            return []