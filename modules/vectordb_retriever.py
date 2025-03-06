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
from llama_index.vector_stores.postgres import PGVectorStore
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