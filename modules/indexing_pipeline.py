"""
# INDEXING PIPELINE
Defines an IndexingPipeline class that provides methods to read documents
from various sources (PDF files, webpages, and directories) and processes
them by splitting the text into chunks, embedding the chunks using a 
provided embedding model, and adding them to a vector store. This version
also logs the indexing events to a dedicated table in the vector database,
including metadata such as the indexing date, source, and document type.
"""

import json
import os
from datetime import datetime
from psycopg2 import pool as pg_pool

from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PyMuPDFReader
from llama_index.readers.web import SimpleWebPageReader # to improve extracting more metadata
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
# from llama_index.retrievers.bm25 import BM25Retriever
# from modules.sentence import SentenceSplitter
from modules.bm25 import BM25Retriever
from uuid import uuid4
# from nltk.tokenize import word_tokenize
# import string

class HydridIndexingPipeline:
    def __init__(self, 
                 embed_model, 
                 vector_store,  
                 chunk_size=512, 
                 chunk_overlap=50,
                 db_connection_params=None,
                 bm25_model=None,
                 bm25_path="",
                 bm25_verbose=True,
                 language="english",
                 ):
        """
        :param embed_model: Embedding model to embed text.
        :param vector_store: Vector store to add nodes.
        :param chunk_size: Maximum chunk size for splitting texts.
        :param db_connection_params: Optional dict with DB connection parameters to log indexing events.
        """
        self._embed_model = embed_model
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._db_connection_params = db_connection_params
        self._bm25_path = bm25_path # this path must help to persist bm25 models
        self._bm25_verbose = bm25_verbose
        self._language = language # ["spanish", "es"] or ["english", "en", True] for bm25 stopwords

        self._text_parser = SentenceSplitter(chunk_size=self._chunk_size,
                                             chunk_overlap=self._chunk_overlap)

        if bm25_model is not None:
            self._bm25_retriever = bm25_model
        else:
            if os.path.exists(self._bm25_path):
                try:
                    self._bm25_retriever = BM25Retriever.from_persist_dir(self._bm25_path)
                    print("Loaded BM25 model from persistent storage.")
                except Exception as e:
                    print(f"Error loading BM25 model from storage: {e}")
                    self._bm25_retriever = None
            else:
                self._bm25_retriever = None  # It can be created once you have nodes to index

        if self._db_connection_params:
            try:
                self._connection_pool = pg_pool.SimpleConnectionPool(1, 10, **self._db_connection_params)
                self._setup_indexing_log_table()
            except Exception as e:
                print(f"Error setting up DB connection pool: {e}")
                self._connection_pool = None
        else:
            self._connection_pool = None
    
    def pdf_reader(self, file_path):
        """
        Reads a PDF file and returns a list of document objects.
        """
        try:
            loader = PyMuPDFReader()
            documents = loader.load(file_path=file_path)
            print(f"Indexing a PDF document with length {len(documents)}")
            return documents
        except Exception as e:
            print(f"Error reading PDF file {file_path}: {e}")
            return []

    def webpage_reader(self, urls, html_to_text=True):
        """
        Reads webpages and returns a list of document objects.
        """
        try:
            reader = SimpleWebPageReader(html_to_text=html_to_text)
            documents = reader.load_data(urls)
            print(f"Indexing a WEBPAGE document with length {len(documents)}")
            return documents
        except Exception as e:
            print(f"Error reading webpages from URLs {urls}: {e}")
            return []

    def directory_reader(self, directory_path):
        """
        Reads all documents from a directory and returns a list of document objects.
        """
        try:
            reader = SimpleDirectoryReader(directory_path)
            documents = reader.load_data()
            print(f"Indexing a DIRECTORY with {len(documents)} documents")
            return documents
        except Exception as e:
            print(f"Error reading directory {directory_path}: {e}")
            return []

    # def _robust_tokenizer(text: str) -> list:
    #     """
    #     Tokenizes the text using nltk.word_tokenize and filters out punctuation.

    #     :param text: The input text to tokenize.
    #     :return: A list of tokenized and lowercased words.
    #     """
    #     # Use nltk to tokenize text
    #     tokens = word_tokenize(text)
    #     # Convert tokens to lowercase and remove punctuation tokens
    #     tokens = [token.lower() for token in tokens if token not in string.punctuation]

    #     return tokens

    def document_processing(self, documents, extra_metadata=None):
        """
        Processes documents by splitting text into chunks, embedding them, and 
        logging the indexing events with proper metadata.
        
        :param documents: List of document objects.
        :param extra_metadata: Optional dictionary with additional metadata 
                               (e.g., {"source_path": "path/to/file", "source_type": "pdf"}).
        """
        try:
            # Ensure unique document_ids
            for doc in documents:
                doc.doc_id = str(uuid4())

            print(f"Document IDs to process: {str([doc.id_ for doc in documents])}")

            # Creating Nodes with Merged IDs & Metadata
            nodes = []

            # Process each document to create nodes with proper IDs
            for doc in documents:
                chunks = self._text_parser.split_text(doc.text)

                for i, chunk in enumerate(chunks):
                    node = TextNode(text=chunk)
                    print(f"Node {i}: {node} ")
                    
                    # Merge source document metadata with any extra metadata
                    metadata = doc.metadata.copy() if doc.metadata else {}
                    if extra_metadata is not None:
                        metadata.update(extra_metadata)
                    metadata.setdefault('indexed_date', datetime.now().isoformat())
                    # Optionally embed the IDs in the metadata as well
                    metadata['document_id'] = doc.id_
                    metadata['chunk_id'] = f"{doc.id_}-{i+1}"
                    print(f"metada = {metadata}")
                    
                    node.metadata = metadata

                    nodes.append(node)

            # Embed text for each node and add embeddings
            for j, node in enumerate(nodes):
                try:
                    node_embedding = self._embed_model.get_text_embedding(
                        node.get_content(metadata_mode="all")
                        )
                    node.embedding = node_embedding
                    print(f"Node {j} with embedding: {node} ")
                except Exception as e:
                    print(f"Error embedding text node {node.metadata.chunk_id}: {e}")

            # Add all nodes to the vector store
            self._vector_store.add(nodes)
            print("Vector store updated with new nodes")

            # Log indexing event for each document
            self._log_indexing_event(documents, extra_metadata)
            print("Indexing event logged")

            # BM25 update
            if self._bm25_retriever is not None:
                try:
                    # BM25 update if exists already
                    self._bm25_retriever.update_index(
                        nodes, 
                        language=self._language,
                        verbose=self._bm25_verbose
                        )
                    os.makedirs(self._bm25_path, exist_ok=True)
                    self._bm25_retriever.persist(self._bm25_path)
                    print(f"BM25 index updated and model persisted to {self._bm25_path}")
                except Exception as e:
                    print(f"Error updating BM25 index: {e}")
            else:
                # BM25 instantiate if not exists already
                try:
                    self._bm25_retriever = BM25Retriever(
                        nodes=nodes,
                        language=self._language,
                        similarity_top_k=10, # fixed for now
                        verbose=self._bm25_verbose
                    )
                    os.makedirs(self._bm25_path, exist_ok=True)
                    self._bm25_retriever.persist(self._bm25_path)
                    print(f"BM25 index created and model persisted to {self._bm25_path}")
                except Exception as e:
                    print(f"Error creating BM25 retriever: {e}")

        except Exception as e:
            print(f"Error processing documents: {e}")

    def _setup_indexing_log_table(self):
        """
        Sets up the log table in the vector database to record indexing events.
        """
        if not self._connection_pool:
            return

        connection = None
        try:
            connection = self._connection_pool.getconn()
            cursor = connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS indexing_logs (
                    id SERIAL PRIMARY KEY,
                    document_id TEXT UNIQUE,
                    doc_title TEXT,
                    source_path TEXT,
                    source_type TEXT,
                    indexed_date TIMESTAMP,
                    metadata JSONB
                )
            """)
            connection.commit()
            cursor.close()
        except Exception as e:
            print(f"Error setting up indexing log table: {e}")
        finally:
            if connection:
                self._connection_pool.putconn(connection)

    def _log_indexing_event(self, documents, extra_metadata=None):
        """
        Logs each document indexing event to the indexing_logs table using document_id as the primary key.
        This ensures that each original document is logged only once, rather than logging every chunk from the document.
        
        :param documents: List of document objects that were indexed.
        :param extra_metadata: Optional extra metadata that may include keys like 'source_path' and 'source_type'.
        """
        if not self._connection_pool:
            return

        connection = None
        try:
            connection = self._connection_pool.getconn()
            cursor = connection.cursor()
            for doc in documents:
                # Retrieve source and document type information
                source_path = (doc.metadata.get("source_path")
                               if doc.metadata and "source_path" in doc.metadata
                               else extra_metadata.get("source_path") if extra_metadata and "source_path" in extra_metadata
                               else "unknown")
                source_type = (extra_metadata.get("source_type")
                               if extra_metadata and "source_type" in extra_metadata
                               else "unknown")
                indexed_date = datetime.now()
                doc_title = (doc.metadata.get("doc_title")
                             if doc.metadata and "doc_title" in doc.metadata
                             else extra_metadata.get("doc_title") if extra_metadata and "doc_title" in extra_metadata
                             else "N/A")

                # Insert the document-level indexing event using document_id as primary key
                cursor.execute("""
                    INSERT INTO indexing_logs (document_id, doc_title, source_path, source_type, indexed_date, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_id) DO NOTHING
                """, (doc.doc_id, doc_title, source_path, source_type, indexed_date, json.dumps(extra_metadata)))
            connection.commit()
            cursor.close()
        except Exception as e:
            print(f"Error logging indexing event: {e}")
        finally:
            if connection:
                self._connection_pool.putconn(connection)
