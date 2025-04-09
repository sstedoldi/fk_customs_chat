"""
# INDEXING PIPELINE
Defines an IndexingPipeline class that provides methods to read documents
from various sources (PDF files, webpages, and directories) and processes
them by splitting the text into chunks, embedding the chunks using a 
provided embedding model, and adding them to a vector store. This version
also logs the indexing events to a dedicated table in the vector database,
including metadata such as the indexing date, source, and document type.
"""

# from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PyMuPDFReader
from llama_index.readers.web import SimpleWebPageReader # to improve extracting more metadata
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

import config
import json
import os
from datetime import datetime
from psycopg2 import pool as pg_pool

class IndexingPipeline:
    def __init__(self, embed_model, vector_store, chunk_size=512, chunk_overlap_prop=10, db_connection_params=None):
        """
        :param embed_model: Embedding model to embed text.
        :param vector_store: Vector store to add nodes.
        :param chunk_size: Maximum chunk size for splitting texts.
        :param db_connection_params: Optional dict with DB connection parameters to log indexing events.
        """
        self._embed_model = embed_model
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap_prop = chunk_overlap_prop
        self._db_connection_params = db_connection_params

        # If DB connection parameters are provided, initialize a connection pool and set up logging table.
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
            return documents
        except Exception as e:
            print(f"Error reading directory {directory_path}: {e}")
            return []

    def document_processing(self, documents, extra_metadata=None):
        """
        Processes documents by splitting text into chunks, embedding them, and 
        logging the indexing events with proper metadata.
        
        :param documents: List of document objects.
        :param extra_metadata: Optional dictionary with additional metadata 
                               (e.g., {"source_path": "path/to/file", "source_type": "pdf"}).
        """
        try:
            # Text Parsing using SentenceSplitter
            text_parser = SentenceSplitter(chunk_size=self._chunk_size,
                                           chunk_overlap=self._chunk_size*self._chunk_overlap_prop//100)
            text_chunks = []
            doc_idxs = []

            for doc_idx, doc in enumerate(documents):
                cur_text_chunks = text_parser.split_text(doc.text)
                text_chunks.extend(cur_text_chunks)
                doc_idxs.extend([doc_idx] * len(cur_text_chunks))

            # Creating Nodes with Merged Metadata
            nodes = []
            for idx, text_chunk in enumerate(text_chunks):
                node = TextNode(text=text_chunk)
                src_doc = documents[doc_idxs[idx]]
                # Start with the source document metadata (if any)
                metadata = src_doc.metadata.copy() if src_doc.metadata else {}
                # Merge any extra metadata provided (e.g., source path or type)
                if extra_metadata is not None:
                        metadata.update(extra_metadata)
                # Add the indexing date if not already provided
                metadata.setdefault('indexed_date', datetime.now().isoformat())
                print(metadata)
                node.metadata = metadata
                nodes.append(node)

            # Embedding Text and Adding Nodes to the Vector Store
            for node in nodes:
                try:
                    node_embedding = self._embed_model.get_text_embedding(
                        node.get_content(metadata_mode="all")
                    )
                    node.embedding = node_embedding
                except Exception as e:
                    print(f"Error embedding text node {node}: {e}")

            self._vector_store.add(nodes)

            # Log the indexing event for the batch of documents
            self._log_indexing_event(documents, extra_metadata)

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
        Logs each document indexing event to the indexing_logs table.
        The log includes a document summary (if available), source info, document type, 
        and the timestamp.
        
        :param documents: List of document objects that were indexed.
        :param extra_metadata: Optional extra metadata that may include keys like 
                               'source_path' and 'source_type'.
        """
        if not self._connection_pool:
            return

        connection = None
        try:
            connection = self._connection_pool.getconn()
            cursor = connection.cursor()
            for doc in documents:
                # Retrieve source and document type from the document metadata or extra_metadata
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
                # Log the entire original metadata as JSON for additional context
                cursor.execute("""
                    INSERT INTO indexing_logs (doc_title, source_path, source_type, indexed_date, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                """, (doc_title, source_path, source_type, indexed_date, json.dumps(extra_metadata)))
            connection.commit()
            cursor.close()
        except Exception as e:
            print(f"Error logging indexing event: {e}")
        finally:
            if connection:
                self._connection_pool.putconn(connection)

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from uuid import uuid4

import string

def robust_tokenizer(text: str) -> list:
    """
    Tokenizes the text using nltk.word_tokenize and filters out punctuation.

    :param text: The input text to tokenize.
    :return: A list of tokenized and lowercased words.
    """
    # Use nltk to tokenize text
    tokens = word_tokenize(text)
    # Convert tokens to lowercase and remove punctuation tokens
    tokens = [token.lower() for token in tokens if token not in string.punctuation]

    return tokens

BM25_INDEX_FILE = config.bm25_index_file

def save_bm25_index(tokenized_docs, file_path=BM25_INDEX_FILE):
    """
    Save the list of tokenized documents (each a list of tokens) to a JSON file.
    """
    try:
        with open(file_path, "w") as f:
            json.dump(tokenized_docs, f)
        print("BM25 index saved successfully.")
    except Exception as e:
        print(f"Error saving BM25 index: {e}")

def load_bm25_index(file_path=BM25_INDEX_FILE):
    """
    Load tokenized documents from a JSON file.
    Returns an empty list if the file does not exist.
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                tokenized_docs = json.load(f)
            print("BM25 index loaded successfully.")
            return tokenized_docs
        except Exception as e:
            print(f"Error loading BM25 index: {e}")
            return []
    else:
        print("No persistent BM25 index found. Starting fresh.")
        return []

class HydridIndexingPipeline:
    def __init__(self, embed_model, vector_store, chunk_size=512, chunk_overlap_prop=10, db_connection_params=None):
        """
        :param embed_model: Embedding model to embed text.
        :param vector_store: Vector store to add nodes.
        :param chunk_size: Maximum chunk size for splitting texts.
        :param db_connection_params: Optional dict with DB connection parameters to log indexing events.
        """
        self._embed_model = embed_model
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap_prop = chunk_overlap_prop
        self._db_connection_params = db_connection_params

        self._bm25_documents = []
        self._bm25_tokenized_docs = load_bm25_index()
        self._bm25_model = BM25Okapi(self._bm25_tokenized_docs) if self._bm25_tokenized_docs else None

        # If DB connection parameters are provided, initialize a connection pool and set up logging table.
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

            # Text Parsing using SentenceSplitter
            text_parser = SentenceSplitter(chunk_size=self._chunk_size,
                                           chunk_overlap=self._chunk_size*self._chunk_overlap_prop//100)

            # Creating Nodes with Merged Metadata and BM25 tokenization
            nodes = []

            # Process each document to create nodes with proper IDs
            for doc in documents:
                chunks = text_parser.split_text(doc.text)
                for i, chunk in enumerate(chunks):
                    node = TextNode(text=chunk)
                    # Assign document and chunk IDs
                    node.document_id = doc.doc_id
                    node.chunk_id = f"{doc.doc_id}-{i+1}"
                    
                    # Merge source document metadata with any extra metadata
                    metadata = doc.metadata.copy() if doc.metadata else {}
                    if extra_metadata is not None:
                        metadata.update(extra_metadata)
                    metadata.setdefault('indexed_date', datetime.now().isoformat())
                    # Optionally embed the IDs in the metadata as well
                    metadata['document_id'] = node.document_id
                    metadata['chunk_id'] = node.chunk_id

                    node.metadata = metadata

                    # Process BM25 tokens and update BM25 indexes
                    tokens = robust_tokenizer(chunk)
                    node.bm25_tokens = tokens
                    self._bm25_documents.append(chunk)
                    self._bm25_tokenized_docs.append(tokens)

                    nodes.append(node)

            # Update BM25 model and persist tokenized documents
            self._bm25_model = BM25Okapi(self._bm25_tokenized_docs)
            save_bm25_index(self._bm25_tokenized_docs)

            # Embed text for each node and add embeddings
            for node in nodes:
                try:
                    node_embedding = self._embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
                    node.embedding = node_embedding
                except Exception as e:
                    print(f"Error embedding text node {node.chunk_id}: {e}")

            # Add all nodes (with chunk and document IDs) to the vector store
            self._vector_store.add(nodes)

            # Log one indexing event per original document using document IDs as primary keys
            self._log_indexing_event(documents, extra_metadata)

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
                    document_id TEXT PRIMARY KEY,
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
