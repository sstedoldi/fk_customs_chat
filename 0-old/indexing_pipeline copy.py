'''
# INDEXING PIPELINE
defines an IndexingPipeline class that provides methods to read documents
from various sources (PDF files, webpages, and directories) and processes
them by splitting the text into chunks, embedding the chunks using a 
provided embedding model, and adding them to a vector store. It streamlines 
the workflow of document indexing for downstream tasks like search or 
analysis.
'''

from pathlib import Path
from llama_index.readers.file import PyMuPDFReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core import SimpleDirectoryReader

class IndexingPipeline:
    def __init__(self, embed_model, vector_store, chunk_size=512):
        self.embed_model = embed_model
        self.vector_store = vector_store
        self.chunk_size = chunk_size

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

    def document_processing(self, documents):
        """
        Processes documents by splitting text into chunks and embedding them.
        """
        try:
            # Step 1: Text Parsing
            text_parser = SentenceSplitter(chunk_size=self.chunk_size)
            text_chunks = []
            doc_idxs = []

            for doc_idx, doc in enumerate(documents):
                cur_text_chunks = text_parser.split_text(doc.text)
                text_chunks.extend(cur_text_chunks)
                doc_idxs.extend([doc_idx] * len(cur_text_chunks))

            # Step 2: Creating Nodes with Metadata
            nodes = []
            for idx, text_chunk in enumerate(text_chunks):
                node = TextNode(text=text_chunk)
                src_doc = documents[doc_idxs[idx]]
                node.metadata = src_doc.metadata
                nodes.append(node)

            # Step 3: Embedding Text and Adding to Vector Store
            for node in nodes:
                try:
                    node_embedding = self.embed_model.get_text_embedding(
                        node.get_content(metadata_mode="all")
                    )
                    node.embedding = node_embedding
                except Exception as e:
                    print(f"Error embedding text node {node}: {e}")

            self.vector_store.add(nodes)
        except Exception as e:
            print(f"Error processing documents: {e}")