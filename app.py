from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import os
from datetime import datetime
import logging
import json
from sentence_transformers import SentenceTransformer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import QueryBundle
# locals
from modules.vector_database import create_database, database_exists, \
                                    connect_to_database, create_vector_store, \
                                    table_exists, connect_to_vector_store
from modules.indexing_pipeline import IndexingPipeline
from modules.vectordb_retriever import VectorDBRetriever
from modules.llm_interaction import simple_chat_openai, simple_chat_aws

# config
from config import db_config, vector_store_config  # Import configuration from the config module
from config import embed_model_config, config_apis

app = Flask(__name__)
CORS(app)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for context
retriever = None
indexing_pipeline = None

def initialize_app(app):
    global retriever, indexing_pipeline

    with app.app_context():
        # Database info
        db_args = {
            'db_def': db_config['db_def'],
            'db_name': db_config['db_name'],
            'host': db_config['host'],
            'password': db_config['password'],
            'port': db_config['port'],
            'user': db_config['user']
        }
        # Connecting to the vector database
        conn = connect_to_database(**db_args) \
            if database_exists(**db_args) \
            else create_database(**db_args, stay_conn=True)
        # Vector store info
        vs_args = {
            'table_name': vector_store_config['table_name'],
            'embed_dim': vector_store_config['embed_dim']
        }
        # Creating or connecting to the vector store
        vector_store = connect_to_vector_store(db_args, conn, **vs_args) \
            if table_exists(conn, vs_args['table_name']) \
            else create_vector_store(db_args, **vs_args)
        # Embedding model
        model_name = embed_model_config['model_name']
        embed_model = HuggingFaceEmbedding(
            model_name=model_name, trust_remote_code=True
        )
        # Cleaning db_args
        db_args.pop('db_def')  # not needed for SimpleConnectionPool
        db_args.update({'dbname': db_args.pop('db_name')}) # to run SimpleConnectionPool
        # Indexing pipeline
        indexing_pipeline = IndexingPipeline(embed_model=embed_model, 
                                             vector_store=vector_store,
                                             chunk_size=300, 
                                             db_connection_params=db_args)
        # Retriever pipeline
        VectorDBRetriever.setup_logging(level=logging.INFO)
        retriever = VectorDBRetriever(
            vector_store=vector_store,
            embed_model=embed_model,
            query_mode="default",
            similarity_top_k=3,
            db_connection_params=db_args  # unpacked ** into the module
        )
        # LLM api config
        config_apis()
        logger.info("App initialization complete.")

@app.route('/')
def index():
    app_name = 'Adubot App Server'
    program_name = 'QA system using llamaindex'
    author_name = 'Argentina - Santiago Tedoldi'
    current_date = datetime.now().strftime("%Y-%m-%d")
    return render_template('index.html', app_name=app_name, program_name=program_name,
                           author_name=author_name, current_date=current_date)

@app.route('/sem_search', methods=['POST'])
def semantic_search():
    try:
        data = request.get_json()
        query = data.get('query')
        if not query:
            return jsonify({'error': 'Query parameter is required'}), 400
        
        query_bundle = QueryBundle(query_str=query)
        result = retriever._retrieve(query_bundle)
        response = [
            {'node': node_with_score.node, 'score': node_with_score.score}
            for node_with_score in result
        ]
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Error in /sem_search endpoint: {e}")
        return jsonify({'error': 'An error occurred during semantic search'}), 500

@app.route('/answer', methods=['POST'])
def answer():
    try:
        single_request = request.get_json(force=True)
        query = single_request.get('query')
        metadata = single_request.get('metadata', {})

        query_bundle = QueryBundle(query_str=query)
        result = retriever._retrieve(query_bundle)        
        
        response = simple_chat_openai(query=query, documents=result)

        # Log metadata (user_id, session_id, timestamp)
        logger.info(f"Query metadata: {json.dumps(metadata)}")

        return jsonify({'response': response, 'metadata': metadata})

    except Exception as e:
        logger.error(f"Error in /answer endpoint: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/index', methods=['POST'])
def index_documents():
    try:
        data = request.get_json()
        source_type = data.get('source_type')
        source_path = data.get('source_path')
        doc_title = data.get('doc_title')
        additional_info = data.get('additional_info')
        comments = data.get('comments')
        metadata = {"source_type" : source_type,
                    "source_path" : source_path,
                    "doc_title" : str(doc_title),
                    "additional_info" : str(additional_info),
                    "comments" : str(comments)}

        if not source_type or not source_path:
            return jsonify({'error': 'source_type and source_path are required'}), 400

        if source_type == 'pdf':
            documents = indexing_pipeline.pdf_reader(source_path)
        elif source_type == 'webpage':
            documents = indexing_pipeline.webpage_reader([source_path]) # [] requiered for the function
        elif source_type == 'directory':
            documents = indexing_pipeline.directory_reader(source_path)
        else:
            return jsonify({'error': 'Invalid source_type provided'}), 400

        indexing_pipeline.document_processing(documents, extra_metadata=metadata)
        return jsonify({'message': 'Indexing completed successfully'}), 200

    except Exception as e:
        logger.error(f"Error in /index endpoint: {e}")
        return jsonify({'error': 'An error occurred during indexing'}), 500

@app.route('/index_history', methods=['GET'])
def index_history():
    """
    Retrieve indexing history from the database.
    """
    connection = None
    try:
        connection = indexing_pipeline._connection_pool.getconn()
        cursor = connection.cursor()

        cursor.execute("""
            SELECT doc_title, source_path, source_type, indexed_date, metadata
            FROM indexing_logs
            ORDER BY indexed_date DESC
            LIMIT 50;
        """)

        history = []
        for row in cursor.fetchall():
            history.append({
                "doc_title": row[0],
                "source_path": row[1],
                "source_type": row[2],
                "indexed_date": row[3].isoformat(),
                "metadata": row[4]
            })

        cursor.close()
        return jsonify(history), 200

    except Exception as e:
        logger.error(f"Error retrieving indexing history: {e}")
        return jsonify({'error': 'An error occurred fetching index history'}), 500

    finally:
        if connection:
            indexing_pipeline._connection_pool.putconn(connection)

if __name__ == '__main__':
    initialize_app(app)
    app.run(port=8080, debug=True)
