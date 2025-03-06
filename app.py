from flask import Flask, jsonify, request, render_template
from flask import session  # to generate secret info about the user
from flask_cors import CORS
import os
from datetime import datetime
import logging
from sentence_transformers import SentenceTransformer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import QueryBundle
# locals
from modules.vector_database import create_database, database_exists, \
                                    connect_to_database, create_vector_store, \
                                    table_exists, connect_to_vector_store
from modules.indexing_pipeline import IndexingPipeline
from modules.vectordb_retriever import VectorDBRetriever
from modules.llm_interaction import call_adubot_prueba1

# config
from config import db_config, vector_store_config  # Import configuration from the config module
from config import embed_model_config, config_apis

app = Flask(__name__)  # , static_folder='static')
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
            'db_name': db_config['db_name'],
            'host': db_config['host'],
            'password': db_config['password'],
            'port': db_config['port'],
            'user': db_config['user']
        }
        # Connecting to the database
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
        # Indexing pipeline
        indexing_pipeline = IndexingPipeline(embed_model, vector_store)
        # Retriever pipeline
        db_args.update({'dbname': db_args.pop('db_name')}) # to tun SimpleConnectionPool
        VectorDBRetriever.setup_logging(level=logging.INFO)
        retriever = VectorDBRetriever(
            vector_store=vector_store,
            embed_model=embed_model,
            query_mode="default",
            similarity_top_k=5,
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
        query = single_request['query']

        query_bundle = QueryBundle(query_str=query)
        result = retriever._retrieve(query_bundle)        
        
        response = call_adubot_prueba1(query=query, documents=result)

        return jsonify({'response': response})

    except Exception as e:
        logger.error(f"Error in /answer endpoint: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500


@app.route('/index', methods=['POST'])
def index_documents():
    try:
        data = request.get_json()
        source_type = data.get('source_type')
        source_path = data.get('source_path')

        if not source_type or not source_path:
            return jsonify({'error': 'source_type and source_path are required'}), 400

        if source_type == 'pdf':
            documents = indexing_pipeline.pdf_reader(source_path)
        elif source_type == 'webpage':
            documents = indexing_pipeline.webpage_reader([source_path])
        elif source_type == 'directory':
            documents = indexing_pipeline.directory_reader(source_path)
        else:
            return jsonify({'error': 'Invalid source_type provided'}), 400

        indexing_pipeline.document_processing(documents)
        return jsonify({'message': 'Indexing completed successfully'}), 200

    except Exception as e:
        logger.error(f"Error in /index endpoint: {e}")
        return jsonify({'error': 'An error occurred during indexing'}), 500


if __name__ == '__main__':
    initialize_app(app)

    # Run the app
    app.run(port=8080, debug=True)

