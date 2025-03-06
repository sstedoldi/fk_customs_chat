import os

def config_apis():
    os.environ['OPENAI_API_KEY'] = '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'
    
db_config = {
    "db_name": "vector_db",
    # "host": "172.28.131.115",  # WSL IP address
    "host": "localhost",
    "port": "5432",
    "user": "sst",
    "password": "password",
}

vector_store_config = {
    "table_name": "normativa_adu",
    "embed_dim": 768  # jinaai/jina-embeddings-v2-base-es embedding dimension
}

embed_model_config = {
    "model_name": "jinaai/jina-embeddings-v2-base-es"
}

# class DevelopmentConfig(Config):
#     DEBUG = True

# class ProductionConfig(Config):
#     DEBUG = False