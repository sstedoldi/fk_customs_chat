import os

db_config = {
    "db_def": "postgres",
    "db_name": "vector_db", 
    # "host": "172.28.131.115",  # WSL IP address
    # "host": "localhost", # depends on the docker network
    "host": "127.0.0.1", # windows localhost
    "port": "5432",
    "user": "sst",
    "password": "password", # move to docker secrets
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