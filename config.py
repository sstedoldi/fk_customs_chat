import os

db_config = {
    "db_def": "postgres",
    "db_name": "vector_db", 
    # "host": "localhost", # depends on the docker network
    # "host": "127.0.0.1", # windows localhost
    "host": "pgvector_db", # docker container name
    "port": "5432",
    "user": "sst", # superuser
    "password": "password", # superuser password
}

vector_store_config = {
    "table_name": "normativa_adu",
    "embed_dim": 768  # jinaai/jina-embeddings-v2-base-es embedding dimension
}

embed_model_config = {
    "model_name": "jinaai/jina-embeddings-v2-base-es"
}