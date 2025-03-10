'''
# DATABASE CONNECTION & VECTOR STORE
This module is designed to manage the creation and connection to PostgreSQL databases and vector stores.
It includes methods for creating a new database, connecting to an existing one, creating a new vector store,
and connecting to an existing vector store using PostgreSQL 16 and PGVectorStore.
'''

import psycopg2
from sqlalchemy import make_url
from llama_index.vector_stores.postgres import PGVectorStore


# Method to create a new database
def create_database(db_def, db_name, host, password, port, user, stay_conn = False, default = True):
    try:
        dbname = db_def if default else db_name
        conn = psycopg2.connect(
            dbname=dbname,
            host=host,
            password=password,
            port=port,
            user=user,
        )
        conn.autocommit = True
        with conn.cursor() as c:
            c.execute(f"DROP DATABASE IF EXISTS {dbname}")
            c.execute(f"CREATE DATABASE {dbname}")
        print(f"Database {dbname} created/reseted successfully.")
        if stay_conn:
            return conn
        conn.close()
    except psycopg2.Error as e:
        print(f"Error creating database {dbname}: {e}")


# Method to check if a database exists
def database_exists(db_def, db_name, host, password, port, user, default = True):
    try:
        dbname = db_def if default else db_name
        conn = psycopg2.connect(
            dbname=dbname,
            host=host,
            password=password,
            port=port,
            user=user,
        )
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (dbname,))
            exists = cursor.fetchone() is not None
        conn.close()
        if exists:
            print(f"Database {dbname} exists.")
        else:
            print(f"Database {dbname} does not exist.")
        return exists
    except psycopg2.Error as e:
        print(f"Error checking if database {dbname} exists: {e}")
        return False


# Method to connect to an existing database
def connect_to_database(db_def, db_name, host, password, port, user, default = True):
    try:
        dbname = db_def if default else db_name
        conn = psycopg2.connect(
            dbname=dbname,
            host=host,
            password=password,
            port=port,
            user=user,
        )
        print(f"Connected to the database {dbname}.")
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database {dbname}: {e}")
        return None


# Method to create a new vector store
def create_vector_store(db_config, table_name, embed_dim):
    try:
        # Datababe info
        vector_store = PGVectorStore.from_params(
            database=db_config['db_name'],
            host=db_config['host'],
            password=db_config['password'],
            port=db_config['port'],
            user=db_config['user'],
            table_name=table_name,
            embed_dim=embed_dim,  # jinaai/jina-embeddings-v2-base-es embedding dimension
        )
        print("Vector store created successfully.")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

# Method to check if a table exists
def table_exists(conn, table_name):
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """, (table_name,))
            return cursor.fetchone()[0]
    except psycopg2.Error as e:
        print(f"Error checking if table {table_name} exists: {e}")
        return False


# Method to connect to an existing vector store
def connect_to_vector_store(db_config, conn, table_name, embed_dim):
    """
    Connect to an existing vector store without creating a new one.
    """
    try:
        # Connect to the database
        if conn is None:
            print("Database connection failed. Unable to connect to vector store.")
            return None

        if table_exists(conn, table_name):
            # If the table exists, connect to the vector store
            vector_store = PGVectorStore.from_params(
                database=db_config['db_name'],
                host=db_config['host'],
                password=db_config['password'],
                port=db_config['port'],
                user=db_config['user'],
                table_name=table_name,
                embed_dim=embed_dim,
            )
            print(f"Connected to the existing vector store using table: {table_name}.")
            return vector_store
        else:
            print(f"Error: The table {table_name} does not exist. Please create the vector store first.")
            return None
    except Exception as e:
        print(f"Error connecting to the vector store: {e}")
        return None
