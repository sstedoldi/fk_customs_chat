# TESTS FOR DATABASE CONNECTION & VECTOR STORE MODULE

import unittest
from unittest.mock import patch, Mock
import psycopg2
from modules.vector_database import (
    create_database,
    database_exists,
    connect_to_database,
    create_vector_store,
    connect_to_vector_store,
    table_exists
)

class TestDatabaseConnection(unittest.TestCase):
    
    @patch('psycopg2.connect')
    def test_create_database_success(self, mock_connect):
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        create_database('test_db', 'localhost', 'password', 5432, 'user')
        mock_conn.cursor().execute.assert_any_call("DROP DATABASE IF EXISTS test_db")
        mock_conn.cursor().execute.assert_any_call("CREATE DATABASE test_db")
    
    @patch('psycopg2.connect', side_effect=psycopg2.Error("Database Error"))
    def test_create_database_failure(self, mock_connect):
        with self.assertLogs(level='ERROR') as log:
            create_database('test_db', 'localhost', 'password', 5432, 'user')
            self.assertIn("Error creating database test_db", log.output[0])
    
    @patch('psycopg2.connect')
    def test_database_exists(self, mock_connect):
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (True,)
        mock_connect.return_value.cursor.return_value = mock_cursor
        exists = database_exists('test_db', 'localhost', 'password', 5432, 'user')
        self.assertTrue(exists)
    
    @patch('psycopg2.connect')
    def test_connect_to_database_success(self, mock_connect):
        conn = connect_to_database('test_db', 'localhost', 'password', 5432, 'user')
        self.assertIsNotNone(conn)
    
    @patch('psycopg2.connect', side_effect=psycopg2.Error("Connection Error"))
    def test_connect_to_database_failure(self, mock_connect):
        conn = connect_to_database('test_db', 'localhost', 'password', 5432, 'user')
        self.assertIsNone(conn)

if __name__ == '__main__':
    unittest.main()
