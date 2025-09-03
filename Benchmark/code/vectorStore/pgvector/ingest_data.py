
# Connect to DB
import os
import psycopg2
import psycopg2.pool

# Chunking 
# from langchain.text_splitter import LatexTextSplitter

# from bs4 import BeautifulSoup


# Embedding Model : Vision Language Model 
# import torch
# import clip
from PIL import Image


# Embeeding Model : Language
from typing import List, Optional



# Parsing
import json








class IngestToDB:

    # Database configuration
    DB_NAME = "RSystems_Embeddings_1_0"
    DB_USER = "postgres"
    DB_PASSWORD = "postgres"
    DB_HOST = "panini.cse.iitd.ac.in"
    DB_PORT = "5434"
    connection_pool = None


    def __init__(self, table_name=None):
        
        self.table_name = table_name

 

        
        
    @staticmethod
    def get_connection_string():    
        """
            Returns the connection string for the PostgreSQL database.
        """ 
        CONNECTION_STRING = f"postgresql+psycopg2://{IngestToDB.DB_USER}:{IngestToDB.DB_PASSWORD}@{IngestToDB.DB_HOST}:{IngestToDB.DB_PORT}/{IngestToDB.DB_NAME}"
        
        return CONNECTION_STRING
    
    @staticmethod
    def connect_to_db()-> psycopg2.pool.SimpleConnectionPool:
        """
            Connect to the PostgreSQL database and return a connection pool. 
        """
        
        # Create a connection pool with a minimum of 2 connections and
        #a maximum of 3 connections
        pool = psycopg2.pool.SimpleConnectionPool( 
            2, 3, user=IngestToDB.DB_USER, 
            password=IngestToDB.DB_PASSWORD,
            host=IngestToDB.DB_HOST, 
            port=IngestToDB.DB_PORT, 
            database=IngestToDB.DB_NAME)
        
        IngestToDB.pool = pool
        return pool
    
    @staticmethod
    def release_all_connections(connection):
        """
            Release the connection back to the pool.
        """
        if IngestToDB.connection_pool:
            IngestToDB.connection_pool.closeall()
    
    def run_query(self,query: str,params: Optional[dict] = None):
        """
            Run a query on the PostgreSQL database.
        """
        results = None
        try:
            # Get a connection from the pool
            connection = IngestToDB.connection_pool.getconn()
            cursor = connection.cursor()
            
            # Execute the query
            if params is None:
                cursor.execute(query)
            else:
                cursor.execute(query,params)
            # # Fetch the results
            # results = cursor.fetchall()
            
            # Commit the changes
            connection.commit()
            
            # Close the cursor
            cursor.close()
            
        except Exception as e:
            print(f"An error occurred: {e}")
        
        finally:
            # Release the connection back to the pool
            IngestToDB.connection_pool.putconn(connection)
        
        return results

    def save_embedding_to_db(self,data: dict)-> None:   
        """
            Save the image and corressponding text embedding to the PostgreSQL database.
        """
        table_name = data.get('table_name',None)
        if not table_name:
            table_name = self.table_name

        image_name = data.get('image_name','')
        image_path = data.get('image_path','')
        embedding = data.get('embedding', '')
        other_metadata = json.dumps(data.get('other_metadata'))

       
        query_str = f"""INSERT INTO {table_name} (image_name, image_path,embedding,other_metadata) 
        VALUES (%(image_name)s, %(image_path)s,%(embedding)s, %(other_metadata)s);"""

        db_dict = {
            "image_name": image_name,
            "image_path": image_path,
            "embedding": embedding,
            "other_metadata": other_metadata
        }

        

        _ = self.run_query(query_str,db_dict)


    
        
if __name__ == "__main__":
  pass