import os
from chromadb.config import Settings

""" chromadb is an opensource database for storing the embeddings (Similar to pinecone"""

#Define the chrom settings.
CHROMA_SETTINGS = Settings(
    chroma_db_impl = 'duckdb+parquet',  #Define the format to store the embeddings in chromdb using duckdb
    persist_directory = "db",           #The directory to store the embeddings
    anonymized_telemetry = False
)