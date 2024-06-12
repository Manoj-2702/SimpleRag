import os, tempfile, qdrant_client
import streamlit as st
from llama_index.llms import OpenAI, Gemini, Cohere
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex, StorageContext
from llama_index.node_parser import SentenceSplitter, CodeSplitter, SemanticSplitterNodeParser, TokenTextSplitter
from llama_index.node_parser.file import HTMLNodeParser, JSONNodeParser, MarkdownNodeParser
from llama_index.vector_stores import QdrantVectorStore, PineconeVectorStore
from pinecone import Pinecone

def reset_pipeline_generated():
    if 'pipeline_generated' in st.session_state:
        st.session_state['pipeline_generated'] = False
        
def upload_file():
    file = st.file_uploader("Upload a file", on_change=reset_pipeline_generated)
    if file is not None:
        file_path = save_uploaded_file(file)
        
        if file_path:
            loaded_file = SimpleDirectoryReader(input_files=[file_path]).load_data()
            print(f"Total documents: {len(loaded_file)}")

            st.success(f"File uploaded successfully. Total documents loaded: {len(loaded_file)}")
            #print(loaded_file)
        return loaded_file
    return None