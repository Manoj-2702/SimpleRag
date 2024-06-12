import os
import sys
import gradio as gr
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceEndpoint
import torch
from langchain_community.vectorstores import FAISS


api_token=os.getenv("HF_TOKEN")


list_llm = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"]  
list_llm_simple = [os.path.basename(llm) for llm in list_llm]


def load_file(pdf_path):
    loaders=[PyPDFLoader(x) for x in pdf_path]
    pages=[]
    
    for loader in loaders:
        pages.extend(loader.load())
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=64
    )
    doc_splits=text_splitter.split_documents(pages)
    return doc_splits


def create_db(splits):
    embeddings=HuggingFaceEmbeddings()
    vectordb=FAISS.from_documents(splits,embeddings)
    return vectordb


def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    if llm_model=="meta-llama/Meta-Llama-3-8B-Instruct":
        llm=HuggingFaceEndpoint(
            repo_id=llm_model,
            huggingfacehub_api_token = api_token,
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_k=top_k,
        )
    else:
        llm=HuggingFaceEndpoint(
            huggingfacehub_api_token = api_token,
            repo_id=llm_model,
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_k=top_k,
        )
        
    memory=ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )