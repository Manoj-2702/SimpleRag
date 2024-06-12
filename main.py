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


list_llm = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"]  
list_llm_simple = [os.path.basename(llm) for llm in list_llm]


def load_file(pdf_path):
    loaders=[PyPDFLoader(x) for x in pdf_path]
    pages=[]
    
    for loader in loaders:
        pages.extend(loader.load())