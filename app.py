import json
import boto3
import os
import sys
import streamlit as st  

### TITAN Embedding model
from langchain.llms.bedrock import Bedrock
from langchain_community.embeddings import BedrockEmbeddings

## DataIngestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

## VectorEmbeddings and  vectorStore
from langchain_community.vectorstores import FAISS


## LLM models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock clients
bedrock = boto3.client(service_name = "bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id = "amazon.titan-embed-text-v1", client = bedrock)


## data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("pdfs")
    documents = loader.load()
    
    text_splitter =  RecursiveCharacterTextSplitter(chunk_size = 10000,
                                                    chunk_overlap = 1000)   
    
    docs = text_splitter.split_documents(documents)
    return docs

### Vector store

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    ##create the Anthropic Model
    llm=Bedrock(model_id="ai21.j2-mid-v1",client=bedrock,
                model_kwargs={'maxTokens':512})
    
    return llm

def get_llama2_llm():
    ##create the Anthropic Model
    llm=Bedrock(model_id="meta.llama2-70b-chat-v1",client=bedrock,
                model_kwargs={'max_gen_len':512})
    
    return llm


prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = vectorstore_faiss.as_retriever(
            search_type = "similarity", search_kwarge = {"k":3}
            ),
    return_source_documents = True,
    chain_type_kwargs = {"prompt": PROMPT}
    )

    answer = qa({"query": query})
    return answer['result']


def save_uploaded_file(uploaded_file, save_path):
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False

def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("File Upload:")
        # Move the file uploader to the sidebar
        uploaded_file = st.sidebar.file_uploader("Choose a file", type=['pdf', 'txt', 'docx'])
        if uploaded_file is not None:
            # Save the uploaded file
            if save_uploaded_file(uploaded_file, 'pdfs'):
                st.success(f"File {uploaded_file.name} uploaded successfully.")
            else:
                st.error("Failed to save the file.")

        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()  # Ensure this processes files from the 'pdfs' folder
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm=get_claude_llm()
            
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm=get_llama2_llm()
            
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")  

if __name__ == "__main__":
    main()
