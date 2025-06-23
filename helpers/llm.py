import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from neo4j import GraphDatabase
from openai import OpenAI


def llm():
    """
    Initialize and return a default LLM instance.
    """
    openai_api_key = st.secrets["openai"]["api_key"]
    
    return OpenAI(api_key=openai_api_key)
    
def cypher_llm():
    """
    Initialize and return a default Cypher LLM instance.
    """
    openai_api_key = st.secrets["openai"]["api_key"]
    
    return ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        openai_api_key=openai_api_key
    )

def qa_llm():
    """
    Initialize and return a default QA LLM instance.
    """
    openai_api_key = st.secrets["openai"]["api_key"]
    
    return ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.3,         # small creativity for fluent answers
        openai_api_key=openai_api_key
    )


