import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from neo4j import GraphDatabase
from openai import OpenAI




# Define available models and their configurations
AVAILABLE_MODELS = {
    "gpt-4o-mini": {
        "temperature": 0,
        "max_tokens": 2000,
        "response_style": "professional",
        "language": "turkish",
        "model_kwargs": {
            "top_p": 0.95,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": None
        }
    }
}

# Define response styles and their configurations
RESPONSE_STYLES = {
    "professional": {
        "temperature": 0.1,
        "system_prompt": "Sen muhasebe ve vergi konularında uzman bir asistansın. Profesyonel ve resmi bir dil kullan."
    },


}

# Define keywords and their corresponding vector indexes
VECTOR_INDEX_KEYWORDS = {
    "kosgeb": {
        "index_name": "kosgeb_vector_index",
        "keywords": ["kosgeb", "küçük işletme", "girişimci", "destek", "hibe", "teşvik"],
        "label": "Chunk"
    },
    "tax": {
        "index_name": "tax_law_vector_index",
        "keywords": ["vergi", "kdv", "gelir vergisi", "kurumlar vergisi", "damga vergisi", "vergi kanunu"],
        "label": "Article"
    }
}

# Define eligibility response schemas
ELIGIBILITY_SCHEMAS = [
    ResponseSchema(name="eligible", description="YES if eligible, NO otherwise."),
    ResponseSchema(name="reason", description="Reason for eligibility decision.")
]

def llm():
    """
    Initialize and return a default LLM instance.
    """
    openai_api_key = st.secrets["openai"]["api_key"]
    
    return OpenAI(api_key=openai_api_key)
    





def determine_vector_index(query):
    """
    Determine which vector index to use based on the query content.
    
    Args:
        query (str): The user's query
        
    Returns:
        dict: Dictionary containing index_name and label
    """
    query = query.lower()
    
    # Check each index's keywords
    for index_type, config in VECTOR_INDEX_KEYWORDS.items():
        for keyword in config["keywords"]:
            if keyword in query:
                return {
                    "index_name": config["index_name"],
                    "label": config["label"]
                }
    
    # Default to kosgeb index if no specific keywords found
    return {
        "index_name": "kosgeb_vector_index",
        "label": "Chunk"
    }



