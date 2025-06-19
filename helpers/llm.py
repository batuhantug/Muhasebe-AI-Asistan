import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from neo4j import GraphDatabase

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
    
    

def get_company_info(driver, company_name):
    """
    Get company information from Neo4j database.
    """
    with driver.session() as session:
        result = session.run("""
        MATCH (c:Company)-[:IN_SECTOR]->(s:Sector)
        WHERE c.name = $company_name
        RETURN c.name AS name, c.yas AS yas, c.statu AS statu, s.name AS sektor
        """, company_name=company_name)
        return result.single()

def get_eligibility_criteria(driver):
    """
    Get eligibility criteria from Neo4j database.
    """
    with driver.session() as session:
        result = session.run("""
        MATCH (e:Eligibility)
        RETURN e.min_age AS min_age, e.max_age AS max_age,
               e.applicant_types AS applicant_types, e.industries AS industries
        """)
        return result.single()

def check_company_eligibility(driver, company_name, llm):
    """
    Check if a company is eligible for KOSGEB support.
    """
    # Get company information
    company = get_company_info(driver, company_name)
    if not company:
        return None, "Company not found in database"
    
    # Get eligibility criteria
    criteria = get_eligibility_criteria(driver)
    if not criteria:
        return None, "Eligibility criteria not found in database"
    
    # Create parser
    parser = StructuredOutputParser.from_response_schemas(ELIGIBILITY_SCHEMAS)
    
    # Create prompt template
    prompt_template = PromptTemplate(
        template="""
        You are an eligibility expert.

        Evaluate if the company is eligible according to the eligibility criteria.

        Company Information:
        - Name: {name}
        - Age: {yas}
        - Type: {statu}

        Eligibility Criteria:
        - Min Age: {min_age}
        - Max Age: {max_age}
        - Allowed Applicant Types: {applicant_types}
        - Allowed Industries: {industries}

        Company Sector: {sektor}

        {format_instructions}
        """,
        input_variables=["name", "yas", "statu", "sektor", "min_age", "max_age", "applicant_types", "industries"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Format prompt
    prompt = prompt_template.format(
        name=company["name"],
        yas=company["yas"],
        statu=company["statu"],
        sektor=company["sektor"],
        min_age=criteria["min_age"],
        max_age=criteria["max_age"],
        applicant_types=criteria["applicant_types"],
        industries=criteria["industries"]
    )
    
    # Get response
    response = llm.invoke(prompt)
    parsed = parser.parse(response.content)
    
    return parsed["eligible"].strip().upper() == "YES", parsed["reason"]

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

def get_llm_conditions(query=None, model_name="gpt-4o-mini", style="professional"):
    """
    Determine LLM conditions based on query, model, and style.
    """
    conditions = AVAILABLE_MODELS[model_name].copy()
    style_config = RESPONSE_STYLES[style]
    
    # Update conditions with style settings
    conditions.update({
        "temperature": style_config["temperature"],
        "system_prompt": style_config["system_prompt"]
    })
    
    # Add query-based adjustments here later
    if query:
        # Example: Adjust temperature based on query complexity
        if len(query.split()) > 20:
            conditions["temperature"] = min(conditions["temperature"] + 0.1, 1.0)
    
    return conditions

def get_llm(temperature=0.1, model_name="gpt-4o-mini", style="professional", query=None):
    """
    Initialize the ChatOpenAI model with custom conditions.
    
    Args:
        temperature (float): Model temperature
        model_name (str): Name of the model to use
        style (str): Response style to use
        query (str): Optional query to adjust conditions
    
    Returns:
        ChatOpenAI: Initialized LLM instance
    """
    openai_api_key = st.secrets["openai"]["api_key"]
    
    # Get conditions based on parameters
    conditions = get_llm_conditions(query, model_name, style)
    
    # Initialize LLM with conditions
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=conditions["temperature"],
        model_name=model_name,
        max_tokens=conditions["max_tokens"]
    )
    
    return llm, conditions["system_prompt"]

