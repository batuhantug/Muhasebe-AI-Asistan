import streamlit as st
import os
from ingestion.pdf_loader import load_data
from ingestion.web_scraper import load_gib_data
from helpers.neo4j_helper import load_company_data
from helpers.llm import llm, cypher_llm, qa_llm


from openai import OpenAI
from neo4j import GraphDatabase
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

import json
from datetime import datetime
from langchain.prompts import PromptTemplate
from helpers.llm import llm



from langchain_neo4j import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain.chat_models import ChatOpenAI


neo4j_uri = st.secrets["neo4j"]["uri"]
neo4j_user = st.secrets["neo4j"]["user"]
neo4j_password = st.secrets["neo4j"]["password"]

openai_api_key = st.secrets["openai"]["api_key"]


graph = Neo4jGraph(url=neo4j_uri, username=neo4j_user, password=neo4j_password)


embedding_model = OpenAIEmbeddings (model="text-embedding-3-small", openai_api_key=openai_api_key)
# Initialize Neo4j driver (IMPORTANT: you forgot this part)
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

llm = llm()
cypher_llm = cypher_llm()
qa_llm = qa_llm()

# Initialize memory
def initialize_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )






def is_database_empty(driver):
    with driver.session() as session:
        result = session.run("MATCH (d:Document) RETURN count(d) as count")
        count = result.single()["count"]
        return count == 0

def search_vector_db(driver, query, embedding_model, limit=5):
    # Check if database is empty
    if is_database_empty(driver):
        st.sidebar.warning("Database is empty. Please upload some documents first.")
        return []
        
    # Get query embedding
    query_embedding = embedding_model.embed_query(query)
    
    # Determine which vector index to use
    index_info = determine_vector_index(query)
    
    # Search in Neo4j
    with driver.session() as session:
        try:
            # First try vector search
            result = session.run(f"""
            CALL db.index.vector.queryNodes('{index_info["index_name"]}', $k, $embedding)
            YIELD node, score
            RETURN node.content as text, score, node.category as category, 
                   CASE WHEN node.title IS NOT NULL THEN node.title ELSE '' END as title
            ORDER BY score DESC
            LIMIT $limit
            """, 
            k=5,  # number of nearest neighbors
            embedding=query_embedding,
            limit=limit
            )
            
            # Collect results
            documents = []
            for record in result:
                if record["text"]:  # Only add if text is not None
                    documents.append({
                        "text": record["text"],
                        "score": record["score"],
                        "category": record["category"],
                        "title": record["title"]
                    })
            
            # If no results from vector search, try direct similarity search
            if not documents:
                st.sidebar.info(f"No results from vector search in {index_info['index_name']}, trying direct similarity search...")
                result = session.run(f"""
                MATCH (n:{index_info["label"]})
                WHERE n.content IS NOT NULL
                RETURN n.content as text, 1.0 as score, 
                       n.category as category,
                       CASE WHEN n.title IS NOT NULL THEN n.title ELSE '' END as title
                LIMIT $limit
                """,
                limit=limit
                )
                
                for record in result:
                    if record["text"]:  # Only add if text is not None
                        documents.append({
                            "text": record["text"],
                            "score": record["score"],
                            "category": record["category"],
                            "title": record["title"]
                        })
            
            if not documents:
                st.sidebar.warning(f"No relevant documents found in {index_info['index_name']}.")
            else:
                st.sidebar.success(f"Found {len(documents)} relevant documents in {index_info['index_name']}.")
            
            return documents
            
        except Exception as e:
            st.sidebar.error(f"Error during search: {str(e)}")
            return []

# Streamlit UI
st.title("MuhasebAI")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = initialize_memory()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Bana kosgeb destekleri ve vergiler ile ilgili istediğini sorabilirsin..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            try:
                
                chain = GraphCypherQAChain.from_llm(
                    graph=graph,
                    cypher_llm=cypher_llm,
                    qa_llm=cypher_llm,
                    verbose=True,
                    allow_dangerous_requests=True,
                )
                
                response = chain.invoke({"query": prompt})

                response = response.get('result')

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Sidebar selector
sidebar_option = st.sidebar.selectbox(
    "Select Section",
    ["Load PDF", "LoadTax Laws", "Add Company"]
)

if sidebar_option == "Load PDF":
    st.sidebar.header("Load PDF")
    
    # Add button to process existing PDFs
    if os.path.exists("data") and len(os.listdir("data")) > 0:
        if st.sidebar.button("Process Existing PDFs"):
            for file in os.listdir("data"):
                if file.endswith(".pdf"):
                    save_path = os.path.join("data", file)
                    with st.spinner(f'Processing {file}...'):
                        try:
                            load_data(save_path, client, embedding_model, driver)
                            st.sidebar.success(f"Ingested {file} into database")
                        except Exception as e:
                            st.sidebar.error(f"Error processing {file}: {str(e)}")

    uploaded_pdfs = st.sidebar.file_uploader(
        "Upload PDFs for Ingestion",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_pdfs:
        os.makedirs("data", exist_ok=True)
        for pdf in uploaded_pdfs:
            save_path = os.path.join("data", pdf.name)
            with open(save_path, "wb") as f:
                f.write(pdf.getbuffer())

            st.sidebar.write(f"Saved {pdf.name}")

            with st.spinner(f'Processing {pdf.name}...'):
                try:
                    load_data(save_path, client, embedding_model, driver)
                    st.sidebar.success(f"Ingested {pdf.name} into database")
                except Exception as e:
                    st.sidebar.error(f"Error processing {pdf.name}: {str(e)}")

elif sidebar_option == "LoadTax Laws":
    st.sidebar.header("Tax Laws Management")
    
    if st.sidebar.button("Update Tax Laws"):
        with st.spinner("Updating tax laws..."):
            
            load_gib_data(driver, embedding_model)

            st.sidebar.info("Tax laws updated Successfully")

elif sidebar_option == "Add Company":
    st.sidebar.header("Add New Company")
    
    # Company Information Form
    company_name = st.sidebar.text_input("Company Name")
    company_type = st.sidebar.selectbox("Company Type", ["Sermaye Şirketi", "Şahıs Şirketi"])


    # Add person age input if status is "Gerçek Kişi"
    person_age = None
    if company_type == "Şahıs Şirketi":
        person_age = st.sidebar.number_input("Person Age", min_value=18, max_value=100, value=18)
    
    company_age = st.sidebar.number_input("Company Age (Years)", min_value=0, max_value=100, value=0)
    company_sector = st.sidebar.text_input("Sector")
    export_status = st.sidebar.checkbox("Does the company export?")
    
    # Financial Information
    income = st.sidebar.number_input("Annual Income (TL)", min_value=0, value=0)
    expenses = st.sidebar.number_input("Annual Expenses (TL)", min_value=0, value=0)
    last_tax_payment = st.sidebar.date_input("Last Tax Payment Date")
    employee_count = st.sidebar.number_input("Number of Employees", min_value=0, value=0)
    
    # Annual Revenue with dynamic year management
    st.sidebar.subheader("Annual Revenue")
    
    # Initialize session state for years if not exists
    if 'revenue_years' not in st.session_state:
        st.session_state.revenue_years = []
    
    # Add new year
    new_year = st.sidebar.number_input("Add New Year", min_value=1900, max_value=2100, value=datetime.now().year)
    if st.sidebar.button("Add Year") and new_year not in st.session_state.revenue_years:
        st.session_state.revenue_years.append(new_year)
        st.session_state.revenue_years.sort()
        st.experimental_rerun()
    
    # Display and manage existing years
    revenue_data = {}
    years_to_remove = []
    
    for year in st.session_state.revenue_years:
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            revenue = st.number_input(f"{year} Revenue", min_value=0, value=0, key=f"revenue_{year}")
            revenue_data[year] = revenue
        with col2:
            if st.button("🗑️", key=f"remove_{year}"):
                years_to_remove.append(year)
    
    # Remove selected years
    for year in years_to_remove:
        st.session_state.revenue_years.remove(year)
        st.experimental_rerun()
    
    if st.sidebar.button("Add Company"):
        company_data = {
            "name": company_name,
            "type": company_type,
            "yaş": company_age,
            "sektor": company_sector,
            "ihracat_yapiyor_mu": export_status,
            "gelir": income,
            "gider": expenses,
            "son_vergi_odeme_tarihi": last_tax_payment.strftime("%Y-%m-%d"),
            "calisan_sayisi": employee_count,
            "yillik_ciro": revenue_data
        }
        
        # Add person age to company data if it exists
        if person_age is not None:
            company_data["kisi_yasi"] = person_age
        
        with st.spinner("Adding company to database..."):
            try:
                load_company_data(driver, company_data)
                st.sidebar.success("Company added successfully!")
            except Exception as e:
                st.sidebar.error(f"Error adding company: {str(e)}")

