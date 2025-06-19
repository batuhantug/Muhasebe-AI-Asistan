import streamlit as st
import os
from ingestion.pdf_loader import load_data
from ingestion.web_scraper import load_gib_data
from helpers.neo4j_helper import load_company_data
from helpers.llm import get_llm, determine_vector_index

from openai import OpenAI
from neo4j import GraphDatabase
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import json
from datetime import datetime
from langchain.prompts import PromptTemplate
from helpers.llm import check_company_eligibility
from helpers.llm import llm


llm = llm()

# Custom conditions for LLM responses
CUSTOM_CONDITIONS = {
    "default": {
        "temperature": 0.2,
        "max_tokens": 1000,
        "response_style": "professional",
        "language": "turkish"
    }
    # Add more conditions later
}

def get_llm_conditions(query):
    """
    Determine which conditions to apply based on the query.
    This function can be expanded later with more sophisticated logic.
    """
    # For now, return default conditions
    return CUSTOM_CONDITIONS["default"]


def is_kosgeb_related(query):
    system_prompt = (
        "Sen bir sınıflandırma asistanısın. Kullanıcının sorusunun KOSGEB destekleri, hibeleri, teşvikleriyle ilgili olup olmadığını değerlendir.\n"
        "Sadece 'Evet' veya 'Hayır' olarak yanıt ver."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    answer = response.choices[0].message.content.strip().lower()

    return answer.startswith("evet")

def is_tax_related(query):
    system_prompt = (
        "Sen bir sınıflandırma asistanısın. Kullanıcının sorusunun Vergi kanunları ile ilgili olup olmadığını değerlendir.\n"
        "Sadece 'Evet' veya 'Hayır' olarak yanıt ver."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    answer = response.choices[0].message.content.strip().lower()

    return answer.startswith("evet")

def if_it_start_with_company(query, llm_instance):
    """
    Use LLM to check if the query starts with a company name and return the company name if found.
    """
    prompt_template = PromptTemplate(
        input_variables=["query"],
        template="""
        Aşağıdaki sorunun başında şirket ismi var mı kontrol et.
        Eğer başında şirket ismi varsa, sadece şirket ismini yaz.
        Eğer başında şirket ismi yoksa 'YOK' yaz.
        Sadece şirket ismini veya 'YOK' yaz, başka bir şey yazma.

        Örnekler:
        "ABC şirketi hakkında bilgi ver" -> "ABC"
        "XYZ firmasının cirosu nedir" -> "XYZ"
        "DEF Ltd. Şti. hakkında" -> "DEF"
        "GHI A.Ş. ile ilgili" -> "GHI"
        "Vergi hakkında bilgi ver" -> "YOK"
        "Şirketler hakkında" -> "YOK"

        Soru: {query}
        """
    )
    
    formatted_prompt = prompt_template.format(query=query)
    response = llm_instance.invoke(formatted_prompt).content.strip()
    
    return None if response.upper() == "YOK" else response

def extract_company_name(query, llm_instance):
    """
    Use LLM to extract company name from the query.
    """
    prompt_template = PromptTemplate(
        input_variables=["query"],
        template="""
        Aşağıdaki sorudan şirket ismini çıkar. Eğer şirket ismi yoksa 'YOK' yaz.
        Sadece şirket ismini veya 'YOK' yaz, başka bir şey yazma.

        Soru: {query}
        """
    )
    
    formatted_prompt = prompt_template.format(query=query)
    response = llm_instance.invoke(formatted_prompt).content.strip()
    
    return None if response.upper() == "YOK" else response

def get_company_relationships(driver, company_name):
    """
    Get all relationships and related information for a company.
    """
    with driver.session() as session:
        # First try exact match
        result = session.run("""
        MATCH (c:Company)
        WHERE c.name = $company_name
        OPTIONAL MATCH (c)-[r]->(related)
        OPTIONAL MATCH (c)-[:IN_SECTOR]->(s:Sector)
        OPTIONAL MATCH (c)-[:HAS_ANNUAL_REVENUE]->(ar:AnnualRevenue)
        RETURN c.name as name,
               c.statu as statu,
               c.yas as yas,
               s.name as sector,
               collect(distinct {type: type(r), related: related.name}) as relationships,
               collect(distinct {year: ar.year, revenue: ar.revenue}) as revenues
        """, company_name=company_name)
        
        company_data = result.single()
        
        # If no exact match, try fuzzy search
        if not company_data:
            result = session.run("""
            MATCH (c:Company)
            WHERE c.name CONTAINS $company_name
            OPTIONAL MATCH (c)-[r]->(related)
            OPTIONAL MATCH (c)-[:IN_SECTOR]->(s:Sector)
            OPTIONAL MATCH (c)-[:HAS_ANNUAL_REVENUE]->(ar:AnnualRevenue)
            RETURN c.name as name,
                   c.statu as statu,
                   c.yas as yas,
                   s.name as sector,
                   collect(distinct {type: type(r), related: related.name}) as relationships,
                   collect(distinct {year: ar.year, revenue: ar.revenue}) as revenues
            LIMIT 1
            """, company_name=company_name)
            company_data = result.single()
        
        return company_data

# Initialize memory
def initialize_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

neo4j_uri = st.secrets["neo4j"]["uri"]
neo4j_user = st.secrets["neo4j"]["user"]
neo4j_password = st.secrets["neo4j"]["password"]

openai_api_key = st.secrets["openai"]["api_key"]

client = OpenAI(
    api_key=openai_api_key
) # e.g., your OpenAI or HuggingFace client

embedding_model = OpenAIEmbeddings (model="text-embedding-3-small", openai_api_key=openai_api_key)
# Initialize Neo4j driver (IMPORTANT: you forgot this part)
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

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
                # Check if query contains a company name and KOSGEB support question
                query_lower = prompt.lower()
                is_kosgeb_related = is_kosgeb_related(query_lower)
                
                # Initialize response variable
                response = None
                
                # Get LLM instance
                llm_instance, system_prompt = get_llm(
                    model_name="gpt-4o-mini",
                    style="professional",
                    query=prompt
                )
                
                # Check if query starts with company name
                company_name = if_it_start_with_company(prompt, llm_instance)
                if company_name:
                    # Get company relationships
                    company_data = get_company_relationships(driver, company_name)
                    if not company_data:
                        response = f"{company_name} şirketi veritabanında bulunamadı."
                    else:
                        # Create a prompt template for company analysis
                        prompt_template = PromptTemplate(
                            input_variables=["company_data", "question"],
                            template="""
                            Sen bir muhasebe ve vergi uzmanısın. Aşağıdaki şirket bilgilerini analiz ederek soruyu yanıtla.

                            Şirket Bilgileri:
                            - İsim: {company_data[name]}
                            - Statü: {company_data[statu]}
                            - Yaş: {company_data[yas]}
                            - Sektör: {company_data[sector]}

                            İlişkiler:
                            {relationships}

                            Yıllık Ciro Bilgileri:
                            {revenues}

                            Soru: {question}

                            Lütfen verilen bilgilere dayanarak detaylı bir analiz yap. 
                            Özellikle:
                            1. Şirketin sektördeki konumunu değerlendir
                            2. Finansal performansını analiz et
                            3. İlişkilerini ve bağlantılarını açıkla
                            4. Varsa, potansiyel riskleri ve fırsatları belirt
                            """
                        )
                        
                        # Format relationships
                        relationships_text = "\n".join([
                            f"- {rel['type']}: {rel['related']}"
                            for rel in company_data['relationships']
                        ]) if company_data['relationships'] else "İlişki bulunamadı"
                        
                        # Format revenues
                        revenues_text = "\n".join([
                            f"- {rev['year']}: {rev['revenue']} TL"
                            for rev in company_data['revenues']
                        ]) if company_data['revenues'] else "Ciro bilgisi bulunamadı"
                        
                        # Format the prompt
                        formatted_prompt = prompt_template.format(
                            company_data=company_data,
                            question=prompt,
                            relationships=relationships_text,
                            revenues=revenues_text
                        )
                        
                        # Get response from LLM
                        response = llm_instance.invoke(formatted_prompt).content
                
                if is_kosgeb_related:
                    # Search in KOSGEB vector database
                    index_info = {
                        "index_name": "kosgeb_vector_index",
                        "label": "Chunk"
                    }
                    st.sidebar.info(f"Searching in {index_info['index_name']} for {index_info['label']} nodes...")
                    
                    # Search vector database for relevant documents
                    relevant_docs = search_vector_db(driver, prompt, embedding_model)
                    
                    if not relevant_docs:
                        response = "Üzgünüm, KOSGEB destekleri hakkında yeterli bilgi bulamadım. Lütfen önce bazı KOSGEB dokümanları yükleyin."
                    else:
                        # Create a prompt template that includes the retrieved documents
                        prompt_template = PromptTemplate(
                            input_variables=["question", "context"],
                            template="""
                            Sen bir KOSGEB uzmanısın. Aşağıdaki bağlamı kullanarak soruyu yanıtla.

                            Bağlam:
                            {context}

                            Soru: {question}

                            Lütfen verilen bağlama dayanarak net ve doğru bir yanıt ver. 
                            Eğer bir konudan emin değilsen veya bağlamda ilgili bilgi yoksa, bunu belirt.
                            Yanıtını Türkçe olarak ver ve mümkün olduğunca detaylı açıkla."""
                        )
                        
                        # Format the context from retrieved documents
                        context = "\n\n".join([
                            f"Belge {i+1} (İlgi: {doc['score']:.2f}):\n"
                            f"Başlık: {doc['title']}\n"
                            f"Kategori: {doc['category']}\n"
                            f"İçerik: {doc['text']}"
                            for i, doc in enumerate(relevant_docs)
                        ])
                        
                        # Format the prompt
                        formatted_prompt = prompt_template.format(
                            question=prompt,
                            context=context
                        )
                        
                        # Get response from LLM
                        response = llm_instance.invoke(formatted_prompt).content
                

                is_tax_related = is_tax_related(query_lower)

                if is_tax_related:
                    # Search in tax law vector database
                    index_info = {
                        "index_name": "tax_law_vector_index",
                        "label": "Article"
                    }
                    st.sidebar.info(f"Searching in {index_info['index_name']} for {index_info['label']} nodes...")
                    
                    # Search vector database for relevant documents with relationships
                    with driver.session() as session:
                        # Get query embedding
                        query_embedding = embedding_model.embed_query(prompt)
                        
                        # Search with relationships
                        result = session.run("""
                        CALL db.index.vector.queryNodes('tax_law_vector_index', $k, $embedding)
                        YIELD node, score
                        WITH node, score
                        OPTIONAL MATCH (node)-[r]->(related)
                        RETURN node.content as text, score, node.category as category, 
                               node.title as title,
                               collect(distinct {type: type(r), related: related.title}) as relationships
                        ORDER BY score DESC
                        LIMIT $limit
                        """, 
                        k=5,
                        embedding=query_embedding,
                        limit=5
                        )
                        
                        relevant_docs = []
                        for record in result:
                            if record["text"]:
                                relevant_docs.append({
                                    "text": record["text"],
                                    "score": record["score"],
                                    "category": record["category"],
                                    "title": record["title"],
                                    "relationships": record["relationships"]
                                })
                    
                    if not relevant_docs:
                        response = "Üzgünüm, vergi kanunları hakkında yeterli bilgi bulamadım. Lütfen önce vergi kanunları yükleyin."
                    else:
                        # Create a prompt template that includes the retrieved documents and relationships
                        prompt_template = PromptTemplate(
                            input_variables=["question", "context"],
                            template="""
                            Sen bir vergi uzmanısın. Aşağıdaki bağlamı kullanarak soruyu yanıtla.

                            Bağlam:
                            {context}

                            Soru: {question}

                            Lütfen verilen bağlama dayanarak net ve doğru bir yanıt ver. 
                            Eğer bir konudan emin değilsen veya bağlamda ilgili bilgi yoksa, bunu belirt.
                            Yanıtını Türkçe olarak ver ve mümkün olduğunca detaylı açıkla.
                            Özellikle:
                            1. İlgili vergi kanunu maddelerini ve yönetmelikleri belirt
                            2. İlgili maddelerin birbiriyle olan ilişkilerini açıkla
                            3. Varsa, ilgili yönetmelik ve tebliğleri belirt"""
                        )
                        
                        # Format the context from retrieved documents with relationships
                        context = "\n\n".join([
                            f"Belge {i+1} (İlgi: {doc['score']:.2f}):\n"
                            f"Başlık: {doc['title']}\n"
                            f"Kategori: {doc['category']}\n"
                            f"İçerik: {doc['text']}\n"
                            f"İlişkiler:\n" + "\n".join([
                                f"- {rel['type']}: {rel['related']}"
                                for rel in doc['relationships']
                            ])
                            for i, doc in enumerate(relevant_docs)
                        ])
                        
                        # Format the prompt
                        formatted_prompt = prompt_template.format(
                            question=prompt,
                            context=context
                        )
                        
                        # Get response from LLM
                        response = llm_instance.invoke(formatted_prompt).content

                # If no response was generated, use default search
                if response is None:
                    # Determine which vector index to use based on the query
                    index_info = determine_vector_index(prompt)
                    st.sidebar.info(f"Searching in {index_info['index_name']} for {index_info['label']} nodes...")
                    
                    # Search vector database for relevant documents
                    relevant_docs = search_vector_db(driver, prompt, embedding_model)
                    
                    # Get LLM instance with custom conditions
                    llm_instance, system_prompt = get_llm(
                        model_name="gpt-4o-mini",
                        style="professional",
                        query=prompt
                    )
                    
                    # Create a prompt template that includes the retrieved documents and chat history
                    prompt_template = PromptTemplate(
                        input_variables=["question", "context", "chat_history"],
                        template=f"""{system_prompt}

                        Önceki Konuşma:
                        {{chat_history}}

                        Bağlam:
                        {{context}}

                        Soru: {{question}}

                        Lütfen verilen bağlama ve konuşma geçmişine dayanarak net ve doğru bir yanıt ver. 
                        Eğer bir konudan emin değilsen veya bağlamda ilgili bilgi yoksa, bunu belirt.
                        Yanıtını Türkçe olarak ver ve mümkün olduğunca detaylı açıkla."""
                    )
                    
                    if not relevant_docs:
                        response = "Üzgünüm, veritabanında sorunuzu yanıtlamak için yeterli bilgi bulamadım. Lütfen önce bazı dokümanlar yükleyin veya sorunuzu yeniden ifade edin."
                    else:
                        # Format the context from retrieved documents
                        context = "\n\n".join([
                            f"Belge {i+1} (İlgi: {doc['score']:.2f}):\n"
                            f"Başlık: {doc['title']}\n"
                            f"Kategori: {doc['category']}\n"
                            f"İçerik: {doc['text']}"
                            for i, doc in enumerate(relevant_docs)
                        ])
                        
                        # Get chat history from memory
                        chat_history = st.session_state.memory.buffer
                        
                        # Format the prompt
                        formatted_prompt = prompt_template.format(
                            question=prompt,
                            context=context,
                            chat_history=chat_history
                        )
                        
                        # Get response from LLM
                        response = llm_instance.invoke(formatted_prompt).content
                
                # Save to memory
                st.session_state.memory.save_context(
                    {"input": prompt},
                    {"output": response}
                )
                
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
    company_status = st.sidebar.selectbox("Company Status", ["Gerçek Kişi", "Tüzel Kişi"])
    
    # Add person age input if status is "Gerçek Kişi"
    person_age = None
    if company_status == "Gerçek Kişi":
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
        st.session_state.revenue_years = [2020, 2021, 2022, 2023]
    
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
            "statü": company_status,
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

