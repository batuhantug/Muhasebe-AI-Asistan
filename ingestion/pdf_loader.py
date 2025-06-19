from langchain_community.document_loaders import PyPDFLoader
import json
import os
import re



def clean_newlines_from_chunks(text):
    text = text.replace('\n', ' ').replace('\r', ' ').strip()
    return text

def load_pdf_with_langchain(file_path):
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Combine all pages into one string
    full_text = "\n".join([doc.page_content for doc in documents])
    return full_text

def kosgeb_semantic_chunker(text, client, model="gpt-4o-mini"):
    prompt = f"""
You are a legal document processor specialized in Turkish support programs and regulations.

Task:
- Analyze the following legal text.
- Identify the official name of the support program described in the text.
- Extract important metadata.
- Split the text into semantic chunks.
- Remove all newlines from content.
- Classify chunks.

Extract:
- program_name: The official full name of the support program.
- industries: list of supported sectors.
- min_age and max_age if applicable
- applicant_types: ["Gerçek Kişi", "Limited Şirket", "Anonim Şirket", etc.].
- funding_types: ["Geri Ödemesiz", "Geri Ödemeli", "Hibe", "Kredi", "Teminat"].
- max_support_amount: Maximum total support amount (in TL).
- support_duration: Support period (in months or years).
- education_requirements: list of required education/training.
- application_period: application start and end dates.
- required_documents: list of documents required for application.
- evaluation_criteria: how applications are evaluated.

Also split into:
Chunks:
- For each chunk output:
  - title
  - category: one of ["Mevzuat", "Tanım", "Şartlar", "Destek", "Süreç", "Sektör"]
  - content: full content without newlines

Be careful with category selection. Try to extract complete and detailed information even if the data is scattered.

Example output format:

{{
  "program_name": "Girişimci Destek Programı",
  "industries": ["Bilişim", "İmalat", "Enerji"],
  "max_age": 65, 
  "min_age": 18,
  "applicant_types": ["Gerçek Kişi", "Limited Şirket", "Anonim Şirket"],
  "funding_types": ["Geri Ödemesiz", "Geri Ödemeli"],
  "max_support_amount": "500000 TL",
  "min_support_amount": "10000 TL",
  "support_duration": "24 Ay",
  "education_requirements": ["KOSGEB Girişimcilik Eğitimi"],
  "application_period": "01/01/2025 - 31/12/2025",
  "required_documents": ["İmza Sirküleri", "Faaliyet Belgesi", "İş Planı"],
  "evaluation_criteria": ["İstihdam etkisi", "İnovasyon seviyesi", "Finansal yeterlilik"],
  "chunks": [
    {{"title": "...", "category": "...", "content": "..." }},
    ...
  ]
}}

Here is the text:
{text}

Output only valid JSON.
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    result = response.choices[0].message.content.strip()

    # CLEAN OUTPUT
    # Remove ```json or ``` block if present
    result_clean = re.sub(r"```(?:json)?", "", result).strip().rstrip("```").strip()

    try:
        data = json.loads(result_clean)
    except Exception as e:
        data = {"error": str(e), "raw_response": result}
    return data

def store_program_with_full_data(program_data, driver, embedding_model):
    
    program_name = program_data["program_name"]
    industries = program_data.get("industries", [])
    min_age = program_data.get("min_age", None)
    max_age = program_data.get("max_age", None)
    applicant_types = program_data.get("applicant_types", [])
    funding_types = program_data.get("funding_types", [])
    max_support_amount = program_data.get("max_support_amount", None)
    support_duration = program_data.get("support_duration", None)
    education_requirements = program_data.get("education_requirements", [])
    application_period = program_data.get("application_period", None)
    required_documents = program_data.get("required_documents", [])
    evaluation_criteria = program_data.get("evaluation_criteria", [])
    chunks = program_data.get("chunks", [])
    
    with driver.session() as session:

        # PROGRAM NODE
        session.run("""
            MERGE (p:Program {name: $program_name})
            SET p.application_period = $application_period,
                p.support_duration = $support_duration,
                p.max_support_amount = $max_support_amount,
                p.funding_types = $funding_types,
                p.evaluation_criteria = $evaluation_criteria
        """, program_name=program_name,
             application_period=application_period,
             support_duration=support_duration,
             max_support_amount=max_support_amount,
             funding_types=funding_types,
             evaluation_criteria=evaluation_criteria)

        # ELIGIBILITY NODE
        eligibility_title = f"{program_name} için Kriterler"
        session.run("""
            MATCH (p:Program {name: $program_name})
            MERGE (e:Eligibility {program: $program_name})
            SET e.title = $eligibility_title,
                e.min_age = $min_age,
                e.max_age = $max_age,
                e.applicant_types = $applicant_types,
                e.industries = $industries
            MERGE (p)-[:HAS_ELIGIBILITY]->(e)
        """, program_name=program_name,
             eligibility_title=eligibility_title,
             min_age=min_age,
             max_age=max_age,
             applicant_types=applicant_types,
             industries=industries)

        # EDUCATION NODE
        session.run("""
            MATCH (p:Program {name: $program_name})
            MERGE (ed:Education {program: $program_name})
            SET ed.requirements = $education_requirements
            MERGE (p)-[:HAS_REQUIREMENT]->(ed)
        """, program_name=program_name, education_requirements=education_requirements)

        # DOCUMENT NODE
        session.run("""
            MATCH (p:Program {name: $program_name})
            MERGE (doc:Document {program: $program_name})
            SET doc.requirements = $required_documents
            MERGE (p)-[:HAS_REQUIREMENT]->(doc)
        """, program_name=program_name, required_documents=required_documents)

        # CHUNKS WITH EMBEDDINGS
        for chunk in chunks:
            title = chunk["title"]
            category = chunk["category"]
            content = chunk["content"]
            embedding = embedding_model.embed_query(content)
            rel_type = category.replace(" ", "_").replace("-", "_")
            
            session.run(f"""
                MATCH (p:Program {{name: $program_name}})
                MERGE (ch:Chunk {{program: $program_name, title: $title}})
                SET ch.content = $content, ch.embedding = $embedding
                MERGE (p)-[:`{rel_type}`]->(ch)
            """, program_name=program_name, title=title, content=content, embedding=embedding)





def create_pdf_vector_index(driver, dimensions=1536, similarity_function='cosine'):
    query = f"""
    CREATE VECTOR INDEX kosgeb_vector_index
    FOR (c:Chunk) ON (c.embedding)
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {dimensions},
            `vector.similarity_function`: '{similarity_function}'
        }}
    }}
    """
    with driver.session() as session:
        session.run(query)
        print("Vector index created for pdf files.")



def load_data(pdf_path, client, embedding_model, driver):
    # Create vector index for PDF documents

    
    text = load_pdf_with_langchain(pdf_path)
    
    clean_text = clean_newlines_from_chunks(text)
    
    chunks = kosgeb_semantic_chunker(clean_text, client)
    
    store_program_with_full_data(chunks, driver, embedding_model)

    create_pdf_vector_index(driver)

