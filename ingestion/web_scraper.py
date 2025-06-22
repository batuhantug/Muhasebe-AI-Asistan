import requests
from bs4 import BeautifulSoup
import json

def download_and_parse_tax_law_data(common_title, url):
    """
    Download HTML from URL, parse it with BeautifulSoup,
    extract embedded JSON data, and return a dict with common title and data.
    """

    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    

    data = json.loads(soup.text.strip())

    # Return wrapped data with one common title
    return {
        "title": common_title,
        "data": data
    }

def clean_html_tags_from_data(json_data):
    """
    Removes HTML tags from the 'icerik' field in each item of json_data['data'].
    Returns the cleaned JSON data.
    """
    for item in json_data.get("data", []):
        html_content = item.get("icerik", "")
        # Parse with BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        # Extract clean text
        clean_text = soup.get_text(separator=" ", strip=True)
        # Replace the 'icerik' field with cleaned text
        item["icerik"] = clean_text
    return json_data

def create_law_graph(tx, law_title, provisions, embedding_model):
    
    tx.run("MERGE (law:Law {title: $law_title})", law_title=law_title)

    for provision in provisions:
        nid = provision.get('nid')
        baslik = provision.get('baslik')
        icerik = provision.get('icerik')
        no = provision.get('no')
        ad = provision.get('ad')
        tip = provision.get('tip')
        bolum = provision.get('bolum')

        # Get embedding vector using OpenAIEmbeddings
        embedding = embedding_model.embed_query(icerik)

        tx.run("""
            MERGE (law:Law {title: $law_title})
            MERGE (provision:Provision {nid: $nid})
            SET provision.baslik = $baslik,
                provision.icerik = $icerik,
                provision.no = $no,
                provision.ad = $ad,
                provision.tip = $tip,
                provision.bolum = $bolum,
                provision.embedding = $embedding
            MERGE (provision)-[:PART_OF]->(law)
        """, law_title=law_title, nid=nid, baslik=baslik, icerik=icerik,
           no=no, ad=ad, tip=tip, bolum=bolum, embedding=embedding)
        
def create_provision_vector_index(driver, dimensions=1536, similarity_function='cosine'):
    query = f"""
    CREATE VECTOR INDEX tax_law_vector_index
    FOR (p:Provision) ON (p.embedding)
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {dimensions},
            `vector.similarity_function`: '{similarity_function}'
        }}
    }}
    """
    with driver.session() as session:
        session.run(query)
        print("Vector index created for tax files.")

def load_gib_data(driver, embedding_model):
    
    data = download_and_parse_tax_law_data("Gelir Vergisi", "https://www.gib.gov.tr/mevzuat_ac/?cmd=getMevzuatIcerikKanunaGore&sk=80385&mevzuat=km")
    
    cleaned_data = clean_html_tags_from_data(data)
    
    with driver.session() as session:
        session.write_transaction(create_law_graph, cleaned_data["title"], cleaned_data["data"], embedding_model)
        
        print("Data with embeddings imported successfully!")
    
    create_provision_vector_index(driver)