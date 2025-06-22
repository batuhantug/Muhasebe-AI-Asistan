import streamlit as st

def add_company_neo4j(tx, company):
    # Create Company node
    query = """
    MERGE (c:Company {name: $name})
    SET c.type = $type,
        c.yas = $yas,
        c.ihracat_yapiyor_mu = $ihracat_yapiyor_mu,
        c.gelir = $gelir,
        c.gider = $gider,
        c.son_vergi_odeme_tarihi = $son_vergi_odeme_tarihi,
        c.calisan_sayisi = $calisan_sayisi
    """

    # Add person age if company type 
    if company["type"] == "Sermaye Şirketi" or "Şahıs Şirketi"in company:
        query = query.replace("c.calisan_sayisi = $calisan_sayisi", 
                            "c.calisan_sayisi = $calisan_sayisi, c.kisi_yasi = $kisi_yasi")
    
    tx.run(query,
    name=company["name"],
    type=company["type"],
    yas=company["yaş"],
    ihracat_yapiyor_mu=company["ihracat_yapiyor_mu"],
    gelir=company["gelir"],
    gider=company["gider"],
    son_vergi_odeme_tarihi=company["son_vergi_odeme_tarihi"],
    calisan_sayisi=company["calisan_sayisi"],
    kisi_yasi=company.get("kisi_yasi"))  # Use get() to safely handle optional field

    # Create Sector node and relationship
    tx.run("""
    MATCH (c:Company {name: $name})
    MERGE (s:Sector {name: $sector_name})
    MERGE (c)-[:IN_SECTOR]->(s)
    """,
    name=company["name"],
    sector_name=company["sektor"])

    # Create AnnualRevenue nodes and relationships
    for year, revenue in company["yillik_ciro"].items():
        tx.run("""
        MATCH (c:Company {name: $name})
        MERGE (ar:AnnualRevenue {year: $year})
        SET ar.revenue = $revenue
        MERGE (c)-[:HAS_ANNUAL_REVENUE]->(ar)
        """,
        name=company["name"],
        year=str(year),
        revenue=revenue)

    # Create Relationships for law data
def create_tax_relationship(tx, company):
    company_type = company.get("type", "")
    
    if company_type.startswith("Şahıs Şirketi"):
        law_title = "Gelir Vergisi"
    elif company_type.startswith("Sermaye Şirketi"):
        law_title = "Kurumlar Vergisi"
    else:
        return  # unknown type, skip linking

    tx.run("""
        MATCH (c:Company {name: $company_name})
        MATCH (l:Law {title: $law_title})
        MERGE (c)-[:VERGIYE_TABİ]->(l)
    """, {
        "company_name": company["name"],
        "law_title": law_title
    })

# check company relationships
def check_eligibility_and_link(tx, company_data):
    # Step 1: Fetch all programs and eligibility criteria
    result = tx.run("""
        MATCH (p:Program)-[:HAS_ELIGIBILITY]->(e:Eligibility)
        RETURN p.title AS program_title, e.title AS criteria_title,
               e.applicant_types AS applicant_types,
               e.industries AS industries,
               e.min_age AS min_age,
               e.max_age AS max_age
    """)
    programs = result.data()

    # Step 2: Loop through each program and check rule-based eligibility
    for prog in programs:

        tx.run("""
                MATCH (c:Company {name: $company_name})
                MATCH (e:Eligibility {title: $criteria_title})
                MERGE (c)-[:UYGUN_OLDUĞU_PROGRAM]->(e)
            """, {
                "company_name": company_data["name"],
                "criteria_title": prog["criteria_title"]
            })


def load_company_data(driver, company_data):
    with driver.session() as session:
        session.write_transaction(add_company_neo4j, company_data)
        session.write_transaction(create_tax_relationship, company_data)
        session.write_transaction(check_eligibility_and_link, company_data)
    driver.close()
