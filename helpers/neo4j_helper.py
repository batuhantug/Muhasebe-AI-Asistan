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
def evaluate_company_against_programs(driver, company_data, client):
    with driver.session() as session:
        # 1. Get all programs with eligibility info
        result = session.run("""
            MATCH (p:Program)-[:HAS_ELIGIBILITY]->(e:Eligibility)
            RETURN p.title AS program_title, e.title AS criteria_title,
                   e.applicant_types AS applicant_types,
                   e.industries AS industries,
                   e.min_age AS min_age,
                   e.max_age AS max_age
        """)
        
        programs = result.data()

    for prog in programs:
        # 2. Build eligibility criteria
        criteria = {
            "program": prog["program_title"],
            "title": prog["criteria_title"],
            "applicant_types": prog["applicant_types"] or [],
            "industries": prog["industries"] or [],
            "min_age": prog.get("min_age", 0),
            "max_age": prog.get("max_age", 150)
        }

        # 3. Use LLM to assess match
        prompt = f"""
Aşağıdaki şirketin '{criteria["program"]}' adlı destek programına uygun olup olmadığını değerlendir:

🎯 **Program Kriterleri:**
- Başvuru Tipleri: {', '.join(criteria['applicant_types'])}
- Sektörler: {', '.join(criteria['industries'])}
- Yaş Aralığı: {criteria['min_age']} - {criteria['max_age']}

🏢 **Şirket Bilgileri:**
- Statü: {company_data.get('statü')}
- Tip: {company_data.get('tip')}
- Kişi Yaşı: {company_data.get('kisi_yasi')}
- Sektör: {company_data.get('sektor')}

❓ Bu şirket programa başvuru yapabilir mi? net bir karşılaştırma yapma alakalı cevaplar varsa de evet de. 
Kısaca neden uygun veya neden uygun değil olduğunu belirt. cümlen evet veya hayır ile başlamalı.
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        decision = response.choices[0].message.content.strip().lower()

        # 4. If "evet", create relationship
        if decision.startswith("evet"):
            with driver.session() as session:
                session.run("""
                    MATCH (c:Company {name: $company_name})
                    MATCH (p:Program {title: $program_title})
                    MERGE (c)-[:UYGUN_OLDUĞU_PROGRAM]->(p)
                """, {
                    "company_name": company_data["name"],
                    "program_title": criteria["program"]
                })


def load_company_data(driver, company_data, client):
    with driver.session() as session:
        session.write_transaction(add_company_neo4j, company_data)
        session.write_transaction(create_tax_relationship, company_data)
        session.write_transaction(evaluate_company_against_programs, company_data, client)
    driver.close()
