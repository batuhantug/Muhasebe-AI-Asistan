# MuhasebAI - Muhasebe ve Vergi Asistanı

MuhasebAI, muhasebe ve vergi konularında uzmanlaşmış bir yapay zeka asistanıdır. KOSGEB destekleri, vergi kanunları ve şirket bilgileri hakkında akıllı yanıtlar sunar.

## Özellikler

- 🤖 Akıllı Sohbet Arayüzü
- 📚 KOSGEB Destekleri Bilgi Tabanı
- 📜 Vergi Kanunları Entegrasyonu
- 🏢 Şirket Bilgileri Yönetimi
- 🔍 Vektör Tabanlı Arama
- 💾 Neo4j Grafik Veritabanı Entegrasyonu

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Neo4j veritabanını kurun ve yapılandırın:
   - Neo4j Desktop veya Neo4j Aura kullanabilirsiniz
   - Vektör indekslerini oluşturun

3. OpenAI API anahtarınızı ayarlayın:
   - `.streamlit/secrets.toml` dosyasını oluşturun
   - API anahtarınızı ekleyin:
```toml
[openai]
api_key = "your-api-key"

[neo4j]
uri = "your-neo4j-uri"
user = "your-username"
password = "your-password"
```

## Kullanım

1. Uygulamayı başlatın:
```bash
streamlit run app.py
```

2. Tarayıcınızda `http://localhost:8501` adresine gidin

3. Özellikler:
   - PDF Yükleme: KOSGEB ve vergi dokümanlarını yükleyin
   - Sohbet: Muhasebe ve vergi konularında sorular sorun
   - Şirket Yönetimi: Şirket bilgilerini ekleyin ve yönetin

## Veritabanı Şeması

### Node Tipleri
- `Chunk`: KOSGEB doküman parçaları
- `Article`: Vergi kanunu maddeleri
- `Company`: Şirket bilgileri
- `Sector`: Sektör bilgileri
- `Eligibility`: KOSGEB uygunluk kriterleri

### İlişkiler
- `(Company)-[:IN_SECTOR]->(Sector)`
- `(Company)-[:IS_ELIGIBLE_FOR]->(Eligibility)`

## Vektör İndeksleri

- `kosgeb_vector_index`: KOSGEB dokümanları için
- `tax_law_vector_index`: Vergi kanunları için

## Geliştirme

### Yeni Özellik Ekleme
1. İlgili modülü belirleyin
2. Gerekli fonksiyonları ekleyin
3. Streamlit arayüzünü güncelleyin
4. Test edin

### Veritabanı Güncelleme
1. Neo4j sorgularını hazırlayın
2. Vektör indekslerini güncelleyin
3. Veri yükleme işlemlerini test edin

## Katkıda Bulunma

