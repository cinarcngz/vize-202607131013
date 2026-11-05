# vize-202607131013
Akıllı Şehir RAG Bilgi Asistanı ve Anomali Tespiti Projesi

# Akıllı Şehir RAG Bilgi Asistanı ve Anomali Tespiti Projesi

## Proje Hakkında
Bu proje, akıllı şehirlerde kullanılan sensör verilerini analiz etmek ve şehirle ilgili bilgilere erişimi kolaylaştırmak amacıyla geliştirilmiştir.

Proje iki temel bölümden oluşmaktadır:

1. **RAG Bilgi Asistanı:** Şehir bilgilerini içeren dokümanlardan TF-IDF ve benzerlik analizi kullanarak sorulara ilgili cevapları bulmayı amaçlar.
2. **Anomali Tespiti:** Trafik ve sensör verilerini analiz ederek olağan dışı değerleri tespit eder.

## Kullanılan Teknolojiler
- Python
- Pandas
- Scikit-learn
- TF-IDF
- Cosine Similarity
- Pytest
- GitHub Actions

## Proje Dosyaları
- `app.py`: Ana uygulama ve analiz fonksiyonları
- `data/sehir_verileri.csv`: Şehir sensör verileri
- `docs/sehir_bilgisi.md`: Şehir bilgi dokümanı
- `tests/test_app.py`: Otomatik testler
- `requirements.txt`: Proje bağımlılıkları

## Kurulum
```bash
pip install -r requirements.txt
```

## Testleri Çalıştırma
```bash
python -m pytest tests/ -v
```

## Otomatik Test
GitHub Actions kullanılarak kod değişikliklerinde otomatik testler çalıştırılır.

## Proje Amacı
Akıllı şehirlerde veri analizi, bilgiye erişim ve anomali tespiti süreçlerini Python kullanarak örneklemektir.
