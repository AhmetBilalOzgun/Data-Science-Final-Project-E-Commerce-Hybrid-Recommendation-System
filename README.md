# E-Commerce Recommendation System
### Veri Bilimi Final Projesi — 2025/2026 Bahar Dönemi

## Proje Ekibi
| Öğrenci No | Ad Soyad |
|------------|----------|
| 1306240133 | Ahmet Bilal Özgün |

---

## Proje Özeti

Olist Brezilya E-Ticaret veri seti üzerinde hibrit ürün öneri sistemi. İki yöntemi birleştirir:

- **Collaborative Filtering**: Kullanıcı-ürün matrisi üzerinde SVD (matris çarpanlara ayırma) ile kişisel tercih tahmini
- **Content-Based Filtering**: Çok dilli cümle gömme modeli (`paraphrase-multilingual-MiniLM-L12-v2`) ve FAISS vektör arama indeksi ile anlamsal ürün benzerliği ve serbest metin araması
- **Hibrit**: `0.6 × CF + 0.4 × CB` ağırlıklı birleşim

### Araştırma Soruları
1. Çok kategorili alıcılar, bölgeye göre ürünleri farklı mı değerlendiriyor?
2. SVD matris çarpanları küresel ortalama tabanına kıyasla ne kadar iyileşme sağlıyor?
3. Anlamsal öneriler gerçek birlikte satın alımlarla ne ölçüde örtüşüyor?

---

## Veri Kaynağı

**Brazilian E-Commerce Public Dataset by Olist**
- Kaynak: [Kaggle — Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- Lisans: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- 8 CSV dosyası, ~100k sipariş, 2016–2018

> Veri dosyaları lisans kısıtları nedeniyle repoya dahil edilmemiştir. Aşağıdaki kurulum adımlarına bakınız.

---

## Kurulum ve Çalıştırma

### 1. Repoyu klonla
```bash
git clone <repo-url>
cd finalProject
```

### 2. Sanal ortamı oluştur
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Bağımlılıkları yükle
```bash
pip install -r requirements.txt
```

> **Not:** `sentence-transformers` PyTorch'u (~800MB) gerektirir. İlk kurulum zaman alabilir.
> `paraphrase-multilingual-MiniLM-L12-v2` modeli (~471MB) ilk çalıştırmada otomatik indirilir.

### 4. Veriyi indir
[Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) sayfasından 8 CSV dosyasını indirip `data/raw/` klasörüne koy:

```
data/raw/
├── olist_orders_dataset.csv
├── olist_order_items_dataset.csv
├── olist_products_dataset.csv
├── olist_customers_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_sellers_dataset.csv
└── olist_geolocation_dataset.csv
```

### 5. Notebook'u çalıştır
```bash
jupyter lab notebook.ipynb
```

Kernel → **Restart & Run All** ile tüm hücreleri çalıştır.

---

## Kullanılan Kütüphaneler

Tam ortam kilidi için `requirements.txt` dosyasındaki exact version pinleri kullanılmalıdır. Ana kütüphaneler:

| Kütüphane | Versiyon | Kullanım |
|-----------|----------|----------|
| pandas | 3.0.1 | Veri yükleme ve işleme |
| numpy | 2.4.3 | Sayısal hesaplamalar |
| matplotlib | 3.11.0 | Görselleştirme |
| seaborn | 0.13.2 | İstatistiksel görselleştirme |
| scikit-learn | 1.9.0 | Ön işleme |
| scipy | 1.18.0 | İstatistiksel testler (Mann-Whitney U) |
| scikit-surprise | 1.1.5 | SVD collaborative filtering |
| faiss-cpu | 1.14.3 | Vektör benzerlik indeksi |
| sentence-transformers | 5.6.0 | Çok dilli anlamsal gömme |
| jupyterlab | 4.5.6 | Notebook çalıştırma |
| reportlab | 5.0.0 | PDF rapor üretimi |

---

## Proje Yapısı

```
finalProject/
├── README.md
├── CLAUDE.md           # AI geliştirme ortamı bağlamı
├── todos.md            # Görev takip listesi
├── design.md           # Mimari ve tasarım kararları
├── notebook.ipynb      # Ana teslim dosyası
├── prompts.md          # AI prompt günlüğü
├── requirements.txt    # Bağımlılıklar
├── data/
│   └── raw/            # Olist CSV dosyaları (gitignore'd)
└── report/
    └── report.pdf      # Kısa rapor (3-5 sayfa)
```

---

## Teslim Tarihi

**3 Temmuz 2026, saat 12:30**
