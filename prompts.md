# Prompt Log — E-Commerce Recommendation System

Chronological log of important AI prompts used during development.
Format: `[Phase] Prompt -> What it produced`

---

## Phase 1 — Project Initialization

**Prompt 1** (2026-06-26):
> "This is my final project for the lecture "Data Science". This is the instruction our teacher gave us: Merhaba arkadaşlar,
> Final sınavı yerine dönem projesi olarak bu dönem boyunca öğrendiğimiz veri bilimi iş akışını (veri toplama, temizleme, keşifsel analiz, görselleştirme ve temel modelleme vb.) gerçek veya gerçekçi bir veri seti üzerinde uygulamanız üzerine bir çalışma sizlerden bekliyorum. Çalışma yöntemi "vibe coding" kullanabilirsiniz: kodu satır satır elle yazmak yerine bir AI kodlama asistanını (Claude, ChatGPT, Copilot, Cursor vb.) birincil geliştirme ortağı olarak kullanacaksınız. Buna karşılık üretilen her çıktıyı anlamak, doğrulamak ve gerekirse sorulan soruları cevaplamak sizin sorumluluğunuzda olacaktır.
> Son teslim tarihi: 3 Temmuz 2026, saat 12:30
> Çalışma şekli: Bireysel veya 2 kişilik gruplar
> Tema havuzu (birini seçin, örnek veri kaynağıyla — verilen kaynaklar sadece örnektir, bunlarla sınırlı kalmak zorunda değilsiniz, aynı temaya uygun başka bir açık veri seti de kullanabilirsiniz):
> * Akıllı Şehir & Toplu Taşıma — İBB Açık Veri Portalı (İETT hat/yolcu verileri)
> * Sağlık & Halk Sağlığı — TÜİK sağlık istatistikleri, Kaggle sağlık veri setleri
> * Finans & Ekonomi — BIST hisse verileri, TCMB döviz kurları
> * E-ticaret & Müşteri Davranışı — Kaggle Online Retail / Brazilian E-Commerce
> * Sosyal Medya & Metin Analizi — açık haber başlığı korpusları, duygu analizi veri setleri
> * Enerji & Çevre — EPDK elektrik tüketim verileri, İBB hava kalitesi verileri
> * Spor Analitiği — açık futbol lig istatistikleri
> * Eğitim & Öğrenci Performansı — OECD PISA verileri
> Teknik kapsam (zorunlu bileşenler):
> * Veri yükleme ve temizleme: eksik veri, aykırı değer, tip dönüşümleri
> * Keşifsel veri analizi (EDA): temel istatistikler, en az 4 farklı görselleştirme türü
> * En az 3 somut araştırma sorusu tanımlanmalı ve analizle yanıtlanmalı
> * Basit modelleme: uygun bir regresyon, sınıflandırma veya kümeleme yöntemi (kapsamlı hiperparametre optimizasyonu beklenmiyor, tek savunulabilir model yeterli)
> * Sonuçların veri setinin teması bağlamında yorumlanması
> İki haftalık süre nedeniyle kapsam yine de dar tutulmalı; derinlik genişlikten önceliklidir.
> Teslim edilecekler:
> * Çalışır notebook (.ipynb): veri yükleme, temizleme, EDA, modelleme adımlarının tümünü içeren, kod bloklarının yorum satırlarıyla açıklandığı tek bir dosya
> * Kısa rapor (PDF, 3-5 sayfa): problem tanımı, kullanılan veri seti ve kaynağı, yöntem, bulgular, sınırlamalar, öğrenilenler
> * Prompt günlüğü: geliştirme sürecinde kullanılan en az 10-15 önemli prompt, kronolojik sırayla ve hangi adımda kullanıldığı belirtilerek
> * README: Proje ekibi bilgileri (Öğrenci numarası, adı soyadı), projeyi çalıştırma talimatları, kullanılan kütüphaneler, veri kaynağının adı ve lisansı
> Teslim formatı:
> * Tüm proje bir GitHub repository linki olarak teslim edilmeli (Github hesabı açıp tüm çalışmanızı orada hazırlayıp link olarak paylaşmalısınız.)
> * Ayrıca hazırladığınız kısa rapor şeffaf dosya içinde çıktı olarak ilgili tarihte dersin öğretim üyesi veya asistanına teslim edilecektir.
> Kod blokları hakkında teslimde kısa sorular sorulabilir, o yüzden AI'nin yazdığı her şeyi anlamış olun.
> Lets
> initalize our project with a strong claude.md that describes our project and our goal, a strong todos.md so we can
> track our progress, a strong design.md to keep the consistency within the code and the tech stack. We will do our
> project in the "E-Commerce and Customer Behavior" since it aligns with my goals in Moodmatch/Catch etc. First lets
> design the solution we will go with, I want to use a mix of collaborative filtering and content based filtering (with a
> vector db). Lets create the plan"
-> Created the project architecture, task checklist, design notes, Olist dataset direction, and hybrid recommendation approach.

**Prompt 2** (2026-06-26):
> "look at the todo.md and complete the phase 1"
-> Completed project setup tasks and initial dependency/file scaffolding.

**Prompt 3** (2026-06-26):
> "I've downloaded the datas and added to the data/raw now install deps"
-> Installed the project dependencies into the configured Python environment.

## Phase 2 — Data Loading & Cleaning

**Prompt 4** (2026-06-26):
> "analyze todo.md- compare it with the current codebase. After that lets make a plan for Data loading and cleaning"
-> Planned Phase 2 notebook implementation against the existing repository state.

**Prompt 5** (2026-06-26):
> "update todos.md after each session"
-> Established the working convention of keeping `todos.md` synchronized with completed phases.

**Prompt 6** (2026-06-26):
> "review the phase 2 implementation we did"
-> Reviewed the data loading, cleaning, joins, null handling, and derived bucket columns.

## Phase 3 — EDA

**Prompt 7** (2026-06-26):
> "Create a plan for phase 3 in todos.md"
-> Planned the required EDA visualizations and interpretation cells.

## Phase 4 — RQ1

**Prompt 8** (2026-06-26):
> "Look at todos.md phase 4. Create a plan to implement it"
-> Planned the regional rating-bias analysis, multi-category buyer flag, and Mann-Whitney U test.

## Phase 5 — RQ2 Collaborative Filtering

**Prompt 9** (2026-06-26):
> "now read todos.md phase 5 and create a plan for implementation"
-> Planned the SVD collaborative-filtering workflow, temporal evaluation, and baseline comparison.

**Prompt 10** (2026-06-26):
> "update todos"
-> Updated checklist progress after Phase 5 planning/implementation work.

## Phase 6 — RQ3 Content-Based Filtering

**Prompt 11** (2026-06-26):
> "read todos.md and write a plan to implement phase 6"
-> Planned sentence-transformer metadata embeddings, FAISS indexing, semantic search, and overlap@5 evaluation.

## GitHub / README

**Prompt 12** (2026-06-26):
> "Now for me to write to github can you give me a description and create a readme.md"
-> Drafted the GitHub project description and README content.

**Prompt 13** (2026-06-26):
> "lets connect our github project https://github.com/AhmetBilalOzgun/Data-Science-Final-Project-E-Commerce-Hybrid-Recommendation-System.git"
-> Configured the repository remote for GitHub submission.

## Phase 9 — Final Polish

**Prompt 14** (2026-06-26):
> "Read claude.md before doing anything. After that read the project files and implement phase 9 in todos.md"
-> Triggered final submission polish planning after reading `CLAUDE.md`, `todos.md`, and existing project files.

**Prompt 15** (2026-06-26):
> "Implement the plan."
-> Applied Phase 9 final polish, verification, and submission steps.
