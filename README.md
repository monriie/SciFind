# SciFind

Semantic search for scientific abstracts using Sentence-BERT MiniLM, fine-tuned with Multiple Negatives Ranking Loss on BEIR–SciFact. Includes research notebooks and a React/Tailwind browser demo comparing pretrained and fine-tuned retrieval.

## Alur penelitian

**SciFact → pemeriksaan dan preprocessing → pasangan query–dokumen positif → fine-tuning MNRL → embedding → cosine similarity → ranking Top-K → evaluasi.**

- Model awal: `sentence-transformers/all-MiniLM-L6-v2`.
- Koleksi pencarian: 5.183 dokumen, masing-masing berupa judul dan abstrak.
- Pelatihan: 809 query train, 919 pasangan positif.
- Pengujian: 300 query test dengan 339 pasangan relevansi.
- Evaluasi: nDCG@10, MAP@100, dan Recall@100.
- Bahasa eksperimen dan query yang ditujukan: Inggris.

Query test tidak digunakan untuk fine-tuning. Seluruh corpus tetap menjadi kandidat pencarian. Sistem mengambil dokumen, belum mengklasifikasikan kebenaran klaim atau menghasilkan jawaban medis.

## Struktur repository

```text
TKI_SciFact_Laptop.ipynb   # Training, retrieval, evaluasi dan analisis di laptop
colab/
  TKI_SciFact.ipynb        # Notebook untuk environment Colab
requirements.txt          # Dependensi penelitian dan Jupyter
interface/
  index.html              # Halaman masuk aplikasi
  app.js                  # Komponen React dan tampilan perbandingan
  search-engine.js        # Encoder dan retrieval dalam Web Worker
  retrieval.js            # Cosine, ranking dan rincian skor
  styles.css              # CSS Tailwind yang sudah dikompilasi
  prepare_assets.py       # Siapkan library browser, ONNX dan embedding
  package_static.py       # Buat folder deploy statis dan ZIP
  validate_assets.py      # Pemeriksaan model, embedding dan ranking acuan
  verify.html             # Pemeriksaan integrasi di browser
```

Folder `md/`, `outputs/`, model ONNX, embedding, corpus hasil ekspor, library vendor, build tools, dan ZIP **tidak disimpan dalam Git**. Script pembantu untuk membuat dokumen/notebook juga dikecualikan. File lokal tetap ada; `.gitignore` hanya mengatur pelacakan Git.

## Jalankan penelitian di laptop

Contoh Windows PowerShell dari folder repository:

```powershell
python -m venv .venv-scifact
.\.venv-scifact\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-scifact\Scripts\python.exe -m jupyter lab
```

Buka `TKI_SciFact_Laptop.ipynb` dan gunakan kernel environment tersebut. Jika dependensi sudah dipasang melalui `requirements.txt`, lanjut dari cell 02; cell 01 notebook menyediakan instalasi yang sama. Jalankan berurutan sampai ekspor artefak.

Notebook otomatis memakai CUDA bila tersedia atau CPU jika tidak. Konfigurasi laptop menggunakan tiga epoch dan train batch delapan. Internet dibutuhkan untuk unduhan awal model dan dataset serta pengecekan revision Hugging Face. Hasil disimpan pada `outputs/scifact_laptop/` di direktori kerja kernel.

## Demo browser

Interface menyediakan **Pretrained**, **Fine-tuned**, dan **Bandingkan keduanya**. Setiap model memakai embedding corpus yang sesuai. Demo menampilkan token, embedding, distribusi cosine, ranking, abstrak, dan rincian perhitungan skor.

**Setelah clone, aset besar belum tersedia.** Jalankan notebook laptop, kemudian ikuti langkah persiapan aset pada [panduan interface](interface/README.md#persiapan-ulang-aset-hanya-pengembangan). Persiapan membutuhkan Python dan unduhan dependency, tetapi aplikasi yang sudah disiapkan tidak membutuhkan backend Node.js, npm, atau API inference.

Sesudah aset tersedia:

```powershell
python -m http.server 8765 --bind 127.0.0.1 --directory interface
```

Buka `http://localhost:8765`. Python hanya menyajikan file statis; pencarian diproses di browser. Jangan membuka `index.html` melalui `file://`.

Untuk deploy:

```powershell
python interface/package_static.py
```

Unggah isi `interface/dist/` ke hosting statis yang mendukung ukuran file model sekitar 91 MB per model. Rincian, lisensi, dan pemeriksaan browser tersedia pada [README interface](interface/README.md).

## Hasil eksperimen laptop

| Model | nDCG@10 | MAP@100 | Recall@100 |
|---|---:|---:|---:|
| Pretrained | 0,645082 | 0,603074 | 0,925000 |
| Fine-tuned MNRL | 0,696376 | 0,646635 | 0,956000 |

Hasil tersebut berasal dari satu run laptop: tiga epoch, batch delapan, seed 42. Ketiga nilai rata-rata meningkat, tetapi tidak semua query membaik. Checkpoint final belum dipilih menggunakan validation, sehingga belum dinyatakan sebagai checkpoint terbaik. Skor cosine adalah kemiripan vektor, bukan probabilitas relevansi. Dukungan query bahasa Indonesia belum dievaluasi.

## Tim

Charlistio Aditirta Wijaya · Andreas Calvin · Monica Amrina Rosyada · Fitran Husein

## Sumber

- [Model all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — Apache 2.0.
- [Dataset BEIR–SciFact](https://huggingface.co/datasets/BeIR/scifact) dan [qrels](https://huggingface.co/datasets/BeIR/scifact-qrels) — CC BY-SA 4.0.
- Atribusi dan lisensi dependency: [NOTICE interface](interface/NOTICE.md).

Lisensi model, dataset, dan dependency mengikuti sumber masing-masing; dokumen ini tidak menetapkan lisensi baru untuk seluruh kode repository.
