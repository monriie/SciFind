# SciFind — demo semantic search di browser

Aplikasi **React + Tailwind CSS**, seluruh pencarian berjalan di browser. Tidak ada backend inference, Node.js, npm, API key, atau proses training ketika aplikasi digunakan.

## Jalankan sekarang

Dari folder proyek TKI:

```powershell
python -m http.server 8765 --bind 127.0.0.1 --directory interface
```

Buka **http://localhost:8765** melalui Chrome atau Edge terbaru. Python hanya menyajikan file statis pada preview; tidak menerima atau menghitung query. Sesudah deploy, static hosting menggantikan server preview ini. Membuka `index.html` langsung melalui `file://` tidak didukung karena aplikasi memakai ES modules, Web Worker, dan pemuatan model.

## Dua model dan mode perbandingan

- **Pretrained:** bobot awal all-MiniLM-L6-v2 dan embedding corpus yang dibuat dengan bobot tersebut.
- **Fine-tuned:** checkpoint `outputs/scifact_laptop/model_final`, tiga epoch MNRL, dan embedding corpus asli dari notebook.
- **Bandingkan keduanya:** query yang sama diproses secara terpisah, lalu Top-K ditampilkan berdampingan. Tabel perubahan posisi melacak artikel menggunakan document ID. Masing-masing model mempunyai alur dan rincian cosine sendiri.

Skor cosine dua encoder tidak boleh langsung ditafsirkan sebagai kenaikan relevansi. Untuk query benchmark yang tersedia dan cocok persis, aplikasi menghitung nDCG@10, AP@100, dan Recall@100 berdasarkan qrels. Untuk satu query, namanya AP, bukan MAP. Query bebas tidak diberi ground truth buatan.

## Urutan demo

1. Pilih model atau mode perbandingan.
2. Ketik query Inggris sendiri atau pilih chip contoh, lalu klik **Telusuri artikel**.
3. Baca judul, cosine, dan potongan abstrak sumber.
4. Buka **Alur perhitungan** untuk melihat input asli/normalisasi, token dan ID, pemotongan input, embedding 384 dimensi, distribusi cosine seluruh corpus, dan ranking.
5. Klik artikel lalu **Dari mana skor ini?** untuk melihat delapan kontribusi dimensi terbesar, jumlah kontribusi dimensi lain, dan total cosine.
6. Pada mode perbandingan, klik **Lihat alur model ini** untuk memilih alur pretrained atau fine-tuned.
7. **Ekspor hasil** mengunduh JSON lokal yang berisi query, vektor, ranking dan hasil tahap perhitungan. Mode perbandingan mengekspor kedua hasil.
8. Halaman **Hasil eksperimen** menampilkan evaluasi 300 query test dari notebook. Angka tersebut terpisah dari hasil query bebas pengguna.

Pencarian memakai exact dot product terhadap **5.183 vektor dokumen 384 dimensi** yang dinormalisasi L2. Skor ini sama dengan cosine similarity, tidak memakai skor acak atau pencarian kata kunci pengganti.

## Deploy sebagai situs statis

Jalankan persiapan aset dan packaging hanya jika model/data/sumber berubah. Folder **`dist/`** yang dihasilkan oleh `package_static.py` merupakan hasil final yang bisa diunggah ke hosting statis. Tidak ada build command Node.js. Seluruh dependency browser, CSS hasil kompilasi, dua model ONNX, corpus, dan embedding ikut disertakan.

```powershell
python interface/package_static.py
```

Upload **isi `interface/dist/`** dan pertahankan struktur direktorinya. Situs tidak memakai router berbasis path, sehingga dapat ditempatkan di subfolder. Gunakan HTTP/HTTPS dan hosting yang menyajikan `.js`, `.mjs`, `.wasm`, `.json`, `.onnx`, serta `.f32` sebagai file asli, bukan mengalihkannya ke HTML. MIME penting: `.js/.mjs` JavaScript dan `.wasm` `application/wasm`.

**Perhatikan kapasitas:** setiap model FP32 sekitar 91 MB. Hosting harus mengizinkan ukuran file tersebut. Beberapa layanan static hosting memiliki batas ukuran file yang lebih kecil; jangan memilihnya tanpa memeriksa batas tersebut. Paket total juga mencakup dua embedding, corpus, dan runtime. Aplikasi ini mempertahankan FP32 agar hasil mendekati notebook; tidak diam-diam memakai model quantized yang dapat mengubah ranking.

Pemuatan pertama lebih berat daripada aplikasi biasa: satu model ±91 MB dan runtime WebAssembly. Mode perbandingan memuat dua model dan membutuhkan lebih banyak memori. Setelah siap, pencarian berikutnya menggunakan model dalam memori. Durasi bergantung pada perangkat. Tab yang ditutup akan melepas memori model; browser dapat menyimpan cache file sesuai kebijakannya.

## Persiapan ulang aset (hanya pengembangan)

Aset browser sudah disiapkan pada folder ini. Script Python berikut diperlukan hanya untuk membuat ulang file hasil konversi, bukan ketika website digunakan atau dideploy:

```powershell
python -m pip install --target interface/.build/python --no-deps onnx==1.19.1 onnxruntime==1.23.2 protobuf ml_dtypes flatbuffers
python interface/prepare_assets.py --download
python interface/prepare_assets.py --export
python interface/prepare_assets.py --pretrained
```

Script ekspor juga memakai PyTorch, Sentence Transformers, dan NumPy dari environment penelitian. Pretrained diambil dari cache Hugging Face pada revision yang tercatat di eksperimen. Jika cache belum tersedia, unduh checkpoint sesuai revision terlebih dahulu. Jangan mengubah revision tanpa membuat ulang embedding pasangannya.

Tailwind menggunakan compiler standalone resmi, tanpa instalasi Node.js/npm terpisah. Dari folder `interface`:

```powershell
.\.build\tailwindcss.exe -i styles.input.css -o styles.css --minify
```

Jalankan kompilasi kembali jika menambah class Tailwind. `styles.css` hasil kompilasi sudah disertakan, sehingga tidak ada compiler atau Tailwind CDN saat aplikasi dipakai.

## Validasi

- Ekspor ONNX diperiksa dengan ONNX checker dan dibandingkan dengan output PyTorch pada beberapa input, termasuk input panjang.
- `data/export-validation-*.json` menyimpan selisih maksimum embedding hasil ekspor masing-masing model.
- Halaman **Tentang penelitian → Jalankan verifikasi** membandingkan embedding dan Top-5 di browser terhadap tiga query acuan hasil Python untuk kedua model.
- `verify.html` menyediakan pemeriksaan integrasi browser yang juga mencakup query kosong, truncation, urutan ranking dan penjumlahan cosine.
- `python interface/validate_assets.py` memeriksa dimensi, normalisasi, pasangan model/data, serta ranking acuan dari embedding yang disertakan.

Pemeriksaan konversi di Python tidak otomatis membuktikan semua browser bekerja sama. Jalankan verifikasi browser pada perangkat demo sebelum presentasi. Browser preview otomatis pada sesi pembuatan memiliki kendala akses, sehingga inspeksi visual dan integrasi WASM di browser perlu diperiksa melalui halaman verifikasi tersebut.

## Sumber dan batas penjelasan

Corpus: [BEIR–SciFact](https://huggingface.co/datasets/BeIR/scifact), **CC BY-SA 4.0**, dan [qrels SciFact](https://huggingface.co/datasets/BeIR/scifact-qrels). Model awal: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), Apache 2.0. Fine-tuned adalah checkpoint penelitian Kelompok 6, bukan model lain dari internet. File sumber tidak diubah; teks hanya disusun untuk pengiriman statis, dan embedding ditulis sebagai float32 little-endian.

Lisensi dependency ada di `vendor/licenses/`. Implementasi menggunakan React 18.3.1, HTM 3.1.1 untuk menulis komponen tanpa compiler JSX, Tailwind 3.4.17, Transformers.js 3.7.2, dan ONNX Runtime Web sesuai dependency versi tersebut.

Rincian kontribusi dimensi menjelaskan **asal hitungan**, bukan penjelasan linguistik atau bukti ilmiah. Dimensi embedding bukan kata. Cosine bukan probabilitas relevansi. Artikel yang membantah query dapat tetap relevan. Aplikasi tidak memberikan jawaban medis atau mengklasifikasikan klaim SUPPORT/REFUTE.
