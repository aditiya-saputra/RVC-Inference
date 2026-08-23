# RVC-Inference

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aditiya-saputra/RVC-Inference/blob/main/RVC_Inference_Colab.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Versi **inference-only** dari [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) — tanpa kode training, plus downloader model dan notebook Google Colab **No UI** (aman dari pembatasan WebUI Colab).

## Isi

| File / Folder | Fungsi |
|---|---|
| `RVC_Inference_Colab.ipynb` | Notebook Colab No UI: clone → download model → konversi → hasil audio |
| `infer/cli.py` | CLI inference (single/batch, multi-speaker) |
| `tools/download_models.py` | Downloader base model (HuBERT, RMVPE) + voice model (.pth/.index/.zip) |
| `webui.py` | WebUI lokal (inference saja: konversi, pemisahan vokal, download model) |

## Pakai di Google Colab

Klik badge **Open In Colab** di atas → runtime GPU → *Run all*.
Notebook otomatis clone repo ini (`REPO_URL` sudah mengarah ke sini), download base model, lalu tinggal upload voice model `.pth`/`.index`/`.zip` dan audio input.

> ⚠️ Colab free tier memblokir usage terkait RVC (*runtime disconnected — disallowed usage*). Jika kena, gunakan **Kaggle**:

## Pakai di Kaggle (alternatif gratis, tanpa blokir)

Gunakan `RVC_Inference_Kaggle.ipynb`:
1. Verifikasi akun dengan nomor HP di https://www.kaggle.com/settings
2. Buat Notebook → Settings: **Accelerator = GPU T4 x2**, **Internet = On**
3. File → Import Notebook → `RVC_Inference_Kaggle.ipynb`
4. Run all — voice model bisa diisi via `MODEL_URL` (URL halaman HuggingFace langsung bisa) atau di-upload sebagai Kaggle Dataset
5. Hasil muncul di panel **Output** `/kaggle/working/output`

Kuota GPU Kaggle: ±30 jam/minggu gratis.

## Pakai lokal (CLI)

```bash
python tools/download_models.py --base                 # HuBERT + RMVPE
python tools/download_models.py --url <url_model.pth>  # voice model
python infer/cli.py --model assets/weights/model.pth --input in.wav --output out.wav
```

## Pakai lokal (WebUI)

```bash
python webui.py
```

## Catatan

- Base model diunduh dari HuggingFace `lj1995/VoiceConversionWebUI`; mirror bisa dipakai via env `RVC_HF_ENDPOINT` (mis. `https://hf-mirror.com`)
- Training telah dihapus sepenuhnya dari fork ini
