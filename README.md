# Quran Better For Me (Static GitHub Pages)

This repository hosts a static website that provides:
- Quran Arabic text + English translation
- Semantic similarity pairs (Arabic-only embeddings)
- Lexical token overlap pairs (Arabic-only tokens)
- Hadith Arabic text display for paired results

No external data scraping is used. Only provided datasets are used.

---

## 1) Put raw datasets in the repo

Create a folder:

raw/
- quran.csv
- Quran_English.csv
- All_Hadith_Clean.csv

(Optionally: quran-simple.xml)

---

## 2) Create a Python environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r scripts/requirements.txt
