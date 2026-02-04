# %% [1] Imports & config
import os
import re
import json
import math
import hashlib
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

# Embeddings
from sentence_transformers import SentenceTransformer

# Semantic NN
#import faiss


# Input datasets (your uploaded paths)
QURAN_AR_PATH = "../raw/quran.csv"
QURAN_EN_PATH = "../raw/Quran_English.csv"
HADITH_PATH   = "../raw/All_Hadith_Clean.csv"

# Output repo folders (relative to repo root)
OUT_DATA_DIR = "../data"
OUT_META_DIR = os.path.join(OUT_DATA_DIR, "meta")
OUT_QURAN_TEXT_DIR  = os.path.join(OUT_DATA_DIR, "quran_text")
OUT_QURAN_PAIRS_DIR = os.path.join(OUT_DATA_DIR, "quran_pairs")
OUT_HADITH_TEXT_DIR = os.path.join(OUT_DATA_DIR, "hadith_text")
OUT_SEARCH_DIR      = os.path.join(OUT_DATA_DIR, "search_index")

os.makedirs(OUT_META_DIR, exist_ok=True)
os.makedirs(OUT_QURAN_TEXT_DIR, exist_ok=True)
os.makedirs(OUT_QURAN_PAIRS_DIR, exist_ok=True)
os.makedirs(OUT_HADITH_TEXT_DIR, exist_ok=True)
os.makedirs(OUT_SEARCH_DIR, exist_ok=True)

# Sharding
HADITH_SHARD_SIZE = 1000  # ~50k -> ~51 shards

# Similarity params
TOPK_QURAN_SEMANTIC = 20
TOPK_HADITH_SEMANTIC = 50
TOPK_QURAN_LEXICAL = 20
TOPK_HADITH_LEXICAL = 50

# Embedding model (Arabic-only passages)
EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBED_BATCH_SIZE = 256

# Vector preview (for UI display only)
VEC_PREVIEW_DIMS = 8

# Under-200MB guidance: keep token lists short in pair outputs
MAX_SHARED_TOKENS_STORED = 8


# %% [2] Helpers: JSON writing (minified)
def write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))

def surah_to_shard_name(surah: int) -> str:
    return f"{surah:03d}"

def safe_str(x):
    return "" if pd.isna(x) else str(x)


# %% [3] Arabic normalization & tokenization
# Recommended normalization (as you approved):
# - remove harakat/diacritics
# - remove tatweel
# - normalize alef variants (أإآ -> ا)
# - normalize ؤئ -> ء? (we'll normalize hamza carriers moderately)
# - normalize ى -> ي
# - keep ة as ة (do NOT map to ه by default)

AR_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
AR_TATWEEL_RE = re.compile(r"\u0640")
AR_PUNCT_RE = re.compile(r"[^\u0600-\u06FF0-9\s]")  # keep Arabic block + digits + space
AR_MULTI_SPACE_RE = re.compile(r"\s+")

def normalize_ar(text: str) -> str:
    text = safe_str(text)
    text = AR_DIACRITICS_RE.sub("", text)
    text = AR_TATWEEL_RE.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "ء").replace("ئ", "ء")
    # remove non-Arabic punctuation (keep Arabic letters/digits/spaces)
    text = AR_PUNCT_RE.sub(" ", text)
    text = AR_MULTI_SPACE_RE.sub(" ", text).strip()
    return text

def load_stopwords_ar(path_txt: str) -> set:
    sw = set()
    with open(path_txt, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t and not t.startswith("#"):
                sw.add(normalize_ar(t))
    return sw

def tokenize_ar(text: str, stopwords: set) -> list:
    text = normalize_ar(text)
    toks = [t for t in text.split(" ") if t]
    toks2 = []
    for t in toks:
        if t.isdigit():
            continue
        if len(t) < 2:
            continue
        if t in stopwords:
            continue
        toks2.append(t)
    return toks2


# %% [4] English tokenization for GUI search + fuzzy
EN_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
EN_MULTI_SPACE_RE = re.compile(r"\s+")

def normalize_en(text: str) -> str:
    text = safe_str(text).lower()
    text = EN_PUNCT_RE.sub(" ", text)
    text = EN_MULTI_SPACE_RE.sub(" ", text).strip()
    return text

def load_stopwords_en_default() -> set:
    # Minimal built-in list to avoid external downloads;
    # you can expand this if desired.
    return {
        "the","a","an","and","or","but","if","then","than","that","this","those","these",
        "is","are","was","were","be","been","being",
        "of","to","in","on","for","with","as","by","at","from","into","about","over","under",
        "he","she","it","they","them","his","her","its","their","you","your","we","our","us",
        "i","me","my","mine",
        "not","no","nor","so","too","very","can","could","may","might","shall","should","will","would"
    }

def tokenize_en(text: str, stopwords: set) -> list:
    text = normalize_en(text)
    toks = [t for t in text.split(" ") if t]
    toks2 = []
    for t in toks:
        if t.isdigit():
            continue
        if len(t) < 2:
            continue
        if t in stopwords:
            continue
        toks2.append(t)
    return toks2

def trigrams(token: str):
    if len(token) <= 3:
        return {token}
    return {token[i:i+3] for i in range(len(token)-2)}


# %% [5] Load datasets + validate join rule
q_ar = pd.read_csv(QURAN_AR_PATH)
q_en = pd.read_csv(QURAN_EN_PATH)
h_df = pd.read_csv(HADITH_PATH)

# Standardize column names
q_ar.columns = [c.strip().lower() for c in q_ar.columns]
# expected: surah, ayah, arabic_text
# Quran English expected: Surah, Ayat, english_text
q_en.columns = [c.strip().lower() for c in q_en.columns]
# Hadith expected: Serial, Book, Reference, Arabic Text, English Text
h_df.columns = [c.strip().lower() for c in h_df.columns]

# Build IDs
q_ar["surah"] = q_ar["surah"].astype(int)
q_ar["ayah"]  = q_ar["ayah"].astype(int)
q_ar["ayah_id"] = q_ar["surah"].astype(str) + ":" + q_ar["ayah"].astype(str)

q_en["surah"] = q_en["surah"].astype(int)
q_en["ayat"]  = q_en["ayat"].astype(int)
q_en["ayah_id"] = q_en["surah"].astype(str) + ":" + q_en["ayat"].astype(str)

# Validate join coverage
set_ar = set(q_ar["ayah_id"].tolist())
set_en = set(q_en["ayah_id"].tolist())
missing_en = sorted(list(set_ar - set_en))[:10]
missing_ar = sorted(list(set_en - set_ar))[:10]

if missing_en or missing_ar:
    raise ValueError(
        "English/Arabic join mismatch detected.\n"
        f"Arabic IDs missing in English (sample): {missing_en}\n"
        f"English IDs missing in Arabic (sample): {missing_ar}\n"
        "Fix dataset alignment or update join logic."
    )

# Merge for output text usage
q_en_small = q_en[["ayah_id","english_text"]].copy()
q = q_ar.merge(q_en_small, on="ayah_id", how="left")

# Hadith IDs (combined)
h_df["serial"] = h_df["serial"].astype(int)
h_df["hadith_id"] = (
    h_df["book"].astype(str) + "|" +
    h_df["reference"].astype(str) + "|" +
    h_df["serial"].astype(str)
)

# Arabic-only hadith text for embeddings/tokenization
if "arabic text" in h_df.columns:
    hadith_ar_col = "arabic text"
else:
    raise ValueError("Could not find hadith Arabic column 'Arabic Text' (lowercased to 'arabic text').")

# Keep GUI fields minimal (Arabic + metadata)
h_keep = h_df[["hadith_id","serial","book","reference",hadith_ar_col]].copy()
h_keep.rename(columns={hadith_ar_col:"arabic_text"}, inplace=True)

print("Loaded:")
print("Quran rows:", len(q), "Hadith rows:", len(h_keep))


# %% [6] Stopwords files: write Arabic stopwords into repo (edit/extend anytime)
# Bundled stopwords (curated) - you can add more.
STOPWORDS_AR_TXT = os.path.join(OUT_SEARCH_DIR, "stopwords_ar.txt")

if not os.path.exists(STOPWORDS_AR_TXT):
    base_ar = [
        "و","في","على","من","إلى","عن","ما","ماذا","اذا","إن","أن","كان","كانت","يكون","تكون",
        "هذا","هذه","ذلك","تلك","هؤلاء","اولئك","هو","هي","هم","هن","نحن","انت","انتم","أنت",
        "لا","لم","لن","قد","ثم","او","أو","بل","كل","حتى","مع","بين","عند","إذ","اذ","الا","إلا",
        "أي","أى","اي","أين","اين","كيف","لماذا","لما","لأن","لان","إنما","إنه","انه","إنهم","انهم"
    ]
    # normalize + unique
    base_ar_norm = sorted({normalize_ar(x) for x in base_ar if x.strip()})
    with open(STOPWORDS_AR_TXT, "w", encoding="utf-8") as f:
        f.write("# Arabic stopwords (normalized). Add more lines as needed.\n")
        for w in base_ar_norm:
            f.write(w + "\n")

stop_ar = load_stopwords_ar(STOPWORDS_AR_TXT)
stop_en = load_stopwords_en_default()


# %% [7] Tokenize Quran Arabic + Hadith Arabic (for lexical similarity + UI matched tokens)
q["arabic_norm"] = q["arabic_text"].map(normalize_ar)
q["arabic_tokens"] = q["arabic_text"].map(lambda t: tokenize_ar(t, stop_ar))
q["tok_set"] = q["arabic_tokens"].map(lambda xs: set(xs))
q["tok_len"] = q["tok_set"].map(len)

h_keep["arabic_norm"] = h_keep["arabic_text"].map(normalize_ar)
h_keep["arabic_tokens"] = h_keep["arabic_text"].map(lambda t: tokenize_ar(t, stop_ar))
h_keep["tok_set"] = h_keep["arabic_tokens"].map(lambda xs: set(xs))
h_keep["tok_len"] = h_keep["tok_set"].map(len)

print("Avg Quran token count:", float(q["tok_len"].mean()))
print("Avg Hadith token count:", float(h_keep["tok_len"].mean()))


# %% [8] Build English search index + trigram index (fuzzy by query-vs-token)
q["english_tokens"] = q["english_text"].map(lambda t: tokenize_en(t, stop_en))

english_token_to_ayah = defaultdict(list)
for ayah_id, toks in zip(q["ayah_id"], q["english_tokens"]):
    for t in set(toks):
        english_token_to_ayah[t].append(ayah_id)

# trigram index for vocabulary tokens (reduces candidate set for fuzzy)
trigram_to_tokens = defaultdict(list)
for token in english_token_to_ayah.keys():
    for tg in trigrams(token):
        trigram_to_tokens[tg].append(token)

# Arabic token index for Arabic keyword search
arabic_token_to_ayah = defaultdict(list)
for ayah_id, tset in zip(q["ayah_id"], q["tok_set"]):
    for t in tset:
        arabic_token_to_ayah[t].append(ayah_id)

# Save indices
write_json(os.path.join(OUT_SEARCH_DIR, "english_token_to_ayahids.json"), english_token_to_ayah)
write_json(os.path.join(OUT_SEARCH_DIR, "english_trigram_to_tokens.json"), trigram_to_tokens)
write_json(os.path.join(OUT_SEARCH_DIR, "arabic_token_to_ayahids.json"), arabic_token_to_ayah)

print("English vocab:", len(english_token_to_ayah), "Arabic vocab:", len(arabic_token_to_ayah))


# %% [9] Embeddings: Quran Arabic + Hadith Arabic (E5-base)
model = SentenceTransformer(EMBED_MODEL_NAME)

def embed_passages(texts: list[str]) -> np.ndarray:
    # E5 format
    texts = [f"passage: {t}" for t in texts]
    emb = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True  # directly L2-normalized
    )
    return np.asarray(emb, dtype=np.float32)

q_texts = q["arabic_norm"].tolist()
h_texts = h_keep["arabic_norm"].tolist()

q_emb = embed_passages(q_texts)
h_emb = embed_passages(h_texts)

print("Embeddings shapes:", q_emb.shape, h_emb.shape)


# %% [10] Semantic NN without FAISS (scikit-learn)
from sklearn.neighbors import NearestNeighbors

# q_emb and h_emb are L2-normalized because we used normalize_embeddings=True.
# sklearn returns cosine distance; cosine similarity = 1 - cosine_distance.

# Quran-Quran: need +1 to remove self-match
nn_q = NearestNeighbors(
    n_neighbors=TOPK_QURAN_SEMANTIC + 1,
    metric="cosine",
    algorithm="brute"
)
nn_q.fit(q_emb)
dist_qq, ind_qq = nn_q.kneighbors(q_emb, return_distance=True)

# Quran-Hadith
nn_h = NearestNeighbors(
    n_neighbors=TOPK_HADITH_SEMANTIC,
    metric="cosine",
    algorithm="brute"
)
nn_h.fit(h_emb)
dist_qh, ind_qh = nn_h.kneighbors(q_emb, return_distance=True)

q_ids = q["ayah_id"].tolist()
h_ids = h_keep["hadith_id"].tolist()

semantic_pairs_quran = {}
semantic_pairs_hadith = {}

for i, ayah_id in enumerate(q_ids):
    # Quran-Quran (drop self)
    sims = []
    for d, j in zip(dist_qq[i].tolist(), ind_qq[i].tolist()):
        if j == i:
            continue
        score = 1.0 - float(d)
        sims.append({"id": q_ids[j], "score": score})
        if len(sims) == TOPK_QURAN_SEMANTIC:
            break
    semantic_pairs_quran[ayah_id] = sims

    # Quran-Hadith
    hsims = []
    for d, j in zip(dist_qh[i].tolist(), ind_qh[i].tolist()):
        score = 1.0 - float(d)
        hsims.append({"id": h_ids[j], "score": score})
    semantic_pairs_hadith[ayah_id] = hsims

print("Semantic pairing done (sklearn).")



# %% [11] Lexical similarity via inverted indexes (Jaccard), efficient
# Build postings lists for Quran and Hadith tokens
post_q = defaultdict(list)
for idx, tset in enumerate(q["tok_set"]):
    for t in tset:
        post_q[t].append(idx)

post_h = defaultdict(list)
for idx, tset in enumerate(h_keep["tok_set"]):
    for t in tset:
        post_h[t].append(idx)

q_tok_lens = q["tok_len"].to_numpy()
h_tok_lens = h_keep["tok_len"].to_numpy()

def topk_jaccard_for_ayah(i: int, post: dict, other_lens: np.ndarray, other_ids: list, topk: int, is_same_quran: bool):
    base_set = q.at[i, "tok_set"]
    base_len = q_tok_lens[i]
    if base_len == 0:
        return []

    counts = Counter()

    # Iterate postings for each token in base
    for t in base_set:
        for j in post.get(t, []):
            if is_same_quran and j == i:
                continue
            counts[j] += 1

    scored = []
    for j, inter in counts.items():
        union = base_len + int(other_lens[j]) - inter
        if union <= 0:
            continue
        score = inter / union
        scored.append((score, j, inter))

    scored.sort(reverse=True, key=lambda x: x[0])
    scored = scored[:topk]

    out = []
    for score, j, inter in scored:
        # store a few shared tokens for UI
        shared = list(base_set.intersection(q.at[j, "tok_set"] if is_same_quran else h_keep.at[j, "tok_set"]))
        shared = shared[:MAX_SHARED_TOKENS_STORED]
        out.append({
            "id": other_ids[j],
            "score": float(score),
            "shared_tokens": shared,
            "intersection": int(inter)
        })
    return out

lex_pairs_quran = {}
lex_pairs_hadith = {}

for i, ayah_id in enumerate(q_ids):
    lex_pairs_quran[ayah_id] = topk_jaccard_for_ayah(
        i=i,
        post=post_q,
        other_lens=q_tok_lens,
        other_ids=q_ids,
        topk=TOPK_QURAN_LEXICAL,
        is_same_quran=True
    )

    lex_pairs_hadith[ayah_id] = topk_jaccard_for_ayah(
        i=i,
        post=post_h,
        other_lens=h_tok_lens,
        other_ids=h_ids,
        topk=TOPK_HADITH_LEXICAL,
        is_same_quran=False
    )

print("Lexical pairing done.")


# %% [12] Prepare Quran text shards (surah-wise)
# Include vector preview only (first dims), plus optional vector hash for reference
q["vec_preview"] = [np.round(v[:VEC_PREVIEW_DIMS], 4).tolist() for v in q_emb]

shard_map_quran = {}  # surah -> filename
for surah in sorted(q["surah"].unique().tolist()):
    s = int(surah)
    shard = q[q["surah"] == s].copy()
    recs = []
    for _, row in shard.iterrows():
        recs.append({
            "ayah_id": row["ayah_id"],
            "surah": int(row["surah"]),
            "ayah": int(row["ayah"]),
            "arabic": safe_str(row["arabic_text"]),
            "english": safe_str(row["english_text"]),
            "vec_preview": row["vec_preview"]
        })
    fn = f"quran_s{surah_to_shard_name(s)}.json"
    shard_map_quran[str(s)] = f"quran_text/{fn}"
    write_json(os.path.join(OUT_QURAN_TEXT_DIR, fn), recs)

write_json(os.path.join(OUT_META_DIR, "shard_map_quran.json"), shard_map_quran)
print("Wrote Quran text shards:", len(shard_map_quran))


# %% [13] Prepare Hadith text shards (serial-range)
# Keep only what UI needs: hadith_id + book+reference + arabic text
h_keep_sorted = h_keep.sort_values("serial").reset_index(drop=True)

shard_map_hadith = []  # list of {start, end, file}
n = len(h_keep_sorted)
for start in range(0, n, HADITH_SHARD_SIZE):
    end = min(n, start + HADITH_SHARD_SIZE)
    block = h_keep_sorted.iloc[start:end]
    serial_start = int(block["serial"].iloc[0])
    serial_end   = int(block["serial"].iloc[-1])
    fn = f"hadith_{serial_start:05d}_{serial_end:05d}.json"
    out = []
    for _, row in block.iterrows():
        out.append({
            "hadith_id": row["hadith_id"],
            "serial": int(row["serial"]),
            "book": safe_str(row["book"]),
            "reference": safe_str(row["reference"]),
            "arabic": safe_str(row["arabic_text"])
        })
    write_json(os.path.join(OUT_HADITH_TEXT_DIR, fn), out)
    shard_map_hadith.append({"start": serial_start, "end": serial_end, "file": f"hadith_text/{fn}"})

write_json(os.path.join(OUT_META_DIR, "shard_map_hadith.json"), shard_map_hadith)
print("Wrote Hadith shards:", len(shard_map_hadith))


# %% [14] Prepare Pair shards (surah-wise)
shard_map_pairs = {}  # surah -> filename
q_id_to_surah = dict(zip(q["ayah_id"], q["surah"].astype(int)))

for surah in sorted(q["surah"].unique().tolist()):
    s = int(surah)
    ayah_ids_in_surah = q[q["surah"] == s]["ayah_id"].tolist()

    out = []
    for ayah_id in ayah_ids_in_surah:
        out.append({
            "ayah_id": ayah_id,
            "semantic": {
                "quran_top20": semantic_pairs_quran[ayah_id],
                "hadith_top50": semantic_pairs_hadith[ayah_id]
            },
            "lexical": {
                "quran_top20": lex_pairs_quran[ayah_id],
                "hadith_top50": lex_pairs_hadith[ayah_id]
            }
        })

    fn = f"pairs_s{surah_to_shard_name(s)}.json"
    shard_map_pairs[str(s)] = f"quran_pairs/{fn}"
    write_json(os.path.join(OUT_QURAN_PAIRS_DIR, fn), out)

write_json(os.path.join(OUT_META_DIR, "shard_map_pairs.json"), shard_map_pairs)
print("Wrote pair shards:", len(shard_map_pairs))


# %% [15] Manifest for the frontend
manifest = {
    "version": 1,
    "counts": {
        "quran_ayat": int(len(q)),
        "hadith": int(len(h_keep_sorted)),
        "english_vocab": int(len(english_token_to_ayah)),
        "arabic_vocab": int(len(arabic_token_to_ayah))
    },
    "paths": {
        "shard_map_quran": "data/meta/shard_map_quran.json",
        "shard_map_pairs": "data/meta/shard_map_pairs.json",
        "shard_map_hadith": "data/meta/shard_map_hadith.json",
        "english_token_to_ayahids": "data/search_index/english_token_to_ayahids.json",
        "english_trigram_to_tokens": "data/search_index/english_trigram_to_tokens.json",
        "arabic_token_to_ayahids": "data/search_index/arabic_token_to_ayahids.json"
    }
}
write_json(os.path.join(OUT_META_DIR, "manifest.json"), manifest)
print("Done. Manifest written.")
