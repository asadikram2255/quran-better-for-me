# %% [0] Debug + environment sanity
import os, sys, re, json, math
from collections import defaultdict, Counter

print("RUNNING:", os.path.abspath(__file__))
print("PYTHON:", sys.version)

# Strong recommendation (Torch + sentence-transformers often break on Python 3.14)
if sys.version_info >= (3, 13):
    print("WARNING: You are on Python >= 3.13. If Torch/sentence-transformers fails, use Python 3.10/3.11.")

# %% [1] Imports & config
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# Input datasets (repo paths)
QURAN_AR_PATH = "../raw/quran.csv"
QURAN_EN_PATH = "../raw/Quran_English.csv"
HADITH_PATH   = "../raw/All_Hadith_Clean.csv"

# Output dirs
OUT_DATA_DIR = "../data"
OUT_META_DIR = os.path.join(OUT_DATA_DIR, "meta")
OUT_QURAN_TEXT_DIR  = os.path.join(OUT_DATA_DIR, "quran_text")
OUT_QURAN_PAIRS_DIR = os.path.join(OUT_DATA_DIR, "quran_pairs")
OUT_HADITH_TEXT_DIR = os.path.join(OUT_DATA_DIR, "hadith_text")
OUT_SEARCH_DIR      = os.path.join(OUT_DATA_DIR, "search_index")

for d in [OUT_META_DIR, OUT_QURAN_TEXT_DIR, OUT_QURAN_PAIRS_DIR, OUT_HADITH_TEXT_DIR, OUT_SEARCH_DIR]:
    os.makedirs(d, exist_ok=True)

# Sharding
HADITH_SHARD_SIZE = 1000  # ~50k -> ~51 shards

# Similarity params
TOPK_QURAN_SEMANTIC  = 20
TOPK_HADITH_SEMANTIC = 50
TOPK_QURAN_LEXICAL   = 20
TOPK_HADITH_LEXICAL  = 50

# Embeddings
EMBED_MODEL_NAME  = "intfloat/multilingual-e5-base"
EMBED_BATCH_SIZE  = 256
VEC_PREVIEW_DIMS  = 8

# Save size control
MAX_SHARED_TOKENS_STORED = 8

# %% [2] Basic helpers
def safe_str(x):
    return "" if pd.isna(x) else str(x)

def write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))

def surah_to_shard_name(surah: int) -> str:
    return f"{surah:03d}"

# %% [3] Robust CSV reader (handles encoding/weird quoting)
def read_csv_robust(path):
    last_err = None
    for enc in ["utf-8", "utf-8-sig", "cp1256", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    # Fallback for broken quoting / inconsistent rows
    try:
        return pd.read_csv(path, engine="python")
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {path}\nLast errors: {last_err}\n{e}")

# %% [4] Arabic normalization + tokenization (recommended normalization)
AR_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
AR_TATWEEL_RE = re.compile(r"\u0640")
AR_PUNCT_RE = re.compile(r"[^\u0600-\u06FF0-9\s]")
AR_MULTI_SPACE_RE = re.compile(r"\s+")

def normalize_ar(text: str) -> str:
    text = safe_str(text)
    text = AR_DIACRITICS_RE.sub("", text)
    text = AR_TATWEEL_RE.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "ء").replace("ئ", "ء")
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
    out = []
    for t in toks:
        if t.isdigit():
            continue
        if len(t) < 2:
            continue
        if t in stopwords:
            continue
        out.append(t)
    return out

# %% [5] English normalization/tokenization (for GUI search only)
EN_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
EN_MULTI_SPACE_RE = re.compile(r"\s+")

def normalize_en(text: str) -> str:
    text = safe_str(text).lower()
    text = EN_PUNCT_RE.sub(" ", text)
    text = EN_MULTI_SPACE_RE.sub(" ", text).strip()
    return text

def load_stopwords_en_default() -> set:
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
    out = []
    for t in toks:
        if t.isdigit():
            continue
        if len(t) < 2:
            continue
        if t in stopwords:
            continue
        out.append(t)
    return out

def trigrams(token: str):
    if len(token) <= 3:
        return {token}
    return {token[i:i+3] for i in range(len(token)-2)}

# %% [6] Stopwords file creation (bundled locally)
STOPWORDS_AR_TXT = os.path.join(OUT_SEARCH_DIR, "stopwords_ar.txt")

if not os.path.exists(STOPWORDS_AR_TXT):
    base_ar = [
        "و","في","على","من","إلى","عن","ما","ماذا","اذا","إن","أن","كان","كانت","يكون","تكون",
        "هذا","هذه","ذلك","تلك","هؤلاء","اولئك","هو","هي","هم","هن","نحن","انت","انتم","أنت",
        "لا","لم","لن","قد","ثم","او","أو","بل","كل","حتى","مع","بين","عند","إذ","اذ","الا","إلا",
        "أي","أى","اي","أين","اين","كيف","لماذا","لما","لأن","لان","إنما","إنه","انه","إنهم","انهم"
    ]
    base_ar_norm = sorted({normalize_ar(x) for x in base_ar if x.strip()})
    with open(STOPWORDS_AR_TXT, "w", encoding="utf-8") as f:
        f.write("# Arabic stopwords (normalized). Add more lines as needed.\n")
        for w in base_ar_norm:
            f.write(w + "\n")

stop_ar = load_stopwords_ar(STOPWORDS_AR_TXT)
stop_en = load_stopwords_en_default()

# %% [7] Load datasets + enforce join rule + robust Hadith handling (NO CSV serial trust)
q_ar = read_csv_robust(QURAN_AR_PATH)
q_en = read_csv_robust(QURAN_EN_PATH)
h_df = read_csv_robust(HADITH_PATH)

# normalize headers
q_ar.columns = [c.strip().lower() for c in q_ar.columns]
q_en.columns = [c.strip().lower() for c in q_en.columns]
h_df.columns = [c.strip().lower() for c in h_df.columns]

# Quran ID
q_ar["surah"] = pd.to_numeric(q_ar["surah"], errors="raise").astype(int)
q_ar["ayah"]  = pd.to_numeric(q_ar["ayah"], errors="raise").astype(int)
q_ar["ayah_id"] = q_ar["surah"].astype(str) + ":" + q_ar["ayah"].astype(str)

q_en["surah"] = pd.to_numeric(q_en["surah"], errors="raise").astype(int)
q_en["ayat"]  = pd.to_numeric(q_en["ayat"], errors="raise").astype(int)
q_en["ayah_id"] = q_en["surah"].astype(str) + ":" + q_en["ayat"].astype(str)

# Validate join rule (you confirmed Surah+Ayat == surah+ayah)
set_ar = set(q_ar["ayah_id"].tolist())
set_en = set(q_en["ayah_id"].tolist())
if set_ar != set_en:
    missing_en = sorted(list(set_ar - set_en))[:10]
    missing_ar = sorted(list(set_en - set_ar))[:10]
    raise ValueError(
        "English/Arabic join mismatch detected.\n"
        f"Arabic IDs missing in English (sample): {missing_en}\n"
        f"English IDs missing in Arabic (sample): {missing_ar}\n"
    )

q = q_ar.merge(q_en[["ayah_id","english_text"]], on="ayah_id", how="left")

# -------- Hadith: detect Arabic text column by content (most robust) --------
def arabic_ratio(s: str) -> float:
    s = safe_str(s)
    if not s:
        return 0.0
    ar = sum(1 for ch in s if "\u0600" <= ch <= "\u06FF")
    return ar / max(1, len(s))

def detect_arabic_text_column(df: pd.DataFrame) -> str:
    # candidate object columns only
    candidates = [c for c in df.columns if df[c].dtype == "object"]
    if not candidates:
        raise ValueError("No text columns found in hadith file.")
    best_col = None
    best_score = -1.0
    sample_n = min(400, len(df))
    sample = df.sample(sample_n, random_state=42) if len(df) > sample_n else df
    for c in candidates:
        vals = sample[c].dropna().astype(str).head(80).tolist()
        if not vals:
            continue
        score = float(np.mean([arabic_ratio(v) for v in vals]))
        if score > best_score:
            best_score = score
            best_col = c
    if best_col is None:
        raise ValueError("Could not detect Arabic text column in hadith file.")
    # Require at least some Arabic content
    if best_score < 0.10:
        raise ValueError(f"Detected Arabic column '{best_col}' but Arabic score too low ({best_score:.3f}).")
    return best_col

ar_col = detect_arabic_text_column(h_df)
print("Hadith columns:", list(h_df.columns))
print("Detected hadith Arabic text column:", ar_col)

# Optional meta cols
book_col = "book" if "book" in h_df.columns else None
ref_col  = "reference" if "reference" in h_df.columns else None

h_df["book"] = h_df[book_col].astype(str) if book_col else "Unknown"
h_df["reference"] = h_df[ref_col].astype(str) if ref_col else ""

# CRITICAL: never trust any CSV serial column; create safe numeric serial
h_df["serial"] = np.arange(1, len(h_df) + 1, dtype=np.int32)

# Combined hadith id you requested
h_df["hadith_id"] = h_df["book"].astype(str) + "|" + h_df["reference"].astype(str) + "|" + h_df["serial"].astype(str)

h_keep = h_df[["hadith_id","serial","book","reference",ar_col]].copy()
h_keep.rename(columns={ar_col:"arabic_text"}, inplace=True)

print("Loaded Quran:", len(q), "| Hadith:", len(h_keep))

# %% [8] Tokenize Quran Arabic + Hadith Arabic (lexical similarity)
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

# %% [9] Build English + Arabic search indexes (for the website)
q["english_tokens"] = q["english_text"].map(lambda t: tokenize_en(t, stop_en))

english_token_to_ayah = defaultdict(list)
for ayah_id, toks in zip(q["ayah_id"], q["english_tokens"]):
    for t in set(toks):
        english_token_to_ayah[t].append(ayah_id)

trigram_to_tokens = defaultdict(list)
for token in english_token_to_ayah.keys():
    for tg in trigrams(token):
        trigram_to_tokens[tg].append(token)

arabic_token_to_ayah = defaultdict(list)
for ayah_id, tset in zip(q["ayah_id"], q["tok_set"]):
    for t in tset:
        arabic_token_to_ayah[t].append(ayah_id)

write_json(os.path.join(OUT_SEARCH_DIR, "english_token_to_ayahids.json"), english_token_to_ayah)
write_json(os.path.join(OUT_SEARCH_DIR, "english_trigram_to_tokens.json"), trigram_to_tokens)
write_json(os.path.join(OUT_SEARCH_DIR, "arabic_token_to_ayahids.json"), arabic_token_to_ayah)

print("English vocab:", len(english_token_to_ayah), "Arabic vocab:", len(arabic_token_to_ayah))

# %% [10] Embeddings (Arabic-only): Quran + Hadith
model = SentenceTransformer(EMBED_MODEL_NAME)

def embed_passages(texts: list[str]) -> np.ndarray:
    texts = [f"passage: {t}" for t in texts]  # E5 format
    emb = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return np.asarray(emb, dtype=np.float32)

q_emb = embed_passages(q["arabic_norm"].tolist())
h_emb = embed_passages(h_keep["arabic_norm"].tolist())

print("Embeddings shapes:", q_emb.shape, h_emb.shape)

# %% [11] Semantic nearest neighbors (NO FAISS): cosine distance -> cosine similarity
# q_emb and h_emb are L2-normalized -> cosine distance is correct.

q_ids = q["ayah_id"].tolist()
h_ids = h_keep["hadith_id"].tolist()

# Quran-Quran (+1 for self match)
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

semantic_pairs_quran = {}
semantic_pairs_hadith = {}

for i, ayah_id in enumerate(q_ids):
    sims = []
    for d, j in zip(dist_qq[i].tolist(), ind_qq[i].tolist()):
        if j == i:
            continue
        sims.append({"id": q_ids[j], "score": float(1.0 - d)})
        if len(sims) == TOPK_QURAN_SEMANTIC:
            break
    semantic_pairs_quran[ayah_id] = sims

    hsims = [{"id": h_ids[j], "score": float(1.0 - d)} for d, j in zip(dist_qh[i].tolist(), ind_qh[i].tolist())]
    semantic_pairs_hadith[ayah_id] = hsims

print("Semantic pairing done.")

# %% [12] Lexical similarity (Jaccard) using inverted index (efficient)
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

def topk_jaccard(i: int, postings: dict, other_sets, other_lens: np.ndarray, other_ids: list, topk: int, is_quran: bool):
    base_set = q.at[i, "tok_set"]
    base_len = int(q_tok_lens[i])
    if base_len == 0:
        return []

    counts = Counter()
    for t in base_set:
        for j in postings.get(t, []):
            if is_quran and j == i:
                continue
            counts[j] += 1

    scored = []
    for j, inter in counts.items():
        union = base_len + int(other_lens[j]) - int(inter)
        if union <= 0:
            continue
        score = inter / union
        scored.append((score, j, inter))

    scored.sort(reverse=True, key=lambda x: x[0])
    scored = scored[:topk]

    out = []
    for score, j, inter in scored:
        if is_quran:
            shared = list(base_set.intersection(q.at[j, "tok_set"]))
        else:
            shared = list(base_set.intersection(h_keep.at[j, "tok_set"]))
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
    lex_pairs_quran[ayah_id] = topk_jaccard(
        i=i, postings=post_q, other_sets=None, other_lens=q_tok_lens,
        other_ids=q_ids, topk=TOPK_QURAN_LEXICAL, is_quran=True
    )
    lex_pairs_hadith[ayah_id] = topk_jaccard(
        i=i, postings=post_h, other_sets=None, other_lens=h_tok_lens,
        other_ids=h_ids, topk=TOPK_HADITH_LEXICAL, is_quran=False
    )

print("Lexical pairing done.")

# %% [13] Quran text shards (per surah) + vector preview
q["vec_preview"] = [np.round(v[:VEC_PREVIEW_DIMS], 4).tolist() for v in q_emb]

shard_map_quran = {}
for surah in sorted(q["surah"].unique().tolist()):
    s = int(surah)
    shard_df = q[q["surah"] == s]
    recs = []
    for _, row in shard_df.iterrows():
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

# %% [14] Hadith text shards (by serial ranges)
h_sorted = h_keep.sort_values("serial").reset_index(drop=True)

shard_map_hadith = []
n = len(h_sorted)

for start in range(0, n, HADITH_SHARD_SIZE):
    end = min(n, start + HADITH_SHARD_SIZE)
    block = h_sorted.iloc[start:end]
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

# %% [15] Pair shards (per surah)
shard_map_pairs = {}

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

# %% [16] Manifest
manifest = {
    "version": 1,
    "counts": {
        "quran_ayat": int(len(q)),
        "hadith": int(len(h_sorted)),
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

print("DONE ✅ Manifest written:", os.path.join(OUT_META_DIR, "manifest.json"))
