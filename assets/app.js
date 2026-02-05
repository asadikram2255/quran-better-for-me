// Minimal, dependency-free JS

const state = {
  manifest: null,
  shardMapQuran: null,
  shardMapPairs: null,
  shardMapHadith: null,

  // Search indexes
  enTokenToAyah: null,
  enTriToTokens: null,
  arTokenToAyah: null,

  // Caches
  quranTextCache: new Map(),  // surah -> array of ayah records
  pairCache: new Map(),       // surah -> array of pair records
  hadithCache: new Map(),     // shardFile -> array of hadith records

  // quick lookup caches
  quranById: new Map(),       // ayah_id -> record
  pairsByAyah: new Map(),     // ayah_id -> pair record
  hadithById: new Map()       // hadith_id -> record (filled lazily)
};

const els = {
  badge: document.getElementById("statusBadge"),
  enQuery: document.getElementById("enQuery"),
  arQuery: document.getElementById("arQuery"),
  idQuery: document.getElementById("idQuery"),
  searchBtn: document.getElementById("searchBtn"),
  clearBtn: document.getElementById("clearBtn"),
  resultsList: document.getElementById("resultsList"),

  detailEmpty: document.getElementById("detailEmpty"),
  detailView: document.getElementById("detailView"),
  dArabic: document.getElementById("dArabic"),
  dEnglish: document.getElementById("dEnglish"),
  dAyahId: document.getElementById("dAyahId"),

  tabSemantic: document.getElementById("tabSemantic"),
  tabLexical: document.getElementById("tabLexical"),

  semQuran: document.getElementById("semQuran"),
  semHadith: document.getElementById("semHadith"),
  lexQuran: document.getElementById("lexQuran"),
  lexHadith: document.getElementById("lexHadith"),
};

function setBadge(kind, text){
  els.badge.className = `badge ${kind}`;
  els.badge.textContent = text;
}

async function fetchJson(path){
  const res = await fetch(path, {cache:"force-cache"});
  if(!res.ok) throw new Error(`Failed to fetch ${path}`);
  return await res.json();
}

// ---------- Normalization ----------
function normEn(s){
  return (s || "").toLowerCase().replace(/[^a-z0-9\s]/g," ").replace(/\s+/g," ").trim();
}
function tokenizeEn(s){
  const t = normEn(s).split(" ").filter(Boolean);
  return t.filter(x => x.length >= 2);
}
function normAr(s){
  // mirror python normalization roughly
  return (s || "")
    .replace(/[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]/g,"")
    .replace(/\u0640/g,"")
    .replace(/[أإآ]/g,"ا")
    .replace(/ى/g,"ي")
    .replace(/[ؤئ]/g,"ء")
    .replace(/[^\u0600-\u06FF0-9\s]/g," ")
    .replace(/\s+/g," ")
    .trim();
}
function tokenizeAr(s){
  const t = normAr(s).split(" ").filter(Boolean);
  return t.filter(x => x.length >= 2 && !/^\d+$/.test(x));
}

function trigrams(tok){
  if(tok.length <= 3) return new Set([tok]);
  const out = new Set();
  for(let i=0;i<tok.length-2;i++) out.add(tok.slice(i,i+3));
  return out;
}

// Levenshtein distance (small tokens only)
function levenshtein(a,b){
  if(a===b) return 0;
  const m=a.length, n=b.length;
  if(m===0) return n;
  if(n===0) return m;
  const dp = new Array(n+1);
  for(let j=0;j<=n;j++) dp[j]=j;
  for(let i=1;i<=m;i++){
    let prev=dp[0];
    dp[0]=i;
    for(let j=1;j<=n;j++){
      const tmp=dp[j];
      const cost=(a[i-1]===b[j-1])?0:1;
      dp[j]=Math.min(dp[j]+1, dp[j-1]+1, prev+cost);
      prev=tmp;
    }
  }
  return dp[n];
}

function maxAllowedEdits(len){
  if(len <= 4) return 1;
  if(len <= 7) return 2;
  return 2;
}

// ---------- Shard loading ----------
function surahFromAyahId(ayahId){
  const p = ayahId.split(":");
  return parseInt(p[0],10);
}

async function ensureSurahLoaded(surah){
  const key = String(surah);
  if(!state.quranTextCache.has(key)){
    const file = state.shardMapQuran[key];
    const data = await fetchJson("data/" + file);
    state.quranTextCache.set(key, data);
    for(const rec of data) state.quranById.set(rec.ayah_id, rec);
  }
  if(!state.pairCache.has(key)){
    const file = state.shardMapPairs[key];
    const data = await fetchJson("data/" + file);
    state.pairCache.set(key, data);
    for(const rec of data) state.pairsByAyah.set(rec.ayah_id, rec);
  }
}

// Hadith shard map is array [{start,end,file}]
async function ensureHadithById(hadithId){
  if(state.hadithById.has(hadithId)) return;
  // hadithId ends with |serial
  const parts = hadithId.split("|");
  const serial = parseInt(parts[parts.length-1],10);
  const shard = state.shardMapHadith.find(s => serial >= s.start && serial <= s.end);
  if(!shard) return;

  if(!state.hadithCache.has(shard.file)){
    const data = await fetchJson("data/" + shard.file);
    state.hadithCache.set(shard.file, data);
    for(const rec of data) state.hadithById.set(rec.hadith_id, rec);
  }
}

// ---------- Search ----------
async function searchByAyahId(ayahId){
  const surah = surahFromAyahId(ayahId);
  await ensureSurahLoaded(surah);
  const rec = state.quranById.get(ayahId);
  return rec ? [rec] : [];
}

function unique(arr){
  return Array.from(new Set(arr));
}

async function searchByArabicKeyword(q){
  const tok = tokenizeAr(q)[0];
  if(!tok) return [];
  const ids = state.arTokenToAyah[tok] || [];
  const surahs = unique(ids.map(surahFromAyahId));
  for(const s of surahs) await ensureSurahLoaded(s);
  return ids.map(id => state.quranById.get(id)).filter(Boolean);
}

async function searchByEnglishFuzzy(q){
  const toks = tokenizeEn(q);
  if(!toks.length) return [];

  const matchedAyahScores = new Map();
  for(const qt of toks){
    const grams = trigrams(qt);
    const candidates = new Set();
    for(const g of grams){
      const c = state.enTriToTokens[g] || [];
      for(const t of c) candidates.add(t);
    }

    const maxEd = maxAllowedEdits(qt.length);
    const good = [];
    for(const t of candidates){
      if(Math.abs(t.length - qt.length) > maxEd) continue;
      const d = levenshtein(qt, t);
      if(d <= maxEd) good.push({t, d});
    }

    good.sort((a,b)=>a.d-b.d);
    const best = good.slice(0, 10);

    for(const m of best){
      const ids = state.enTokenToAyah[m.t] || [];
      for(const id of ids){
        const prev = matchedAyahScores.get(id) || 0;
        matchedAyahScores.set(id, prev + (maxEd - m.d + 1));
      }
    }
  }

  const ranked = Array.from(matchedAyahScores.entries())
    .sort((a,b)=>b[1]-a[1])
    .slice(0, 200)
    .map(x => x[0]);

  const surahs = unique(ranked.map(surahFromAyahId));
  for(const s of surahs) await ensureSurahLoaded(s);

  return ranked.map(id => state.quranById.get(id)).filter(Boolean);
}

// ---------- Rendering ----------
function renderResults(list){
  els.resultsList.innerHTML = "";
  if(!list.length){
    els.resultsList.classList.add("empty");
    els.resultsList.textContent = "No results found.";
    return;
  }
  els.resultsList.classList.remove("empty");

  for(const rec of list.slice(0, 60)){
    const div = document.createElement("div");
    div.className = "item";
    div.innerHTML = `
      <div class="id">${rec.ayah_id}</div>
      <div>
        <div class="txt" dir="rtl">${rec.arabic}</div>
        <div class="subtxt">${rec.english}</div>
      </div>
    `;
    div.onclick = () => openDetail(rec.ayah_id);
    els.resultsList.appendChild(div);
  }
}

function setTab(name){
  const tabs = document.querySelectorAll(".tab");
  tabs.forEach(t => t.classList.toggle("active", t.dataset.tab===name));
  els.tabSemantic.classList.toggle("hidden", name!=="semantic");
  els.tabLexical.classList.toggle("hidden", name!=="lexical");
}

document.querySelectorAll(".tab").forEach(btn=>{
  btn.addEventListener("click", ()=>setTab(btn.dataset.tab));
});

function fmtScore(x){
  return (Math.round(x*10000)/10000).toFixed(4);
}

function renderPairList(container, items, kind){
  container.innerHTML = "";
  if(!items || !items.length){
    container.innerHTML = `<div class="empty">No items.</div>`;
    return;
  }

  for(const it of items){
    const div = document.createElement("div");
    div.className = "pair";

    let body = "";
    let extra = "";

    if(kind === "quran"){
      const rec = state.quranById.get(it.id);
      body = rec
        ? `<div dir="rtl">${rec.arabic}</div><div class="small">${rec.english}</div>`
        : `<div class="small">Loading…</div>`;
    } else {
      const h = state.hadithById.get(it.id);
      if(h){
        const ar = h.arabic || "";
        const en = h.english || ""; // <-- will show if present in hadith shards
        body = `<div dir="rtl">${ar}</div>`;
        if(en) extra = `<div class="small">${en}</div>`;
        else extra = `<div class="small">${h.book || ""} — ${h.reference || ""}</div>`;
        body += extra;
      } else {
        body = `<div class="small">Loading…</div>`;
      }
    }

    let shared = "";
    if(it.shared_tokens && it.shared_tokens.length){
      shared = `<div class="small">shared tokens: <span dir="rtl">${it.shared_tokens.join(" · ")}</span></div>`;
    }

    div.innerHTML = `
      <div class="pairTop">
        <div class="pairId">${it.id}</div>
        <div class="pairScore">score: ${fmtScore(it.score)}</div>
      </div>
      <div class="pairBody">${body}</div>
      ${shared}
    `;

    container.appendChild(div);
  }
}

async function openDetail(ayahId){
  const surah = surahFromAyahId(ayahId);
  await ensureSurahLoaded(surah);

  const rec = state.quranById.get(ayahId);
  const pairs = state.pairsByAyah.get(ayahId);
  if(!rec || !pairs) return;

  els.detailEmpty.classList.add("hidden");
  els.detailView.classList.remove("hidden");

  els.dArabic.textContent = rec.arabic;
  els.dEnglish.textContent = rec.english;
  els.dAyahId.textContent = rec.ayah_id;

  const semQ = pairs.semantic.quran_top20 || [];
  const lexQ = pairs.lexical.quran_top20 || [];

  const neededSurahs = new Set();
  for(const it of semQ) neededSurahs.add(surahFromAyahId(it.id));
  for(const it of lexQ) neededSurahs.add(surahFromAyahId(it.id));
  for(const s of neededSurahs) await ensureSurahLoaded(s);

  const semH = pairs.semantic.hadith_top50 || [];
  const lexH = pairs.lexical.hadith_top50 || [];

  // Quick-load a few hadith first
  const toLoad = [...new Set([...semH.slice(0,12), ...lexH.slice(0,12)].map(x=>x.id))];
  for(const hid of toLoad) await ensureHadithById(hid);

  renderPairList(els.semQuran, semQ, "quran");
  renderPairList(els.lexQuran, lexQ, "quran");
  renderPairList(els.semHadith, semH, "hadith");
  renderPairList(els.lexHadith, lexH, "hadith");

  // Lazy load rest
  setTimeout(async ()=>{
    const allH = [...new Set([...semH, ...lexH].map(x=>x.id))];
    for(let i=0;i<allH.length;i++){
      await ensureHadithById(allH[i]);
      if(i % 25 === 0){
        renderPairList(els.semHadith, semH, "hadith");
        renderPairList(els.lexHadith, lexH, "hadith");
        await new Promise(r=>setTimeout(r, 10));
      }
    }
    renderPairList(els.semHadith, semH, "hadith");
    renderPairList(els.lexHadith, lexH, "hadith");
  }, 40);

  setTab("semantic");
}

// ---------- Main ----------
async function init(){
  try{
    const manifest = await fetchJson("data/meta/manifest.json");
    state.manifest = manifest;

    state.shardMapQuran = await fetchJson(manifest.paths.shard_map_quran);
    state.shardMapPairs = await fetchJson(manifest.paths.shard_map_pairs);
    state.shardMapHadith = await fetchJson(manifest.paths.shard_map_hadith);

    state.enTokenToAyah = await fetchJson(manifest.paths.english_token_to_ayahids);
    state.enTriToTokens = await fetchJson(manifest.paths.english_trigram_to_tokens);
    state.arTokenToAyah = await fetchJson(manifest.paths.arabic_token_to_ayahids);

    setBadge("ok", `Ready — Quran: ${manifest.counts.quran_ayat} | Hadith: ${manifest.counts.hadith}`);
  } catch(err){
    console.error(err);
    setBadge("err", "Failed to load required JSON files");
  }
}

els.searchBtn.onclick = async () => {
  const en = els.enQuery.value.trim();
  const ar = els.arQuery.value.trim();
  const id = els.idQuery.value.trim();

  els.detailView.classList.add("hidden");
  els.detailEmpty.classList.remove("hidden");

  let results = [];
  try{
    setBadge("warn", "Searching…");
    if(id){
      results = await searchByAyahId(id);
    } else if(ar){
      results = await searchByArabicKeyword(ar);
    } else if(en){
      results = await searchByEnglishFuzzy(en);
    } else {
      setBadge("warn", "Enter a query first");
      renderResults([]);
      return;
    }
    renderResults(results);
    setBadge("ok", `Found ${results.length} ayat`);
  } catch(err){
    console.error(err);
    setBadge("err", "Search failed");
  }
};

els.clearBtn.onclick = () => {
  els.enQuery.value = "";
  els.arQuery.value = "";
  els.idQuery.value = "";
  renderResults([]);
  els.detailView.classList.add("hidden");
  els.detailEmpty.classList.remove("hidden");
  setBadge("ok", "Ready");
};

init();
