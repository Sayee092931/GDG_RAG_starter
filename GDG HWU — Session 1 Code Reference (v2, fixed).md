# GDG HWU — Session 1 Code Reference (v2, post-fix)

**For facilitators.** Every code block below is generated from the same source files as `GDG_HWU_RAG_Starter.ipynb`, so the doc and the notebook cannot drift apart. These blocks were also run against a mock Gemini client (transient failures, rate-limit bursts, outages, mismatched counts) before release.

## What changed after the first run, and why

| Problem seen | Root cause | Fix |
| --- | --- | --- |
| Errors importing the demo file | Opening a notebook from GitHub in Colab brings **only the notebook** — `sample_dataset.txt` was never in the runtime | The demo text now lives **inside the notebook** (`DEMO_TEXT`). Wikipedia failures fall back to it automatically. New `"upload"` source type for students' own files |
| Retrieval returning wrong chunks | After 3 failed retries (only 15 s of waiting) the old `embed_texts` **silently skipped the batch** yet still printed "Embedded 264/264". Every later vector then belonged to the wrong chunk. Reproduced in testing: 264 chunks, 164 vectors, wrong results | Retries now back off 5→80 s and **stop with a clear error** instead of skipping. Counts are checked at every batch and in `retrieve()` |
| Chunks starting with "onstruction of the…" | Fixed-size slicing cut words in half | Chunker backs up to the nearest space; chunks always start and end on whole words |
| Lower-quality matches | Chunks and questions were embedded with no task type | Chunks use `RETRIEVAL_DOCUMENT`, questions use `RETRIEVAL_QUERY` |
| Checkpoint ignored after a restart | Python's `hash()` changes between sessions | Fingerprint uses `md5` |
| Possible Colab install conflicts | `numpy` was being re-installed over Colab's copy | `numpy` and the `wikipedia` package removed from the install; Wikipedia is read via its API with `requests` |
| Private Google Docs failing mysteriously | Google returns a sign-in **HTML page** (status 200) instead of an error | Detected, with a message telling students to share as "Anyone with the link can view" |

**Correction to the earlier guide:** `gemini-embedding-001` returns **3072** numbers per chunk by default, not 768. The matrix shape you'll see is `(number_of_chunks, 3072)`.

**Design decision:** retrieval uses plain NumPy, not Chroma/FAISS. At this scale it takes milliseconds, and the loop inside `retrieve()` is the lesson a vector database would hide. Show Chroma as a 5-minute aside at most.

---

## 0. Install

```python
!pip install -q -U google-genai pypdf requests
print("✅ Packages installed.")
```

## 1. API key + client (with a built-in key test)

```python
from getpass import getpass
from google import genai
from google.genai import types

API_KEY = getpass("Paste your Gemini API key here and press Enter: ")
client = genai.Client(api_key=API_KEY)

EMBED_MODEL = "gemini-embedding-001"

# Sanity check: a bad key or no network should fail HERE, not three cells later.
_probe = client.models.embed_content(model=EMBED_MODEL, contents="hello")
EMBED_DIM = len(_probe.embeddings[0].values)
print(f"✅ Client ready. Each chunk will become a list of {EMBED_DIM} numbers.")
```

**Explain:** `getpass` hides the key. The probe call at the bottom makes a bad key or a network problem fail *here*, with the embedding size printed as a bonus: that number (3072) is how many numbers represent each chunk's meaning.

## 2. Demo text + the universal loader

```python
# Built-in demo text, so nobody has to upload or download a file for Part 1.
# (This is also the automatic fallback if Wikipedia can't be reached on the day.)
DEMO_TEXT = """The Burj Khalifa is a skyscraper in Dubai, United Arab Emirates. At the time of its completion in 2010, it became the tallest structure and building in the world, a title it has held since, standing at a height of over 828 metres (2,716 feet) with more than 160 habitable floors. It was known as the Burj Dubai during construction before being renamed in honor of Khalifa bin Zayed Al Nahyan, the President of the United Arab Emirates at the time.

Construction of the Burj Khalifa began in 2004, with the exterior completed in 2009 and the building officially opened on 4 January 2010. The design is reportedly inspired by the Hymenocallis flower, a regional desert plant, and also draws on patterns found in Islamic architecture. The building's structure uses a design known as a "buttressed core," where a central hexagonal core is braced by three wings arranged around it, allowing the tower to taper as it rises and reducing wind forces on the upper floors.

The tower was designed by the Chicago-based architecture firm Skidmore, Owings and Merrill, with Adrian Smith serving as the lead architect and Bill Baker as the lead structural engineer. The main contractor for construction was Samsung C&T, working alongside Belgian group Besix and the UAE's Arabtec. Thousands of workers and hundreds of engineers from around the world were involved in the project at its peak.

The Burj Khalifa contains a mix of uses, including luxury residences, corporate offices, and the Armani Hotel Dubai, which occupies several of the lower floors. The building also features an outdoor observation deck called "At the Top," located on the 124th and 125th floors, as well as a higher observation deck, "At the Top SKY," on the 148th floor, which was for some time the highest outdoor observation deck in the world.

The tower includes double-deck elevators capable of traveling at high speeds, among the fastest elevators in the world at the time of construction. The building's exterior cladding consists of reflective glazing, aluminum, and textured stainless steel spandrel panels designed to withstand Dubai's extreme summer temperatures.

The Burj Khalifa is part of the larger Downtown Dubai development, a large-scale, mixed-use complex that also includes the Dubai Mall and the Dubai Fountain, a large choreographed fountain system on the manmade Burj Khalifa Lake directly in front of the tower. The fountain performs choreographed water, light, and sound shows set to music, drawing large crowds of residents and tourists in the evenings.

Since opening, the Burj Khalifa has become one of the most recognizable landmarks in the world and a symbol of Dubai's rapid modern development. It regularly hosts New Year's Eve firework and light shows that are broadcast internationally, and the building is lit with LED displays for national holidays and special occasions. It remains a major draw for tourism in Dubai and is frequently cited as an example of record-breaking modern engineering and architecture.

Several other buildings have since been proposed or constructed around the world with ambitions to surpass the Burj Khalifa's height, though as of the mid-2020s it continued to hold the record as the tallest building in the world. Its construction is often referenced in discussions of supertall skyscraper engineering, including innovations in concrete pumping, which set records for the height to which concrete was pumped during construction."""
```

```python
import re
import requests

# Loading and chunking text is FREE regardless of size -- no API call happens until
# the embedding step. These limits only control how many embedding *requests* we
# fire off live. 150,000 chars is roughly 330 chunks, i.e. ~4 API calls.
MIN_CHARS = 1500
MAX_CHARS = 150000
HEADERS = {"User-Agent": "GDG-HWU-RAG-Workshop/1.0 (student workshop, run in Colab)"}


def _extract_gdoc_id(url_or_id):
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url_or_id)
    return match.group(1) if match else url_or_id


def _load_wikipedia(title):
    resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "prop": "extracts", "explaintext": 1,
                "redirects": 1, "titles": title, "format": "json"},
        headers=HEADERS, timeout=20,
    )
    resp.raise_for_status()
    page = next(iter(resp.json()["query"]["pages"].values()))
    if "missing" in page or not page.get("extract"):
        raise ValueError(f"No Wikipedia article found for '{title}'. Check the exact title.")
    text = page["extract"]
    if "may refer to" in text[:300]:
        raise ValueError(f"'{title}' is a disambiguation page -- use a more specific title.")
    return text


def _load_gdoc(url_or_id):
    doc_id = _extract_gdoc_id(url_or_id)
    resp = requests.get(f"https://docs.google.com/document/d/{doc_id}/export?format=txt",
                        headers=HEADERS, timeout=20)
    resp.raise_for_status()
    # A private doc doesn't error -- Google quietly returns a sign-in HTML page instead.
    if "<html" in resp.text[:500].lower():
        raise ValueError("Couldn't read that Google Doc. Set sharing to "
                         "'Anyone with the link can view' and try again.")
    return resp.text


def _load_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    if not text.strip():
        raise ValueError("No text could be extracted. This is probably a scanned/image PDF, "
                         "which this workshop doesn't support -- use a text-based PDF.")
    return text


def _load_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_upload(_ignored=None):
    from google.colab import files  # only exists inside Colab
    print("Choose your .txt or .pdf file...")
    uploaded = files.upload()
    if not uploaded:
        raise ValueError("No file was uploaded.")
    name = next(iter(uploaded))
    return _load_pdf(name) if name.lower().endswith(".pdf") else _load_txt(name)


def load_text(source_type, source=None):
    """Universal loader: any source in, one clean string out, with a size check.

    source_type -> what `source` should be
      "wikipedia" -> exact article title
      "gdoc"      -> Google Doc link or ID (shared as 'Anyone with the link can view')
      "upload"    -> nothing (a file picker pops up; .txt or .pdf)
      "pdf"/"txt" -> a file path already in Colab's file panel
      "paste"     -> the raw text itself
      "demo"      -> nothing (the built-in demo text)
    """
    loaders = {
        "wikipedia": _load_wikipedia,
        "gdoc": _load_gdoc,
        "upload": _load_upload,
        "pdf": _load_pdf,
        "txt": _load_txt,
        "paste": lambda s: s,
        "demo": lambda _: DEMO_TEXT,
    }
    if source_type not in loaders:
        raise ValueError(f"Unknown source_type '{source_type}'. Choose one of: {list(loaders)}")

    text = loaders[source_type](source).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    char_count = len(text)
    print(f"Loaded {char_count:,} characters (about {char_count // 450} chunks).")

    if char_count < MIN_CHARS:
        print(f"⚠️  That's short (under {MIN_CHARS:,} chars). Search works better with "
              f"more material -- consider a longer article or section.")
    if char_count > MAX_CHARS:
        text = text[:MAX_CHARS]
        print(f"⚠️  Trimmed from {char_count:,} to {MAX_CHARS:,} characters to stay inside "
              f"free-tier limits. Tip: pick ONE focused article, not a whole wiki or book.")
    return text
```

**Explain:** `load_text()` is the only place that cares what the content is; everything after sees one string. No API calls happen here. Size is checked *before* anything touches Gemini, and warnings are printed, never swallowed.

```python
try:
    demo_text = load_text("wikipedia", "Burj Khalifa")
except Exception as e:
    print(f"⚠️ Couldn't fetch Wikipedia ({e}).")
    print("   Using the built-in demo text instead -- everything below works the same.")
    demo_text = load_text("demo")

print("\n", demo_text[:500], "...")
```

## 3. Chunking

```python
def chunk_text(text, chunk_size=500, overlap=50):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Don't cut a word in half: pull `end` back to the last space.
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break

        # Step forward, but back up by `overlap` so neighbouring chunks share a little text...
        start = max(end - overlap, start + 1)
        # ...and nudge to the next space so the new chunk starts on a whole word.
        next_space = text.find(" ", start, end)
        if next_space != -1:
            start = next_space + 1
    return chunks


demo_chunks = chunk_text(demo_text)
print(f"Split into {len(demo_chunks)} chunks.")
print("\n--- Example chunk ---\n")
print(demo_chunks[0])
```

**Explain:** two ideas. (1) Chunks are \~500 characters so retrieval is precise. (2) `start = max(end - overlap, ...)` steps back so neighbouring chunks share text — a fact on a boundary is whole in at least one chunk. The word-snapping lines just stop words being cut in half. No API call anywhere in this function.

## 4. Embedding

```python
import time
import pickle
import os
import hashlib
import numpy as np

MAX_BATCH = 100  # Gemini's per-request limit for embed_content


def _fingerprint(texts, task_type):
    # md5 (not Python's hash()) so the fingerprint survives a Colab runtime restart
    h = hashlib.md5(task_type.encode())
    for t in texts:
        h.update(t.encode("utf-8"))
    return h.hexdigest()


def _call_embed(client, batch, task_type):
    """One API call, with automatic retries. Gives up loudly -- never silently."""
    config = types.EmbedContentConfig(task_type=task_type)
    last_error = None
    for attempt in range(5):
        try:
            result = client.models.embed_content(model=EMBED_MODEL, contents=batch, config=config)
            return [e.values for e in result.embeddings]
        except Exception as e:
            last_error = e
            wait = 5 * (2 ** attempt)  # 5, 10, 20, 40, 80 seconds (rate limits need real waits)
            print(f"  ⚠️ API error: {e}\n     Retrying in {wait}s (attempt {attempt + 1}/5)...")
            time.sleep(wait)
    raise RuntimeError(f"Embedding failed after 5 attempts: {last_error}")


def embed_texts(client, texts, checkpoint_path=None, delay_seconds=1,
                task_type="RETRIEVAL_DOCUMENT"):
    """Turn a list of texts into a matrix: one row (vector) per text, same order.
    Progress is saved after every batch, so re-running resumes instead of restarting."""
    fingerprint = _fingerprint(texts, task_type)
    all_embeddings = []

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "rb") as f:
                saved = pickle.load(f)
            if saved.get("fingerprint") == fingerprint:
                all_embeddings = saved["embeddings"]
                print(f"Resuming from checkpoint: {len(all_embeddings)}/{len(texts)} already embedded.")
        except Exception:
            print("Ignoring unreadable checkpoint; starting fresh.")

    for i in range(len(all_embeddings), len(texts), MAX_BATCH):
        batch = texts[i:i + MAX_BATCH]
        vectors = _call_embed(client, batch, task_type)
        if len(vectors) != len(batch):
            # Misaligned embeddings would silently return the WRONG chunks -- so stop.
            raise RuntimeError(f"Sent {len(batch)} chunks but got {len(vectors)} embeddings back.")
        all_embeddings.extend(vectors)
        print(f"  Embedded {len(all_embeddings)}/{len(texts)} chunks...")

        if checkpoint_path:
            with open(checkpoint_path, "wb") as f:
                pickle.dump({"fingerprint": fingerprint, "embeddings": all_embeddings}, f)
        time.sleep(delay_seconds)

    matrix = np.array(all_embeddings)
    assert matrix.shape[0] == len(texts), "embedding count doesn't match chunk count"
    return matrix


demo_embeddings = embed_texts(client, demo_chunks, checkpoint_path="demo_checkpoint.pkl")
print(f"\n✅ Done. Embedding matrix shape: {demo_embeddings.shape}  (chunks x numbers-per-chunk)")
```

**Explain:** this is the **only** step that calls Gemini (besides the one query per search).

- `MAX_BATCH = 100`: a few hundred chunks = a handful of calls.
- `_call_embed`: retries with growing waits (rate limits need real waits), then raises. It never skips a batch.
- The length check: if the number of vectors ever differs from the number of chunks we stop, because misaligned vectors return the *wrong text*.
- Checkpoint: re-running the cell resumes. A finished run makes **zero** API calls.
- Output shape `(chunks, 3072)`: pause on it. That's the embedding matrix.

## 5. Retrieval

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def embed_query(client, query):
    # Questions use task_type RETRIEVAL_QUERY; chunks used RETRIEVAL_DOCUMENT.
    # The model is trained to match the two, which improves search quality.
    return np.array(_call_embed(client, [query], "RETRIEVAL_QUERY")[0])


def retrieve(client, query, chunks, chunk_embeddings, top_k=3):
    if len(chunks) != len(chunk_embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(chunk_embeddings)} embeddings -- "
                         f"re-run the chunking and embedding cells together.")
    query_embedding = embed_query(client, query)
    scores = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def show(results):
    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"#{rank}  (similarity: {score:.3f})")
        print(chunk)
        print("---")
```

```python
query = "Who designed the Burj Khalifa?"
show(retrieve(client, query, demo_chunks, demo_embeddings, top_k=3))
```

```python
# A question the article does NOT cover. Edit it and try your own!
query = "Who won the 2018 FIFA World Cup?"
show(retrieve(client, query, demo_chunks, demo_embeddings, top_k=3))
```

**Explain, slowly, this is the payoff:**

1. `cosine_similarity`: how closely two vectors point the same way. (`+ 1e-10` only avoids dividing by zero; skip it unless asked.)
2. The question is embedded with the same model, but as `RETRIEVAL_QUERY`.
3. `scores = [... for emb in chunk_embeddings]` is vector search: one score per chunk.
4. Sort high→low, keep the top `top_k`.
5. Out-of-scope question → low scores everywhere = the system saying "I don't know." A feature.

## 6. Part 2 — same functions, student's topic

```python
# ---- EDIT THESE TWO LINES ----
MY_SOURCE_TYPE = "wikipedia"   # "wikipedia", "gdoc", "upload", "paste", "txt", "pdf"
MY_SOURCE = "Your topic here"  # article title / gdoc link / (ignored for "upload") / text / path
# -------------------------------

my_text = load_text(MY_SOURCE_TYPE, MY_SOURCE)
my_chunks = chunk_text(my_text)
print(f"\n{len(my_chunks)} chunks ready to embed.")
```

```python
my_embeddings = embed_texts(client, my_chunks, checkpoint_path="my_checkpoint.pkl")
print(f"\n✅ Done. Embedding matrix shape: {my_embeddings.shape}")
```

```python
my_query = "Ask your bot a question about your own topic"
show(retrieve(client, my_query, my_chunks, my_embeddings, top_k=3))
```

**Say explicitly:** *"Nothing new just ran — the same functions, pointed at a different topic."*

---

## Function map

| Function | Calls Gemini? | What it does |
| --- | --- | --- |
| `load_text()` | No | Any of 7 source types in, one clean string out, size-checked |
| `chunk_text()` | No | Overlapping \~500-char chunks on whole-word boundaries |
| `embed_texts()` | **Yes** (≈1 call per 100 chunks) | Chunks → matrix of vectors, batched, retried, checkpointed |
| `cosine_similarity()` | No | How aligned two vectors are |
| `retrieve()` | **Yes** (1 call, for the question) | Embeds the question, scores every chunk, returns the top matches |

## Common errors on the day

| What students see | Meaning | Fix |
| --- | --- | --- |
| `API key not valid` at Step 1 | Typo or wrong key | Re-run the cell, paste again |
| `429` / `RESOURCE_EXHAUSTED` | Free-tier rate limit | Nothing — it retries itself. If it stops after 5 tries, wait a minute and re-run: the checkpoint resumes |
| `Couldn't read that Google Doc` | Doc isn't public | Share → "Anyone with the link can view" |
| `No text could be extracted` | Scanned PDF | Use a text-based PDF, or copy-paste into `"paste"` |
| `No Wikipedia article found` / `disambiguation` | Wrong title | Use the exact article title from the URL |
| `NameError: ... not defined` | A cell above was skipped | Runtime → Run all |
| `chunks but ... embeddings` | Chunking and embedding cells were run at different times | Re-run the chunk cell, then the embed cell |