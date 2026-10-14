# GDG HWU RAG Workshop Starter

Starter repo for the GDG on Campus Heriot-Watt University Dubai RAG workshop series.

## How to use this (attendees)

1. Open `GDG_HWU_RAG_Starter.ipynb` in Google Colab: go to [colab.research.google.com](https://colab.research.google.com) → **File → Open notebook → GitHub**, paste this repo's URL, pick the notebook.
2. Have your free Gemini API key ready (https://aistudio.google.com/apikey).
3. Run the cells top to bottom.

**Everything the notebook needs is inside the notebook.** Colab only copies the `.ipynb` when you open it from GitHub, not the other files in this repo, so the demo text is embedded in the notebook itself. Nothing needs downloading or importing separately. For your own files, use `MY_SOURCE_TYPE = "upload"` in Part 2 and a file picker appears.

## What's here

| File | Purpose |
|---|---|
| `GDG_HWU_RAG_Starter.ipynb` | The Session 1 notebook: setup, shared demo build (Part 1), bring-your-own-topic build (Part 2) |
| `GDG_HWU_RAG_Session2.ipynb` | The Session 2 notebook: generation, grounding, evaluation, tuning, and a lab on your own data. Students **run and tweak**; they don't write code |
| `sample_dataset.txt` | Copy of the demo text. **Not required** by the notebook; handy for practising the `"txt"` source type |

## What Session 1 teaches

- **Universal loading** — `load_text()` takes a Wikipedia title, Google Doc link, uploaded `.txt`/`.pdf`, pasted text, or the built-in demo, always returns one clean string, and size-checks it
- **Chunking** — overlapping ~500-character chunks that never cut a word in half
- **Embedding** — `gemini-embedding-001` (3072 numbers per chunk), batched at 100 per request, with retries and a resumable checkpoint
- **Retrieval** — cosine similarity in plain NumPy, the same mechanism a vector database uses

Session 1 stops at retrieval. Generation is Session 2.

## What Session 2 teaches

Prompt building and generation · grounding (strict instructions, a score cut-off) · grading a RAG system with a test set and **hit@k** · changing one dial at a time (a fair chunk-size experiment) · an optional hybrid-search upgrade. Session 2 starts with a one-cell recap of Session 1, so anyone who missed it can join. The model name is picked automatically from a short list because Google retires models; **after your dry run, pin the working name in `GEN_MODEL_OVERRIDE`**.

## Organizer notes

**Size limits** (`MIN_CHARS`, `MAX_CHARS` in the loader): loading and chunking are free; only embedding calls the API. 150,000 characters ≈ 330 chunks ≈ 4 API calls. 1 MB ≈ 2,200 chunks ≈ 23 calls.

**Reliability behaviour** (tested offline against a mock client, not yet against the live API):
- API errors retry with waits of 5, 10, 20, 40, 80 seconds, then stop with a clear error. A batch is never silently skipped.
- Chunk/embedding counts are verified, so results can never be misaligned.
- Progress is checkpointed; re-running a cell resumes, and a finished run makes zero API calls.
- Wikipedia failure falls back to the built-in demo text automatically.

**Before each session:** do a dry run in a *fresh* Colab with a *fresh* free-tier key on the venue wifi. Published rate limits are not reliable enough to plan from.

## Fixed after the first run
Demo-file import errors (file wasn't in the Colab runtime), silent embedding misalignment under rate limits (wrong chunks returned), words cut in half at chunk edges, no embedding task types, checkpoint lost on restart, and private Google Docs failing silently. Details are in the facilitator Code Reference.
