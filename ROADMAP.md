# JedAI Roadmap — Raw Data → Public Agent

A staged plan to take JedAI from raw text all the way to a publicly usable,
self-improving chatbot/agent. It's grounded in what already exists in this
repo (model, tokenizer, trainer, `finetune_chat.py`, `inference/api.py`,
`chat.py`) and your versioning plan: **V1 ~50M → later 300M → 3B**, free Colab
first, then rented GPUs.

Legend: ✅ done · 🔶 partial / needs work · ⬜ not built yet

---

## The big picture

```
RAW  →  CLEAN  →  TOKENIZE  →  PRETRAIN (base)  →  FINE-TUNE (chat/SFT)
                                                        │
                                          ALIGN (DPO, optional) 
                                                        │
                                   EVAL & RED-TEAM  →  PACKAGE (model card)
                                                        │
                                    SERVE (API + UI)  →  PUBLIC RELEASE
                                                        │
                          MONITOR → COLLECT FEEDBACK → RETRAIN  (the loop)
                                                        │
                              AGENT LAYER (tools, memory, RAG, planning)
```

Each "major version" = a fresh, bigger pretrain on the same cleaned pipeline.
Each "minor version" = more fine-tuning / alignment on the same base.

---

## Phase 0 — Foundations  ✅ (mostly done)

| Piece | Status | File |
|---|---|---|
| Model (RoPE, RMSNorm, SwiGLU, GQA) | ✅ | `model/` |
| Tokenizer (BPE, vocab 16k) | ✅ | `tokenizer/`, `tokenizer/train.py` |
| Trainer + checkpointing | ✅ | `training/trainer.py`, `training/checkpoint.py` |
| Configs 14M → 7B | ✅ | `configs/` |
| Inference + API + CLI chat | ✅ | `inference/`, `chat.py` |
| Chat fine-tune script | 🔶 | `training/finetune_chat.py` (continued-LM style, not masked SFT) |
| Free Colab notebook | 🔶 | `notebooks/train_free.ipynb` (built, **not pushed** to origin) |

**Action items to close Phase 0**
- ⬜ Push the notebook + this roadmap to `origin/main` (you must authorize the push).
- ⬜ Pin exact dependency versions in `requirements.txt` so Colab is reproducible.
- ⬜ Add a `make`-style or `scripts/` one-liner per stage so the pipeline is repeatable.

---

## Phase 1 — Raw data acquisition

**Goal:** a documented, license-clean corpus you can regrow on demand.

- ✅ `data/download_dataset.py` — wikitext / openwebtext via HF `datasets`.
- ✅ `data/download_dialogue.py` — conversational data (daily_dialog etc.).
- ⬜ **Data manifest:** for every source record name, license, URL, token count,
  date pulled in `data/metadata/sources.jsonl`. This is what lets you publish
  legally and reproduce a version later.
- ⬜ **Scale targets** (rule of thumb ≈ 20 tokens/param, Chinchilla-style):
  - 50M model → ~1B tokens (you have ~124M tokenized now — **under-fed**; fine for a V1 learning run, plan more for a "real" 50M).
  - 300M → ~6B tokens.
  - 3B → ~60B tokens (this is where free/cheap data sourcing gets hard).
- ⬜ Add larger mixes when you scale: FineWeb / FineWeb-Edu, C4, The Stack (code),
  Wikipedia dumps, books in public domain. Mix ratios matter — keep a config.

**Gate to pass:** you can run one command and reproduce the raw corpus + manifest.

---

## Phase 2 — Cleaning & deduplication

**Goal:** turn raw dumps into high-quality training text. Garbage in = garbage model; cleaning is the highest-leverage cheap win.

- ✅ `data/clean.py` — current cleaning (JSONL + text normalization).
- ⬜ Harden the cleaning pipeline to cover, in order:
  1. **Decode/normalize** — fix encoding, Unicode NFC, strip control chars.
  2. **Boilerplate removal** — nav bars, headers/footers, HTML remnants.
  3. **Quality filters** — language ID (keep target langs), min/max length,
     symbol-to-word ratio, repetition ratio, perplexity filter (optional).
  4. **Dedup** — exact (hash) then near-dup (MinHash/LSH). Dedup is one of the
     biggest quality multipliers at scale.
  5. **PII / safety scrub** — emails, phone numbers, secrets; basic toxicity filter.
  6. **Decontamination** — remove any text overlapping your eval/benchmark sets.
- ⬜ Emit cleaning stats per source (kept %, dropped reasons) into `data/metadata/`.

**Gate to pass:** spot-check 100 random cleaned samples — they read like clean prose/dialogue, no boilerplate, no dupes.

---

## Phase 3 — Tokenization

**Goal:** stable token stream + frozen tokenizer per major version.

- ✅ Tokenizer trained (vocab 16k), `data/tokenize_dataset.py` → `train.bin`/`val.bin`,
  trainer auto-syncs `vocab_size` from `data/tokenized/metadata.json`.
- ⬜ **Retrain a bigger tokenizer for bigger models** (e.g. 32k for 300M, 32–48k for 3B).
  Tokenizer is frozen *per major version* — changing it = new major version.
- ⬜ Ensure special/chat tokens (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|eot|>`)
  are reserved in vocab now, so the chat format is consistent from pretrain on.
- ⬜ Store `tokenizer.json` alongside each released checkpoint (they must match).

**Gate to pass:** round-trip encode/decode is lossless on held-out text; chat tokens present.

---

## Phase 4 — Pretraining (the base model)

**Goal:** a base "text continuer" that has learned language. This is the expensive phase.

**V1 — 50M, free Colab T4**
- ✅ `configs/free.yaml` (~14M) and `configs/50m_t4.yaml` / `50m_gtx1650.yaml`.
- Run: `python -m training.trainer --config configs/50m_t4.yaml`
- ⬜ Verify train/val loss curves decrease and don't diverge; save curves to disk.
- ⬜ Use checkpoint resume + Google Drive sync so Colab disconnects don't lose progress.

**What to instrument before long runs**
- ⬜ Logging to **Weights & Biases or TensorBoard** (loss, lr, grad-norm, tokens/sec).
- ⬜ Periodic **sample generations** during training (catch garbage early).
- ⬜ Gradient clipping + LR warmup/decay sanity (already in `scheduler.py`/`optimizer.py`).
- ⬜ Checkpoint every N steps to durable storage (Drive/HF Hub), not just local.

**Scaling up (later majors)** — when free Colab is too small:
- Rent GPUs (your plan): A5000 ~$269/mo, or hourly 4090/5090/A100.
- 300M needs gradient accumulation + possibly the `training/distributed.py` path
  (DDP) once you're on multi-GPU. Validate DDP on 2 GPUs before scaling.
- Track **cost per run** and tokens/sec so you can budget the next major version.

**Gate to pass:** base model produces fluent (if not factual) continuations; val perplexity plateaus.

---

## Phase 5 — Fine-tuning into a chatbot (SFT)

**Goal:** turn the base text-continuer into something that *answers*.

- 🔶 `training/finetune_chat.py` exists but trains as plain continued-LM over the
  whole conversation. Upgrade to **proper SFT**:
  - ⬜ **Loss masking:** compute loss only on assistant tokens, not the user/system
    prompt. This is the single most important fix — it teaches *responding*, not
    *parroting the whole transcript*.
  - ⬜ **Consistent chat template** matching `inference/api.py`'s
    `<|system|>/<|user|>/<|assistant|>` format and an explicit end token.
  - ⬜ **Packing** multiple short conversations per sequence with attention/loss
    boundaries to use context efficiently.
- ⬜ Curate an SFT dataset: blend daily_dialog with higher-quality instruction
  data (e.g. open instruction sets) filtered for your "old-school, human-like" voice.
- Run: `python -m training.finetune_chat --base_checkpoint <base.pt> --data <sft.txt>`
- Test: `python chat.py --checkpoint checkpoints/chat/ckpt_chat.pt --chat`

**Gate to pass:** in `chat.py`, the model stays in character, answers the question, and stops cleanly at the end token.

---

## Phase 6 — Alignment & safety (optional but recommended)

**Goal:** make replies more helpful/consistent and less unsafe.

- ⬜ **Preference tuning (DPO)** — lighter-weight than RLHF, no reward model needed.
  Collect or use preference pairs (chosen/rejected) and run DPO on the SFT model.
- ⬜ **Safety SFT** — add refusals for clearly harmful requests so a public bot
  doesn't embarrass you.
- ⬜ **System-prompt hardening** — define JedAI's persona/limits in one place.

**Gate to pass:** model refuses obvious abuse, doesn't loop, doesn't leak the prompt.

---

## Phase 7 — Evaluation & red-teaming

**Goal:** know whether a new version is actually better before shipping.

- 🔶 `evaluation/` has perplexity / accuracy / benchmark stubs — wire them into a
  single `python -m evaluation.benchmark --checkpoint ...` report.
- ⬜ **Automatic metrics:** held-out perplexity, plus a few small task accuracies.
- ⬜ **Vibe/eval set:** a fixed list of ~50 prompts you re-run every version and
  read by hand (a personal "eval harness"). Cheap, catches regressions metrics miss.
- ⬜ **LLM-as-judge** (use a strong Claude model via the API) to score helpfulness/
  coherence of JedAI outputs across versions — automatable A/B comparison.
- ⬜ **Decontamination check** — confirm eval prompts weren't in training data.
- ⬜ Log every version's scores to a `CHANGELOG`/`evals/` table so you can see progress.

**Gate to pass:** new version ≥ old version on your fixed eval set, no safety regressions.

---

## Phase 8 — Packaging & documentation

**Goal:** a release artifact someone (including future-you) can actually use.

- ⬜ **Model card** (`MODEL_CARD.md`): architecture, params, data sources + licenses,
  training compute, intended use, limitations, eval results.
- ⬜ **Technical report** (short): what you did, what worked, what didn't — this is
  the "something to be proud of" learning artifact.
- ⬜ **Versioned checkpoints:** publish to **Hugging Face Hub** (`jadetindoy/jed-ai-50m`)
  with tokenizer + config bundled. Tag versions (`v1.0.0`, `v1.1.0`…).
- ⬜ **License** the weights and state the data provenance clearly.

**Gate to pass:** a stranger can `from_pretrained`-style load your model from the card alone.

---

## Phase 9 — Serving & public availability

**Goal:** anyone with a link can chat with JedAI.

- ✅ `inference/api.py` — FastAPI `/complete`, `/chat`, `/health`.
- ⬜ **Streaming** responses (SSE/WebSocket) — token-by-token feels far more "alive".
- ⬜ **Frontend:** a minimal web chat UI (or embed into your app). Hugging Face
  **Spaces (Gradio)** is the fastest free public demo path.
- ⬜ **Containerize** (`Dockerfile`) the API for reproducible deploys.
- ⬜ **Host:**
  - Free/cheap demo: HF Spaces.
  - Always-on small: a cheap VPS or serverless GPU (Modal, RunPod, Replicate).
- ⬜ **Production hygiene:** rate limiting, API keys, request logging, CORS lockdown
  (currently `allow_origins=["*"]`), input length caps, an abuse/safety filter on
  inputs and outputs.
- ⬜ **Health/uptime monitoring** + autoscale-to-zero to control GPU cost.

**Gate to pass:** public URL, someone outside your machine holds a coherent multi-turn chat.

---

## Phase 10 — The improvement loop (continuous updates)

**Goal:** JedAI gets better over time instead of frozen at launch. This is what
makes it feel like a living product, not a one-off.

```
users chat → log conversations (with consent) → review/curate good+bad turns
   → add to SFT/DPO data → retrain minor version → eval gate → ship → repeat
```

- ⬜ **Feedback capture:** thumbs up/down + optional correction in the UI.
- ⬜ **Data flywheel:** curate logged conversations into new fine-tune data
  (privacy-scrubbed, deduped, decontaminated — Phase 2 applies here too).
- ⬜ **Retraining cadence:** minor versions = periodic fine-tune on fresh feedback;
  major versions = a fresh bigger pretrain when you have the data + budget.
- ⬜ **Regression guard:** every retrain must pass the Phase 7 eval gate before deploy.
- ⬜ **Canary deploys / rollback:** keep the last-known-good checkpoint to revert to.
- ⬜ **Scheduled jobs** for the recurring parts (data refresh, eval runs).

**Gate to pass:** you can ship a v1.x from logged feedback without manual one-off steps.

---

## Phase 11 — Becoming a *real agent*

**Goal:** go beyond chat — JedAI that can *use tools, remember, and act*. A small
50M model is weak at this, so be strategic: the orchestration layer can carry a lot
of the load while the model grows.

**Capabilities to add (roughly in order of value)**
- ⬜ **Memory:** short-term (conversation buffer, already implicit) + long-term
  (vector store of past chats / user facts, retrieved per turn). Mirrors how this
  very assistant uses a memory dir.
- ⬜ **Retrieval (RAG):** let JedAI answer from a knowledge base / your app's docs
  instead of only its weights. Huge quality boost for a small model — it offloads
  facts to retrieval so the model only has to be fluent.
- ⬜ **Tool / function calling:** define a small tool schema (search, calculator,
  app actions). Train/fine-tune on tool-call traces so the model emits structured
  calls; the orchestrator executes and feeds results back.
- ⬜ **Planning loop:** an agent runtime that does observe → think → act → repeat
  with a step budget and stop conditions.
- ⬜ **Guardrails:** validate tool inputs, sandbox actions, cap loops, log every
  action for audit. Especially important once the agent can *do* things.

**Pragmatic note:** for early agent versions, consider a **hybrid** — JedAI handles
persona/chat, and a stronger model (e.g. Claude via API) handles hard
tool-use/planning — then progressively shift load to your own model as it scales to
300M/3B. This lets the agent be genuinely useful *now* while you grow the core model.

**Gate to pass:** JedAI completes a multi-step task using at least one real tool + memory, reliably and safely.

---

## Suggested order of attack (next 5 concrete steps)

1. **Authorize the push** of the notebook + this roadmap to `origin/main`.
2. **Fix SFT loss masking** in `finetune_chat.py` (biggest quality-per-effort win).
3. **Stand up logging** (W&B/TensorBoard) + a fixed **eval prompt set** before the next long train.
4. **Grow the corpus** toward ~1B tokens for a respectable 50M V1 (Phase 1–2).
5. **Ship a public demo** on HF Spaces with the chat checkpoint (Phase 8–9).

---

*Each phase has a "gate to pass" — don't move on until it's green. Versioning:
major = new pretrain, minor = new fine-tune/alignment, patch = serving/bugfix.*
```
