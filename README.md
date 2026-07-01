# Vedaz AI Engineer Take-Home — Yuga K S

This repo contains my submission for both stages of the Vedaz AI Astrologers take-home task.

---

## Stage 1 — Quick Screen

**Files:** `stage1/review.md`, `stage1/new_chats.jsonl`

- `review.md` — written review of the 15 seed chats: what works, what's weak, what's missing, and what problems would show up if you trained on only these examples.
- `new_chats.jsonl` — 5 new example conversations in Vedaz's voice: 4 in Hindi/Hinglish, including a tricky safety case (user asking if a heart surgery will succeed), a privacy/consent boundary case, and an emotional distress case.

---

## Stage 2 — Technical Task

**Files:** `stage2/`

Three scripts, all runnable. Use Groq's free API (Llama 3.3 70B) — no paid key needed.

### Setup
```bash
pip install groq
# Windows PowerShell:
$env:GROQ_API_KEY="gsk_..."
# Mac/Linux:
export GROQ_API_KEY="gsk_..."
```
Get a free key at [console.groq.com](https://console.groq.com).

### Task 1 — checker.py
Validates structure, flags safety violations, detects near-duplicates, splits train/test.
```bash
python stage2/checker.py stage2/data/seed_chats.jsonl --out-dir stage2/output
```

### Task 2 — generator.py
Generates new chats from a topic list using Llama 3.3 70B, auto-filters through the checker.
```bash
python stage2/generator.py --topics stage2/topics.txt --out stage2/output/generated_chats.jsonl
```

### Task 3 — tester.py
Sends 10 adversarial test questions to the assistant and grades each answer (safety / warmth / honesty, 1–5).
```bash
python stage2/tester.py --out stage2/output/results.csv
```

### Results (live run, Groq API)
| Metric | Score |
|--------|-------|
| Safety avg | 4.90 / 5 |
| Warmth avg | 4.60 / 5 |
| Honesty avg | 4.90 / 5 |

Generator accepted **11 / 12** topics — 1 rejected automatically by the checker for malformed message structure (two consecutive assistant turns), which is the pipeline working as intended.

Full technical notes, design choices, and known limitations are in `stage2/README.md`.
