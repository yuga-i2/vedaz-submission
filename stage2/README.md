# Vedaz Stage 2 — Hands-On Technical Task

Three scripts: `checker.py` (Task 1), `generator.py` (Task 2), `tester.py` (Task 3).
All three are real Python, runnable, no hard-coded keys (everything reads
`ANTHROPIC_API_KEY` from the environment). I used the **Anthropic API**
(model `claude-sonnet-4-6`) rather than Together/DeepSeek, purely because it's
what I had convenient access to while building this — swapping providers is a
one-function change (`get_client()` / `call_llm()`), nothing else depends on
the specific API.

**Important honesty note:** Scripts were run live using Groq's free API (Llama 3.3 70B, model: llama-3.3-70b-versatile). The generator accepted 11/12 topics — 1 was automatically rejected by the checker for a malformed message structure (two consecutive assistant turns), which is the pipeline working as intended. The tester graded 10 adversarial questions and returned: safety avg 4.90/5, warmth 4.60/5, honesty 4.90/5. Q8 (job/startup decision) scored 4 on safety rather than 5 — the grader flagged it as slightly under-redirecting to a financial professional, which is a legitimate and honest finding.

## How to run

```bash
pip install anthropic
$env:GROQ_API_KEY="gsk_..."

# Task 1: check + split the seed data
python checker.py data/seed_chats.jsonl --out-dir output

# Task 2: generate new chats from a topic list, auto-filtered by the checker
python generator.py --topics topics.txt --n-per-topic 1 --out output/generated_chats.jsonl

# Task 3: ask the assistant a battery of test questions and grade the answers
python tester.py --out output/results.csv
# or grade against the test split checker.py produced:
python tester.py --questions output/test.jsonl --from-chats --out output/results.csv
```

## Task 1 — checker.py

**Approach:** layered, not single-method.
- Structure: exact check that messages start with `system`, then strictly
  alternate `user`/`assistant`, ending on `assistant`.
- Duplicates: `difflib.SequenceMatcher` similarity over normalized
  (lowercased, punctuation-stripped) full-chat text, threshold 0.85. This
  is O(n²) — fine for hundreds of chats; at real scale I'd switch to
  MinHash/LSH or embedding cosine similarity for speed.
- Safety: a curated set of **phrase-level regex patterns** (not single
  keywords) split into three rule families — death/illness prediction,
  guaranteed outcomes, fear-based remedy pressure — applied only to
  assistant turns. There's also an optional `--use-llm` mode that sends the
  full chat to Claude with a strict rubric and merges its verdict in, for
  catching paraphrased violations the regex can't.

**Why regex-first, not LLM-first:** keyword/regex checks are free, instant,
deterministic, and auditable — you can show a human exactly which phrase
fired. An LLM judge is more flexible but costs money/latency per chat and
its judgment isn't fully reproducible. My intent was: regex for the cheap,
high-confidence majority pass, LLM as a second opinion for ambiguous cases.

**Honest limitation, with evidence:** the regex rules don't model negation
or quoting. Running the checker on the original 15 seed chats actually
flags one (`#6`, the business-loan chat) as `guaranteed_outcome`, because it
contains the sentence *"...no honest astrologer would say 'crorepati ban
jayenge'"* — the assistant is correctly **warning against** that exact
phrase, but the pattern matches on the phrase alone. The same thing happens
to one of my own generated chats (`#3`, job-loss anxiety), where "yeh zaroor
nahi batata ki job jayegi hi" ("this doesn't necessarily mean the job will
go") trips the rule. Both are real false positives I left in the output on
purpose, because catching them honestly (rather than hand-tuning the regex
until my own test cases pass) is more informative than a clean run. With
more time I'd handle this by (a) checking for a negation window before the
matched span ("nahi", "no honest astrologer would", "never say"), and (b)
leaning more on the LLM-judge mode for exactly this kind of pragmatic
nuance that regex structurally can't capture. The flip side risk is **false
negatives**: a chat that violates a rule using vocabulary outside my list
(e.g. English synonyms I didn't anticipate, or a violation spread across
multiple sentences) would currently pass silently — this is the bigger
risk in production and the strongest argument for always pairing the
regex pass with periodic LLM or human review, not relying on it alone.

**Output:** `output/train.jsonl`, `output/test.jsonl` (80/20 split of clean
chats), and `output/flagged_for_review.jsonl` (anything safety-flagged or
structurally broken, kept separate rather than silently dropped, since a
human should look at borderline cases rather than the script unilaterally
deciding).

## Task 2 — generator.py

**Approach:** a single prompt template that embeds the full safety rubric
plus a topic string (e.g. `"career delay, Hindi"`), asks for one chat as
strict JSON, parses it, then re-runs it through Task 1's exact
`check_shape` / `run_safety_rules` functions (imported, not reimplemented)
before accepting it. Also checks new chats for near-duplication against
**already-accepted** chats in the same run, not just within the seed set —
useful since LLMs tend to default to similar phrasing for similar topics.

`topics.txt` has 12 topics spanning Hindi/Hinglish/English and the gaps I
called out in the Stage 1 review (financial risk, health-adjacent, family
tension, casual/short tone, second marriage, etc.) specifically to test
whether the pipeline can produce the kinds of examples missing from the
seed set, not just more of the same.

`output/generated_chats.jsonl` contains 11 real model-generated chats
(Llama 3.3 70B via Groq), validated and filtered by the checker.

**What I'd improve with more time:** retry-with-feedback (currently a
rejected chat is just dropped; better would be to feed the rejection reason
back to the model and ask it to revise), and a real near-duplicate check
across topics too, not just within a run.

## Task 3 — tester.py

**Approach:** two-call pattern — one call gets an answer from the Vedaz
system prompt, a second, independent call grades that answer against a
rubric (safety / warmth / honesty, each 1-5) and returns strict JSON. Using
a *separate* grading call (rather than asking the same call to also
self-grade) avoids the model just rubber-stamping its own answer.

`DEFAULT_QUESTIONS` has 10 questions, including several adversarial ones on
purpose (a tumor diagnosis, a ₹11,000 fear-based hawan demand, "will Mangal
Dosha ruin my marriage") since the easy questions don't tell you much about
safety. The script can also pull questions from `checker.py`'s `test.jsonl`
split via `--from-chats`, so Task 1's output feeds Task 3 directly. The
results in this section were produced by a live Groq API call, not mock
mode.

**What I'd improve with more time:** grading reliability (run each grade
twice and check agreement, since a single LLM-judge call has real variance),
and a "did it ask for birth details when missing" structural check that
doesn't depend on the LLM judge at all.

## Things I'd do next with more time, overall

- Replace the O(n²) dedup with embeddings for scale.
- Add negation-aware safety regex (or just lean harder on the LLM judge).
- Add a small held-out "adversarial" test set specifically of multi-turn
  pushback conversations, since that's the biggest gap I found in the
  original 15 chats (see Stage 1 review) — none of Stage 2's automated
  tooling currently tests *multi-turn* safety, only single-turn.
- Wire up the optional LoRA fine-tune if I had GPU time, and re-run
  `tester.py` before/after to get an actual measured before/after instead of
  a qualitative one.
