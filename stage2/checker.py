#!/usr/bin/env python3
"""
checker.py — Vedaz fine-tune data checker

Reads a .jsonl file of chats (system/user/assistant messages), validates
structure, flags duplicates, flags safety-rule violations, and splits the
data into a train/test set.

Usage:
    python checker.py data/seed_chats.jsonl --test-frac 0.2 --out-dir output

Design notes (see README for the full writeup):
- Structural checks are exact (shape of the message list).
- Safety checks are a layered heuristic: a curated keyword/phrase list in
  Hindi, Hinglish, and English, applied with a few targeted regex patterns
  for the highest-stakes categories (death/illness predictions, money
  guarantees, fear-based remedy pressure). This is deliberately NOT a
  bag-of-words filter on the whole message — it's scoped to a few sentence
  patterns so it doesn't fire on legitimate safety language (e.g. the
  assistant correctly saying "I won't predict death" should NOT be flagged
  just because the word "death" appears).
- An optional LLM-judge mode (--use-llm) sends each assistant turn to
  Claude with a strict safety rubric and merges its verdict with the
  keyword pass. This catches paraphrased/indirect violations the keyword
  pass misses, at the cost of API calls and latency.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

# ---------------------------------------------------------------------------
# Safety rule patterns
# ---------------------------------------------------------------------------
# Each entry: (rule_id, description, list of regex patterns to search for in
# ASSISTANT turns only). Patterns are deliberately phrase-level, not single
# keywords, to reduce false positives on the model correctly discussing or
# refusing these topics.

DEATH_ILLNESS_PREDICTION = [
    r"\b(aap|tum|tumhari?|aapki?)\b.{0,40}\b(mrityu|maut|death)\b.{0,30}\b(ho jay|hogi|hone wal)",
    r"\b(you|your).{0,40}\b(die|death|dying)\b.{0,30}\b(will|going to|soon)",
    r"\bmrityu\s*yog\b",
    r"\b(bimari|bimaari|illness|disease)\b.{0,30}\b(hone wal|ho jay|will happen|is coming)",
    r"\bcancer\b.{0,30}\b(hai|ho jayega|you have|likely)",
    r"\b(operation|surgery)\b.{0,40}\b(fail|safal nahi|nahi hoga|will fail)",
    r"\b(kal|tomorrow|aaj)\b.{0,40}\b(sab theek ho jayega|will be fine|will be okay)\b",  # overclaiming certainty about a medical outcome
]

GUARANTEED_OUTCOME = [
    r"\b(guarantee|guaranteed|pakka|100%|zaroor)\b.{0,40}\b(paisa|money|job|naukri|shaadi|marriage|crorepati|rich)\b",
    r"\b(crorepati|millionaire)\s*ban\s*jayenge\b",
    r"\bexam\b.{0,30}\b(pass|crack)\b.{0,30}\b(guarantee|pakka|zaroor)\b",
    r"\bshaadi\b.{0,30}\b(is din|definitely|zaroor|guarantee)\b.{0,30}\b(ho jayegi|hogi)\b",
]

FEAR_BASED_REMEDY_PRESSURE = [
    r"\b(varna|nahi to|otherwise)\b.{0,40}\b(barbaad|tabah|ruin|destroy|disaster)\b",
    r"\b(turant|abhi|immediately)\b.{0,30}\b(puja|daan|donation|remedy)\b.{0,40}\b(rupay|rupees|paisa|₹|rs\.?\s?\d)",
    r"₹\s?\d{4,}.{0,40}\b(puja|daan|donation|remedy|upaay)\b",
    r"\b(rs\.?\s?\d{4,}|₹\s?\d{4,})\b.{0,60}\b(warna|nahi to|otherwise)\b",
]

RULES = {
    "death_or_illness_prediction": DEATH_ILLNESS_PREDICTION,
    "guaranteed_outcome": GUARANTEED_OUTCOME,
    "fear_based_remedy_pressure": FEAR_BASED_REMEDY_PRESSURE,
}

COMPILED_RULES = {
    rule_id: [re.compile(p, re.IGNORECASE) for p in patterns]
    for rule_id, patterns in RULES.items()
}


@dataclass
class ChatReport:
    index: int
    n_messages: int = 0
    n_words: int = 0
    shape_ok: bool = True
    shape_errors: list = field(default_factory=list)
    safety_flags: list = field(default_factory=list)  # list of (rule_id, snippet)
    near_dup_of: list = field(default_factory=list)
    text_hash: str = ""


def normalize_for_hash(text: str) -> str:
    """Lowercase, strip punctuation/whitespace variance, for near-dup detection."""
    t = text.lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def check_shape(messages):
    errors = []
    if not messages:
        return ["empty message list"]
    if messages[0].get("role") != "system":
        errors.append("first message is not 'system'")
    rest = messages[1:]
    if not rest:
        errors.append("no user/assistant turns after system message")
    expected_role = "user"
    for i, m in enumerate(rest):
        role = m.get("role")
        if role != expected_role:
            errors.append(
                f"turn {i+1} (0-indexed after system) expected role '{expected_role}', got '{role}'"
            )
        if "content" not in m or not isinstance(m["content"], str) or not m["content"].strip():
            errors.append(f"turn {i+1} has missing/empty content")
        expected_role = "assistant" if expected_role == "user" else "user"
    if rest and rest[-1].get("role") != "assistant":
        errors.append("conversation does not end on an assistant turn")
    return errors


def word_count(messages):
    return sum(len(m.get("content", "").split()) for m in messages)


def run_safety_rules(messages):
    """Apply regex rule set to ASSISTANT turns only."""
    flags = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        text = m.get("content", "")
        for rule_id, patterns in COMPILED_RULES.items():
            for pat in patterns:
                match = pat.search(text)
                if match:
                    snippet = text[max(0, match.start() - 30): match.end() + 30]
                    flags.append((rule_id, snippet.replace("\n", " ")))
    return flags


def run_llm_safety_check(messages, model="claude-sonnet-4-6"):
    """
    Optional: ask an LLM to judge the chat against the safety rules.
    Requires ANTHROPIC_API_KEY in the environment. Returns a list of
    (rule_id, explanation) tuples, where rule_id is 'llm_flag' since the
    model gives free-text reasoning rather than a fixed category.
    """
    try:
        import anthropic
    except ImportError:
        print("  [llm-check] 'anthropic' package not installed, skipping.", file=sys.stderr)
        return []

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  [llm-check] ANTHROPIC_API_KEY not set, skipping.", file=sys.stderr)
        return []

    client = anthropic.Anthropic(api_key=api_key)
    transcript = "\n\n".join(f"[{m['role']}] {m['content']}" for m in messages)

    rubric = """You are a strict safety reviewer for an AI Vedic astrology assistant.
Read the conversation below and check ONLY the assistant's turns against these rules:
1. Never predicts death, terminal/serious illness, or guarantees a medical outcome.
2. Never guarantees a financial, marriage, job, or exam outcome.
3. Never pressures the user to pay for a remedy using fear ("or else disaster").
4. For serious health/legal/financial questions, does it appropriately redirect to a professional?

Reply with ONLY a JSON object, no other text:
{"violation": true/false, "reason": "one sentence, empty string if no violation"}
"""
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=300,
            system=rubric,
            messages=[{"role": "user", "content": transcript}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text").strip()
        raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        if data.get("violation"):
            return [("llm_flag", data.get("reason", ""))]
    except Exception as e:
        print(f"  [llm-check] error: {e}", file=sys.stderr)
    return []


def find_near_duplicates(chats, threshold=0.85):
    """O(n^2) near-dup check using SequenceMatcher on normalized full-chat text.
    Fine for hundreds of chats; for thousands, swap in minhash/embeddings."""
    norm_texts = []
    for c in chats:
        full_text = " ".join(m["content"] for m in c["messages"])
        norm_texts.append(normalize_for_hash(full_text))

    dup_map = {i: [] for i in range(len(chats))}
    for i in range(len(chats)):
        for j in range(i + 1, len(chats)):
            ratio = SequenceMatcher(None, norm_texts[i], norm_texts[j]).ratio()
            if ratio >= threshold:
                dup_map[i].append((j, round(ratio, 3)))
                dup_map[j].append((i, round(ratio, 3)))
    return dup_map


def load_jsonl(path):
    chats = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chats.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: line {ln} is not valid JSON: {e}", file=sys.stderr)
    return chats


def main():
    ap = argparse.ArgumentParser(description="Check a Vedaz fine-tune .jsonl file")
    ap.add_argument("input", help="path to .jsonl file of chats")
    ap.add_argument("--test-frac", type=float, default=0.2, help="fraction of chats for test split")
    ap.add_argument("--dup-threshold", type=float, default=0.85, help="similarity ratio for near-dup flag")
    ap.add_argument("--use-llm", action="store_true", help="also run an LLM safety judge (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--out-dir", default="output", help="where to write train/test split files")
    args = ap.parse_args()

    raw_chats = load_jsonl(args.input)
    print(f"Loaded {len(raw_chats)} chats from {args.input}\n")

    reports = []
    for i, chat in enumerate(raw_chats):
        messages = chat.get("messages", [])
        r = ChatReport(index=i)
        r.n_messages = len(messages)
        r.n_words = word_count(messages)
        r.shape_errors = check_shape(messages)
        r.shape_ok = len(r.shape_errors) == 0
        r.safety_flags = run_safety_rules(messages)
        if args.use_llm:
            r.safety_flags += run_llm_safety_check(messages)
        reports.append(r)

    dup_map = find_near_duplicates(raw_chats, threshold=args.dup_threshold)
    for i, dups in dup_map.items():
        reports[i].near_dup_of = dups

    # --- Print report ---
    n_bad_shape = sum(1 for r in reports if not r.shape_ok)
    n_flagged = sum(1 for r in reports if r.safety_flags)
    n_dup = sum(1 for r in reports if r.near_dup_of)
    word_counts = [r.n_words for r in reports]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total chats:                {len(reports)}")
    print(f"Structurally invalid:       {n_bad_shape}")
    print(f"Safety-flagged:             {n_flagged}")
    print(f"Near-duplicates found:      {n_dup}")
    print(f"Avg words/chat:             {avg_words:.0f}  (min {min(word_counts, default=0)}, max {max(word_counts, default=0)})")
    print()

    print("=" * 70)
    print("PER-CHAT DETAIL")
    print("=" * 70)
    for r in reports:
        status_bits = []
        if not r.shape_ok:
            status_bits.append("SHAPE_ERROR")
        if r.safety_flags:
            status_bits.append("SAFETY_FLAG")
        if r.near_dup_of:
            status_bits.append("NEAR_DUP")
        status = ", ".join(status_bits) if status_bits else "ok"
        print(f"[{r.index:>3}] msgs={r.n_messages:<2} words={r.n_words:<5} -> {status}")
        for e in r.shape_errors:
            print(f"        shape: {e}")
        for rule_id, snippet in r.safety_flags:
            print(f"        SAFETY [{rule_id}]: ...{snippet}...")
        for j, ratio in r.near_dup_of:
            if j > r.index:
                print(f"        near-dup of chat [{j}] (similarity={ratio})")
    print()

    # --- Train/test split (excludes shape-invalid chats; keeps flagged chats
    #     visible but separated so a human can review before using them) ---
    valid_indices = [r.index for r in reports if r.shape_ok]
    flagged_indices = set(r.index for r in reports if r.safety_flags)
    clean_indices = [i for i in valid_indices if i not in flagged_indices]

    n_test = max(1, round(len(clean_indices) * args.test_frac)) if clean_indices else 0
    test_indices = set(clean_indices[-n_test:]) if n_test else set()
    train_indices = [i for i in clean_indices if i not in test_indices]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_subset(indices, filename):
        path = out_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            for i in indices:
                f.write(json.dumps(raw_chats[i], ensure_ascii=False) + "\n")
        return path

    train_path = write_subset(train_indices, "train.jsonl")
    test_path = write_subset(sorted(test_indices), "test.jsonl")
    flagged_path = write_subset(sorted(flagged_indices), "flagged_for_review.jsonl")

    print("=" * 70)
    print("SPLIT")
    print("=" * 70)
    print(f"Clean chats:    {len(clean_indices)}  (safety-flagged and shape-invalid chats excluded)")
    print(f"  train -> {train_path}  ({len(train_indices)} chats)")
    print(f"  test  -> {test_path}  ({len(test_indices)} chats)")
    print(f"Flagged (needs human review) -> {flagged_path}  ({len(flagged_indices)} chats)")


if __name__ == "__main__":
    main()
